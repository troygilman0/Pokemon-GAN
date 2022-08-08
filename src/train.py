from tabnanny import check
from threading import local
from tracemalloc import start
from matplotlib.pyplot import step
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from tqdm import tqdm
from PIL import Image

from img_process import load_dataset
from model import Generator, Critic
from plots import plot_loss, plot_fid
from utils import gradient_penalty, calc_fid
import logger
import os
from params import *




SCALAR = torch.cuda.amp.GradScaler()



LOSS_GEN = []
LOSS_CRITIC = []
FID = []


def scale_real(real, layer, alpha):
    scale_transform = transforms.Resize(PIXEL_SCALING[layer])
    prev_scale_transform = transforms.Resize(PIXEL_SCALING[max(layer-1, 0)])
    real_scaled = scale_transform(real)
    real_scaled_prev = prev_scale_transform(real)
    if layer > 0:
        real_scaled_prev = F.interpolate(real_scaled_prev, scale_factor=2, mode="nearest")
    real = alpha * real_scaled + (1.0 - alpha) * real_scaled_prev
    return real


def log_results(gen, real, fixed_noise, layer, alpha):
    global step, session_dir, writer_real, writer_fake
    with torch.no_grad() and torch.cuda.amp.autocast():
        fake = gen(fixed_noise[:8], layer, alpha)
    #real = RAND_AUGMENT[layer](real)
    real_grid = torchvision.utils.make_grid(real[:8], normalize=True, padding=0)
    fake_grid = torchvision.utils.make_grid(fake[:8], normalize=True, padding=0)
    writer_real.add_image("Real", real_grid, global_step=step)
    writer_fake.add_image("Fake", fake_grid, global_step=step)

    FID.append(calc_fid(real, fake))

    if step % (PHASE_DURATION // 10) == 0:
        fake_images = TO_IMAGE(fake_grid)
        fake_images = fake_images.resize((2048, 256), resample=Image.BOX)
        fake_images.save(session_dir + '/fake_data/fake-' + str(step) + '.png')
        plot_loss(session_dir, LOSS_GEN, LOSS_CRITIC)
        plot_fid(session_dir, FID)
        
    step += 1


def train_models(gen, critic, opt_gen, opt_critic, real, layer, alpha, batch_size, device):
    ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
    noise = torch.randn(batch_size, CHANNELS_NOISE, 1, 1).to(device)
    with torch.cuda.amp.autocast():
        fake = gen(noise, layer, alpha)
        augmented = RAND_AUGMENT[layer](torch.concat([real, fake]))
        real, fake = augmented[:batch_size], augmented[batch_size:]
        #print(real.shape, fake.shape)
        critic_real = critic(real, layer, alpha).reshape(-1)
        critic_fake = critic(fake, layer, alpha).reshape(-1)
        grad_penalty = gradient_penalty(critic, real, fake, layer, alpha, device)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake)) 
            + LAMBDA_GP * grad_penalty 
            + (0.001 * torch.mean(critic_real ** 2))
        )
    critic.zero_grad()
    SCALAR.scale(loss_critic).backward(retain_graph=True)
    SCALAR.step(opt_critic)
    SCALAR.update()

    ### Train Generator: max log(D(G(z)))
    with torch.cuda.amp.autocast():
        output = critic(fake, layer, alpha).reshape(-1)
        loss_gen = -torch.mean(output)
    gen.zero_grad()
    SCALAR.scale(loss_gen).backward()
    SCALAR.step(opt_gen)
    SCALAR.update()

    return loss_gen.detach().item(), loss_critic.detach().item()


def train_epoch(gen, critic, opt_gen, opt_critic, epoch, dataloader, layer, alpha, fixed_noise, device, start_time):
    epoch_start_time = logger.start_log(log=False)
    dataloader.sampler.set_epoch(epoch)

    for _, (real) in enumerate(dataloader):
        real = real.to(device)
        real = scale_real(real, layer, alpha)
        batch_size = real.shape[0]
        loss_gen, loss_critic = train_models(gen, critic, opt_gen, opt_critic, real, layer, alpha, batch_size, device)
    
    LOSS_GEN.append(loss_gen)
    LOSS_CRITIC.append(loss_critic)

    log_results(gen, real, fixed_noise, layer, alpha)
    torch.cuda.empty_cache()
    logger.end_log(epoch_start_time, f'Epoch [{epoch}/{PHASE_DURATION}]\
        Loss C: {LOSS_CRITIC[-1]:.2f}, Loss G: {LOSS_GEN[-1]:.2f}\
        Layers: {layer}, Fade: {alpha:.2f}')


def train_layer(gen, critic, opt_gen, opt_critic, dataset, layer, fixed_noise, device, start_time):
    batch_size = BATCH_SIZES[layer]
    sampler = DistributedSampler(dataset=dataset, shuffle=True) 
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=1, pin_memory=True)
    alpha = 0

    if layer > 0:
        for epoch in range(1, PHASE_DURATION + 1):
            alpha = torch.tensor(float(epoch / PHASE_DURATION)) # linear
            train_epoch(gen, critic, opt_gen, opt_critic, epoch, dataloader, layer, alpha, fixed_noise, device, start_time)

    for epoch in range(1, PHASE_DURATION + 1):
        train_epoch(gen, critic, opt_gen, opt_critic, epoch, dataloader, layer, alpha, fixed_noise, device, start_time)

    torch.save({
        'gen_model': gen.state_dict(),
        'critic_model': critic.state_dict(),
        'gen_opt': opt_gen.state_dict(),
        'critic_opt': opt_critic.state_dict()
    }, session_dir + '/checkpoints/layer' + str(layer) + '.pt')


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    dataset = load_dataset('real_data', TRANSFORMS)
    
    gen = Generator(CHANNELS_IN, CHANNELS_NOISE).to(device)
    critic = Critic(CHANNELS_IN).to(device)

    gen_params = sum(param.numel() for param in gen.parameters())
    logger.start_log(f'Generator was created with {gen_params} parameters')
    crtitic_params = sum(param.numel() for param in critic.parameters())
    logger.start_log(f'Critic was created with {crtitic_params} parameters')

    if (LOAD_CHECKPOINT):
        checkpoint = torch.load(LOAD_CHECKPOINT)
        gen.load_state_dict(checkpoint['gen_model'])
        critic.load_state_dict(checkpoint['critic_model'])
        opt_gen.load_state_dict(checkpoint['gen_opt'])
        opt_critic.load_state_dict(checkpoint['critic_opt'])

    gen = DDP(gen, device_ids=[device], find_unused_parameters=True).to(device)
    critic = DDP(critic, device_ids=[device], find_unused_parameters=True).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))
        
    torch.manual_seed(SEED)
    fixed_noise = torch.randn((32, CHANNELS_NOISE, 1, 1)).to(device)

    global step, session_dir, writer_fake, writer_real

    step = 1

    start_time = logger.start_log('======== TRAINING: PRO-GAN ========')

    session_dir = os.path.join(OUT_DIR, str(start_time)[:19])
    try:
        os.mkdir(session_dir)
        os.mkdir(session_dir + "/fake_data")
        os.mkdir(session_dir + "/plots")
        os.mkdir(session_dir + "/checkpoints")
    except:
        pass

    writer_fake = SummaryWriter(session_dir + "/logs/fake")
    writer_real = SummaryWriter(session_dir + "/logs/real")

    dist.barrier()

    layer = INIT_LAYER
    while layer <= LAYERS:
        train_layer(gen, critic, opt_gen, opt_critic, dataset, layer, fixed_noise, device, start_time)
        layer += 1


main()

        