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

from tqdm import tqdm
from PIL import Image

from img_process import load_dataset
from model import Generator, Critic, init_weights
from plots import plot_loss, plot_fid, plot_ada
from utils import gradient_penalty, calc_fid
import logger
import os
from params import *




SCALAR = torch.cuda.amp.GradScaler()



LOSS_GEN = []
LOSS_CRITIC = []
FID = []
RT = []
P = []


def scale_real(real, layer, alpha):
    scale_transform = transforms.Resize(PIXEL_SCALING[layer])
    prev_scale_transform = transforms.Resize(PIXEL_SCALING[max(layer-1, 0)])
    real_scaled = scale_transform(real)
    real_scaled_prev = prev_scale_transform(real)
    if layer > 0:
        real_scaled_prev = F.interpolate(real_scaled_prev, scale_factor=2, mode="nearest")
    real = alpha * real_scaled + (1.0 - alpha) * real_scaled_prev
    return real


def log_results(gen, real, fixed_noise, layer, alpha, loss_gen, loss_critic, fid, rt, p):
    global step, session_dir, writer_real, writer_fake

    LOSS_GEN.append(loss_gen)
    LOSS_CRITIC.append(loss_critic)
    FID.append(fid)
    RT.append(rt)
    P.append(p)

    with torch.no_grad() and torch.cuda.amp.autocast():
        fake = gen(fixed_noise[:8], layer, alpha)
    #real = RAND_AUGMENT[layer](real)
    real_grid = torchvision.utils.make_grid(real[:8], normalize=True, padding=0)
    fake_grid = torchvision.utils.make_grid(fake[:8], normalize=True, padding=0)
    writer_real.add_image("Real", real_grid, global_step=step)
    writer_fake.add_image("Fake", fake_grid, global_step=step)

    if step % (PHASE_DURATION // 20) == 0:
        fake_images = TO_IMAGE(fake_grid)
        fake_images = fake_images.resize((2048, 256), resample=Image.BOX)
        fake_images.save(session_dir + '/fake_data/fake-' + str(step) + '.png')
        plot_loss(session_dir, LOSS_GEN, LOSS_CRITIC)
        plot_fid(session_dir, FID)
        plot_ada(session_dir, RT, P)
        
    step += 1


def train_models(gen, critic, opt_gen, opt_critic, real, layer, alpha, batch_size, rand_transforms, device):
    noise = torch.randn(batch_size, CHANNELS_NOISE, 1, 1).to(device)

    ### Train Discriminator: max log(D(real)) + log(1 - D(G(fake)))
    with torch.cuda.amp.autocast():
        fake = gen(noise, layer, alpha)
        augmented = rand_transforms(torch.concat([real, fake]))
        real, fake = augmented[:batch_size], augmented[batch_size:]
        critic_real = critic(real, layer, alpha).reshape(-1)
        critic_fake = critic(fake, layer, alpha).reshape(-1)
        grad_penalty = gradient_penalty(critic, real, fake, layer, alpha, device)
        #loss_critic = (
        #    -(torch.mean(critic_real) - torch.mean(critic_fake)) 
        #    + LAMBDA_GP * grad_penalty 
        #    + (0.001 * torch.mean(critic_real ** 2))
        #)
        loss_d_real = BCE_LOSS(critic_real, torch.ones_like(critic_real))
        loss_d_fake = BCE_LOSS(critic_fake, torch.zeros_like(critic_fake))
        loss_critic = (loss_d_real + loss_d_fake) / 2
    critic.zero_grad()
    SCALAR.scale(loss_critic).backward(retain_graph=True)
    SCALAR.step(opt_critic)
    SCALAR.update()

    ### Train Generator: max log(D(G(z)))
    with torch.cuda.amp.autocast():
        output = critic(fake, layer, alpha).reshape(-1)
        loss_gen = BCE_LOSS(output, torch.ones_like(output))
    gen.zero_grad()
    SCALAR.scale(loss_gen).backward()
    SCALAR.step(opt_gen)
    SCALAR.update()

    fake = fake.cpu()
    return fake, critic_real, loss_gen.detach().item(), loss_critic.detach().item()


def train_epoch(gen, critic, opt_gen, opt_critic, epoch, dataloader, layer, alpha, fixed_noise, device, start_time):
    global rand_p
    epoch_start_time = logger.start_log(log=False)
    dataloader.sampler.set_epoch(epoch)

    all_real = []
    all_fake = []
    all_d_train = []

    rand_transforms = get_rand_transform(layer, rand_p)

    for _, (real) in enumerate(dataloader):
        real = scale_real(real, layer, alpha)
        all_real.append(real)
        real = real.to(device)
        batch_size = real.shape[0]
        fake, d_train, loss_gen, loss_critic = train_models(gen, critic, opt_gen, opt_critic, real, layer, alpha, batch_size, rand_transforms, device)
        all_fake.append(fake)
        all_d_train.append(d_train)

    all_real = torch.cat(all_real)
    all_fake = torch.cat(all_fake)

    rt = torch.mean(torch.sin(torch.cat(all_d_train))).item()
    if (rt > TARGET_RT):
        rand_p += P_INCREMENT
    elif (rt < TARGET_RT):
        rand_p -= P_INCREMENT
    rand_p = max(0.0, rand_p)
    rand_p = min(0.8, rand_p)
    
    if (device == 0):
        fid = 0#calc_fid(all_real, all_fake)
        log_results(gen, real, fixed_noise, layer, alpha, loss_gen, loss_critic, fid, rt, rand_p)
        logger.end_log(epoch_start_time, f'Epoch [{epoch}/{PHASE_DURATION}]\
            Loss C: {LOSS_CRITIC[-1]:.2f}, Loss G: {LOSS_GEN[-1]:.2f}\
            rt: {rt:.2f}, p: {rand_p:0.2f}\
            Layers: {layer}, Fade: {alpha:.2f}')

    torch.cuda.empty_cache()


def train_layer(gen, critic, opt_gen, opt_critic, dataset, layer, fixed_noise, device, start_time):
    batch_size = BATCH_SIZES[layer]
    sampler = DistributedSampler(dataset=dataset, shuffle=True) 
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
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
    logger.start_log(f'Using device {device}')

    torch.manual_seed(SEED)
    
    gen = Generator(CHANNELS_IN, CHANNELS_NOISE).to(device)
    critic = Critic(CHANNELS_IN).to(device)
    init_weights(gen)
    init_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LR_DISC, betas=(0.0, 0.9))

    gen = DDP(gen, device_ids=[device], find_unused_parameters=True).to(device)
    critic = DDP(critic, device_ids=[device], find_unused_parameters=True).to(device)

    if (LOAD_CHECKPOINT):
        checkpoint = torch.load(LOAD_CHECKPOINT)
        gen.load_state_dict(checkpoint['gen_model'])
        critic.load_state_dict(checkpoint['critic_model'])
        opt_gen.load_state_dict(checkpoint['gen_opt'])
        opt_critic.load_state_dict(checkpoint['critic_opt'])
        gen.to(device)
        critic.to(device)

    gen_params = sum(param.numel() for param in gen.parameters())
    logger.start_log(f'Generator was created with {gen_params} parameters')
    crtitic_params = sum(param.numel() for param in critic.parameters())
    logger.start_log(f'Critic was created with {crtitic_params} parameters')

    dataset = load_dataset('real_data', TRANSFORMS)
        
    fixed_noise = torch.randn((32, CHANNELS_NOISE, 1, 1)).to(device)

    global step, rand_p, session_dir, writer_fake, writer_real

    dist.barrier()
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

    step = 1
    rand_p = 0
    layer = INIT_LAYER
    while layer <= LAYERS:
        train_layer(gen, critic, opt_gen, opt_critic, dataset, layer, fixed_noise, device, start_time)
        layer += 1


main()

        