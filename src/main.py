import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from img_process import load_dataset
from models import Discriminator, Generator, init_weights
from plots import plot_loss
from utils import gradient_penalty, listen, apply_transform
import logger



lr = 2e-4
img_size = 64
channels_img = 3
channels_noise = 100
batch_size = 64
num_epochs = 10000
features_d = 128
features_g = 64
critic_it = 5
lambda_gp = 10
p = 0.10


pre_process_transforms = transforms.Compose([
    transforms.Resize(img_size * 3),
    transforms.CenterCrop(img_size * 2),
    transforms.Resize(img_size),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
    ])

random_transforms = transforms.Compose([
    #transforms.RandomAdjustSharpness(0.5, p=p),
    transforms.RandomHorizontalFlip(p=p),
    transforms.RandomErasing(p=p),
    transforms.RandomRotation(degrees=p),
    transforms.RandomPerspective(p=p)
    ])

to_image_transform = transforms.ToPILImage()

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

dataset = load_dataset('real_data', pre_process_transforms)
logger.start_log(f'Dataset Size: {len(dataset)}')
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
fixed_noise = torch.randn((batch_size, channels_noise, 1, 1)).to(device)

net_critic = Discriminator(channels_img, features_d).to(device)
net_gen = Generator(channels_noise, channels_img, features_g).to(device)
init_weights(net_critic)
init_weights(net_gen)

opt_critic = optim.Adam(net_critic.parameters(), lr=lr, betas=(0.5, 0.9))
opt_gen = optim.Adam(net_gen.parameters(), lr=lr, betas=(0.5, 0.9))

list_loss_disc = []
list_loss_gen = []

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

logger.start_log(f'Using: {device}')
logger.start_log(f'Batch Size: {batch_size}')
logger.start_log(f'Learning Rate: {lr}')
logger.start_log(f'Critic Iterations: {critic_it}')
logger.start_log(f'Features Critic: {features_d}')
logger.start_log(f'Features Gen: {features_g}')
train_start_time = logger.start_log('======== TRAINING: WGAN-GP ========')
for epoch in range(1, num_epochs + 1):
    epoch_start_time = logger.start_log(log=False)

    for batch_idx, (real, _) in enumerate(dataloader):
        real = apply_transform(random_transforms, real)
        real = real.to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        for _ in range(critic_it):
            noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
            fake = net_gen(noise)
            fake = apply_transform(random_transforms, fake)
            critic_real = net_critic(real).reshape(-1)
            critic_fake = net_critic(fake).reshape(-1)
            grad_penalty = gradient_penalty(net_critic, real, fake, device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * grad_penalty
            net_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        ### Train Generator: max log(D(G(z)))
        output = net_critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        net_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    list_loss_disc.append(loss_critic.item())
    list_loss_gen.append(loss_gen.item())
    plot_loss(list_loss_disc, list_loss_gen, lr, critic_it)

    if epoch % 1 == 0:
        logger.end_log(epoch_start_time, f'Epoch [{epoch}/{num_epochs}]\
            Loss C: {loss_critic:.2f}, Loss G: {loss_gen:.2f}')

    with torch.no_grad():
        real_grid = torchvision.utils.make_grid(real[:32], normalize=True)
        writer_real.add_image("Real", real_grid, global_step=epoch)

        fake = net_gen(fixed_noise)
        fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
        writer_fake.add_image("Fake", fake_grid, global_step=epoch)
        if epoch % 100 == 0:
            fake_images = to_image_transform(fake_grid)
            fake_images.save('fake_data/fake' + str(epoch) + '.png')

    input = listen()
    if input == 'q\n':
        break

logger.end_log(train_start_time, f'Finished training model\
    Epoch [{epoch}/{num_epochs}]')
torch.save(net_gen.state_dict(), 'model.pt')
