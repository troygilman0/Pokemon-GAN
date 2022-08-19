import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import Generator
from params import *
from PIL import Image


dist.init_process_group("nccl")
rank = dist.get_rank()
device = rank % torch.cuda.device_count()

gen = Generator(CHANNELS_IN, CHANNELS_NOISE).to(device)
gen = DDP(gen, device_ids=[device], find_unused_parameters=True).to(device)
checkpoint = torch.load('/home/troy/Projects/Pokemon-GAN/out/2022-08-16 01:46:59/checkpoints/layer2.pt')
gen.load_state_dict(checkpoint['gen_model'])
gen.to(device)
gen.eval()

torch.manual_seed(0)
fixed_noise = torch.randn((32, CHANNELS_NOISE, 1, 1)).to(device)

with torch.no_grad() and torch.cuda.amp.autocast():
    fake = gen(fixed_noise, 2, 1.0)

fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
fake_images = TO_IMAGE(fake_grid)
fake_images = fake_images.resize((2048, 1024), resample=Image.BOX)
fake_images.save('fake.png')