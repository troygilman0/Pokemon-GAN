import torch
import torchvision
import torchvision.transforms as transforms
from model import Generator
from params import *
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator(CHANNELS_IN, CHANNELS_NOISE)
gen.load_state_dict(torch.load('/home/troy/Projects/Pokemon-GAN/out/2022-08-03 00:32:40.417722/checkpoints/model-L3.pt'))
gen.to(device)
gen.eval()

torch.manual_seed(0)
fixed_noise = torch.randn((64, CHANNELS_NOISE, 1, 1)).to(device)

with torch.no_grad() and torch.cuda.amp.autocast():
    fake = gen(fixed_noise, LAYERS, 1.0)

fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
fake_images = TO_IMAGE(fake_grid)
fake_images = fake_images.resize((2048, 1024), resample=Image.Resampling.BOX)
fake_images.save('fake.png')