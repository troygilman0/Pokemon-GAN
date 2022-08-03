import torch
import torchvision
import torchvision.transforms as transforms
from model import Generator
from params import *


device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator(CHANNELS_IN, CHANNELS_NOISE)
gen.load_state_dict(torch.load('/home/troy/Projects/Pokemon-GAN/out/2022-07-31 00:43:56.810497/checkpoints/model-L4.pt'))
gen.to(device)
gen.eval()

#torch.manual_seed(0)
fixed_noise = torch.randn((64, CHANNELS_NOISE, 1, 1)).to(device)

with torch.no_grad() and torch.cuda.amp.autocast():
    fake = gen(fixed_noise, LAYERS, 1.0)

fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
fake_images = TO_IMAGE(fake_grid)
#fake_images = fake_images.resize((1024, 154), resample=Image.Resampling.BOX)
fake_images.save('fake.png')