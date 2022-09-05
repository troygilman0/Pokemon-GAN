from configparser import Interpolation
import torchvision.transforms as transforms
import torch

PIXEL_SCALING = {
    0: 4,
    1: 8,
    2: 16,
    3: 32,
    4: 64,
    5: 128,
    6: 256
}

BATCH_SIZES = {
    0: 32,
    1: 32,
    2: 32,
    3: 32,
    4: 32,
    5: 32, #8,
    6: 16, #4
}

LR_GEN = 5e-3
LR_DISC = 2e-5
IMG_SIZE = 256
CHANNELS_IMG = 3
CHANNELS_NOISE = 256
CHANNELS_IN = 256

PHASE_DURATION = 2000
LAYERS = 6
INIT_LAYER = 0

LAMBDA_GP = 10
BCE_LOSS = torch.nn.BCEWithLogitsLoss()

TARGET_RT = 0.4
P_INCREMENT = 0.001


LOAD_CHECKPOINT = None
OUT_DIR = "out/"
SEED = 0

TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), 
    ])


def get_rand_transform(layer, p):
    return transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.8)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(contrast=0.8)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(saturation=0.8)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(hue=0.5)]), p=p),

        transforms.RandomHorizontalFlip(p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(0, translate=(0.125, 0.125), interpolation=transforms.InterpolationMode.BILINEAR)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.BILINEAR)]), p=p),
        
        #transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(PIXEL_SCALING[layer])]), p=p),
        #transforms.RandomErasing(p=p)
        ])


TO_IMAGE = transforms.ToPILImage()