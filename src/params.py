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
    0: 16,
    1: 16,
    2: 16,
    3: 16,
    4: 4,
    5: 4, #8,
    6: 4, #4
}

CHANNELS_IMG = 3
IMG_SIZE = 256
CHANNELS_NOISE = 256

DISC_IT = 10
LR_GEN = 1e-3
LR_DISC = 1e-5
CHANNELS_IN_GEN = 256
CHANNELS_IN_DISC = 256

PHASE_DURATION = 200
LAYERS = 6
INIT_LAYER = 0

LAMBDA_GP = 10
BCE_LOSS = torch.nn.BCEWithLogitsLoss()

TARGET_RT = 0.4
P_INCREMENT = 0.001
P_MAX = 0.8

RAND_BRIGHTNESS_RANGE = (0.5, 1.5)
RAND_CONTRAST_RANGE = (0.5, 1.5)
RAND_SATURATION_RANGE = (0.5, 1.5)
RAND_HUE_RANGE = (-0.5, 0.5)

RAND_TRANSLATE_RANGE = (-0.125, 0.125)
RAND_ROTATE_RANGE = (-90.0, 90.0)



LOAD_CHECKPOINT = None
OUT_DIR = "out/"
SEED = 1

TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), 
    ])


def get_rand_transform(layer, p):
    return transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.4)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(contrast=0.4)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(saturation=0.4)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(hue=0.4)]), p=p),

        transforms.RandomHorizontalFlip(p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(0, translate=(0.125, 0.125), interpolation=transforms.InterpolationMode.BILINEAR)]), p=p),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.BILINEAR)]), p=p),
        
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(PIXEL_SCALING[layer])]), p=p),
        transforms.RandomErasing(p=p)
        ])


TO_IMAGE = transforms.ToPILImage()