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
    0: 256,
    1: 256,
    2: 256,
    3: 128,
    4: 32,
    5: 8,
    6: 4
}

LR = 1e-3
IMG_SIZE = 256
CHANNELS_IMG = 3
CHANNELS_NOISE = 256
CHANNELS_IN = 256

PHASE_DURATION = 5000
LAYERS = 6
INIT_LAYER = 0

LAMBDA_GP = 10

P = 0.5

LOAD_CHECKPOINT = None
OUT_DIR = "out/"
SEED = 0

TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), 
    ])

RAND_AUGMENT = [transforms.RandomApply(torch.nn.ModuleList([
    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomResizedCrop(PIXEL_SCALING[layer]),
    ]), p=P) for layer in range(LAYERS+1)]

TO_IMAGE = transforms.ToPILImage()