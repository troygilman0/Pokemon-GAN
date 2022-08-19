import torch
import sys
import select
import numpy as np
from math import factorial


def gradient_penalty(critic, real, fake, layers, alpha, device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interp_imgs = real * epsilon + fake * (1 - epsilon)
    with torch.cuda.amp.autocast():
        mixed_scores = critic(interp_imgs, layers, alpha)
    gradient = torch.autograd.grad(
        inputs=interp_imgs,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.reshape((gradient.shape[0], -1))
    graident_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((graident_norm - 1) ** 2)
    return gradient_penalty

def listen():
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if line:
            return line
        else: # an empty line means stdin has been closed
            exit(0)
    return None

def apply_transform(transform, data):
    new_data = []
    for x in data:
        new_data.append(transform(x))
    new_data = torch.concat(new_data).view((data.shape))
    return new_data

def calc_fid(real, fake):
    size = min(real.shape[0], fake.shape[0])
    real = real[:size]
    fake = fake[:size]
    real = torch.reshape(real, (size, -1))
    fake = torch.reshape(fake, (size, -1))
    mu_real, sigma_real = torch.mean(real, axis=0), torch.cov(real)
    mu_fake, sigma_fake = torch.mean(fake, axis=0), torch.cov(fake)
    ss_diff = torch.sum((mu_real - mu_fake) ** 2.0)
    cov_mean = torch.sqrt(torch.mm(sigma_real, sigma_fake))
    cov_mean = torch.nan_to_num(cov_mean, nan=-1.0)
    if torch.is_complex(cov_mean):
        cov_mean = cov_mean.real
    fid = ss_diff + torch.trace(sigma_real + sigma_fake - 2.0 * cov_mean)
    return fid.item()

def normalize(data):
    data_mean = torch.mean(data, dim=0)
    data_var = torch.var(data, dim=0)
    data_norm = (data - data_mean) / torch.sqrt(data_var)
    return data_norm

