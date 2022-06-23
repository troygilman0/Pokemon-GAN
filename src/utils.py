import torch
import sys
import select

def gradient_penalty(critic, real, fake, layer_idx, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interp_imgs = real * epsilon + fake * (1 - epsilon)
    mixed_scores = critic(interp_imgs, layer_idx)
    gradient = torch.autograd.grad(
        inputs=interp_imgs,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
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