import torch.nn as nn
import torch.nn.functional as F
import torch


factors = [1, 1, 1, 1, 1/2, 1/4]


class Conv(nn.Module):
    def __init__(self, channels_in, channels_out, gain=2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.scale = (gain / (channels_in * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
        return x
        

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, use_pixelnorm=False):
        super().__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = Conv(channels_in, channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = PixelNorm()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.norm(x) if self.use_pn else x
        x = self.relu(self.conv2(x))
        x = self.norm(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, z_dim):
        super().__init__()

        self.inital = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            Conv(in_channels, in_channels),
            nn.LeakyReLU(0.2),
            PixelNorm())
        self.initial_rgb = Conv(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.blocks, self.rgb_layers = nn.ModuleList([None]), nn.ModuleList([self.initial_rgb])
        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            self.blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixelnorm=True))
            self.rgb_layers.append(Conv(conv_out_channels, 3, kernel_size=1, stride=1, padding=0))
    
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1.0 - alpha) * upscaled)

    def forward(self, x, layers, alpha):
        x = self.inital(x)
        if (layers == 0):
            return self.initial_rgb(x)

        for layer_idx in range(1, layers+1):
            upscaled = self.upsample(self.rgb_layers[layer_idx-1](x))
            x = self.blocks[layer_idx](self.upsample(x))

        x = self.rgb_layers[layer_idx](x)
        return self.fade_in(alpha, upscaled, x)
    

class Critic(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.initial_rgb = Conv(3, in_channels, kernel_size=1, stride=1, padding=0)
        self.initial = nn.Sequential(
            Conv(in_channels + 1, in_channels, gain=2),
            nn.LeakyReLU(0.2),
            Conv(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            Conv(in_channels, 1, kernel_size=1, stride=1, padding=0))

        self.blocks, self.rgb_layers = nn.ModuleList([None]), nn.ModuleList([self.initial_rgb])
        self.relu = nn.LeakyReLU(0.2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i+1])
            conv_out_channels = int(in_channels * factors[i])
            self.rgb_layers.append(Conv(3, conv_in_channels, kernel_size=1, stride=1, padding=0))
            self.blocks.append(ConvBlock(conv_in_channels, conv_out_channels))


    def fade_in(self, alpha, downscaled, generated):
        return alpha * generated + (1 - alpha) * downscaled
    
    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1) # 512 -> 513
    
    def forward(self, x, layers, alpha):

        if (layers == 0):
            x = self.relu(self.initial_rgb(x))
            x = self.minibatch_std(x)
            return self.initial(x)

        downscaled = self.relu(self.rgb_layers[layers-1](self.avg_pool(x)))
        x = self.relu(self.rgb_layers[layers](x))
        x = self.blocks[layers](x)
        x = self.avg_pool(x)
        
        x = self.fade_in(alpha, downscaled, x)
        
        for layer_idx in range(layers-1, 0, -1):
            x = self.blocks[layer_idx](x)
            x = self.avg_pool(x)

        x = self.minibatch_std(x)
        x = self.initial(x)
        return x
