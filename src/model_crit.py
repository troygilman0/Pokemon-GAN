import torch.nn as nn

class ContractConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=1, padding=0)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SameConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, img_size, channels_in, increase_channels=False):
        super().__init__()
        channels_out = int(channels_in * 2) if increase_channels else channels_in
        self.rgb = nn.Conv2d(3, channels_in, kernel_size=1, stride=1, padding=0)
        self.convBlock1 = SameConvBlock(channels_in, channels_out)
        self.convBlock2 = SameConvBlock(channels_out, channels_out)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='nearest')

    def forward(self, x, to_rgb=False):
        if to_rgb:
            x = self.rgb(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.downsample(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer5 = DownsampleLayer(128, 128, increase_channels=True)
        self.layer4 = DownsampleLayer(64, 256, increase_channels=True)
        self.layer3 = DownsampleLayer(32, 512)
        self.layer2 = DownsampleLayer(16, 512)
        self.layer1 = DownsampleLayer(8, 512)
        self.rgb = nn.Conv2d(3, 512, kernel_size=1, stride=1, padding=0)
        self.layer0 = nn.Sequential(
            SameConvBlock(512, 512),
            ContractConvBlock(512, 512))
        self.full = nn.Sequential(
            nn.Linear(32768, 512),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, layer_idx):
        if (layer_idx >= 5):
            x = self.layer5(x, to_rgb=(True if layer_idx == 5 else False))
        if (layer_idx >= 4):
            x = self.layer4(x, to_rgb=(True if layer_idx == 4 else False))
        if (layer_idx >= 3):
            x = self.layer3(x, to_rgb=(True if layer_idx == 3 else False))
        if (layer_idx >= 2):
            x = self.layer2(x, to_rgb=(True if layer_idx == 2 else False))
        if (layer_idx >= 1):
            x = self.layer1(x, to_rgb=(True if layer_idx == 1 else False))
        if (layer_idx == 0):
            x = self.rgb(x)
        x = self.layer0(x)
        x = x.view((1, -1))
        self.full(x)
        return x