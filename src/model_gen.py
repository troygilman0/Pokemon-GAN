import torch.nn as nn


class ExpandConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=1, padding=0)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SameConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, img_size, channels_in, reduce_channels=False):
        super().__init__()
        channels_out = int(channels_in / 2) if reduce_channels else channels_in
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.convBlock1 = SameConvBlock(channels_in, channels_out)
        self.convBlock2 = SameConvBlock(channels_out, channels_out)
        self.rgb = nn.ConvTranspose2d(channels_out, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x, to_rgb=False):
        x = self.upsample(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        if to_rgb:
            x = self.rgb(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            ExpandConvBlock(512, 512),
            SameConvBlock(512, 512))
        self.rgb = nn.ConvTranspose2d(512, 3, kernel_size=1, stride=1, padding=0)
        self.layer1 = UpsampleLayer(8, 512)
        self.layer2 = UpsampleLayer(16, 512)
        self.layer3 = UpsampleLayer(32, 512)
        self.layer4 = UpsampleLayer(64, 512, reduce_channels=True)
        self.layer5 = UpsampleLayer(128, 256, reduce_channels=True)
    
    def forward(self, x, layer_idx):
        x = self.layer0(x)
        if (layer_idx == 0):
            x = self.rgb(x)
        if (layer_idx >= 1):
            x = self.layer1(x, to_rgb=(True if layer_idx == 1 else False))
        if (layer_idx >= 2):
            x = self.layer2(x, to_rgb=(True if layer_idx == 2 else False))
        if (layer_idx >= 3):
            x = self.layer3(x, to_rgb=(True if layer_idx == 3 else False))
        if (layer_idx >= 4):
            x = self.layer4(x, to_rgb=(True if layer_idx == 4 else False))
        if (layer_idx >= 5):
            x = self.layer5(x, to_rgb=(True if layer_idx == 5 else False))
        return x
    
