import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([features_d * 2, 32, 32]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([features_d * 4, 16, 16]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([features_d * 8, 8, 8]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(channels_noise, features_g * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),

            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=4, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def init_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)