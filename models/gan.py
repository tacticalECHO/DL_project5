from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, normalize: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InpaintingGenerator(nn.Module):
    """U-Net style generator that conditions on masked image + binary mask."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.down1 = ConvBlock(in_channels, c, normalize=False)
        self.down2 = ConvBlock(c, c * 2)
        self.down3 = ConvBlock(c * 2, c * 4)
        self.down4 = ConvBlock(c * 4, c * 8)
        self.down5 = ConvBlock(c * 8, c * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = UpBlock(c * 8, c * 8, dropout=0.5)
        self.up2 = UpBlock(c * 16, c * 8, dropout=0.5)
        self.up3 = UpBlock(c * 16, c * 4)
        self.up4 = UpBlock(c * 8, c * 2)
        self.up5 = UpBlock(c * 4, c)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bottleneck = self.bottleneck(d5)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        return self.final(torch.cat([u5, d1], dim=1))


class InpaintingDiscriminator(nn.Module):
    """PatchGAN discriminator over image conditioned on masked image + mask."""

    def __init__(self, in_channels: int = 7, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            ConvBlock(in_channels, c, normalize=False),
            ConvBlock(c, c * 2),
            ConvBlock(c * 2, c * 4),
            ConvBlock(c * 4, c * 8, stride=1),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image: torch.Tensor, masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image, masked, mask], dim=1))


def compose_inpainting(prediction: torch.Tensor, masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return prediction * mask + masked * (1.0 - mask)

