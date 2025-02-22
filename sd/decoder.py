import torch
from torch import nn
from encoder import ConvAttn, ConvResd

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.Conv2d(4, 512, 3, 1, 1),
            ConvResd(512, 512),
            ConvAttn(512),
            ConvResd(512, 512),
            ConvResd(512, 512),
            ConvResd(512, 512),
            ConvResd(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1),
            ConvResd(512, 512),
            ConvResd(512, 512),
            ConvResd(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1),
            ConvResd(512, 256),
            ConvResd(256, 256),
            ConvResd(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1),
            ConvResd(256, 128),
            ConvResd(128, 128),
            ConvResd(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215
        for layer in self:
            x = layer(x)
        return x
