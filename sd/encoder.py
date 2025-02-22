import torch
from torch import nn
from torch.nn import functional as F
from attention import MultiHeadAttention as MHAttn

class ConvResd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        if in_channels != out_channels:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.conv_skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x += self.conv_skip(skip)
        return x

class ConvAttn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = MHAttn(1, channels)
        self.groupnorm = nn.GroupNorm(32, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(n, h * w, c)
        x = self.attention(x)
        x = x.view(n, h, w, c).permute(0, 3, 1, 2)
        x += skip
        return x

class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, 3, 1, 1),
            ConvResd(128, 128),
            ConvResd(128, 128),
            nn.Conv2d(128, 128, 3, 2, 0),
            ConvResd(128, 256),
            ConvResd(256, 256),
            nn.Conv2d(256, 256, 3, 2, 0),
            ConvResd(256, 512),
            ConvResd(512, 512),
            nn.Conv2d(512, 512, 3, 2, 0),
            ConvResd(512, 512),
            ConvResd(512, 512),
            ConvResd(512, 512),
            ConvAttn(512),
            ConvResd(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 1, 1, 0),
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        pads = {layer: (layer.kernel_size[0] - 1) // 2 for layer in self
                if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1)}
        for layer in self:
            if layer in pads:
                x = F.pad(x, (pads[layer], pads[layer], pads[layer], pads[layer]))
            x = layer(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        stdev = var.sqrt()
        x = mean + stdev * noise
        x *= 0.18215
        return x
