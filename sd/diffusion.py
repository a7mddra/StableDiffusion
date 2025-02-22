import torch
from torch import nn
from torch.nn import functional as F
from attention import CrossAttention as CSAttn, MultiHeadAttention as MHAttn

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear_1 = nn.Linear(1, n_embed*4)
        self.linear_2 = nn.Linear(n_embed*4, n_embed*4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

class ConvResd(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.conv_skip = nn.Identity()
    
    def forward(self, x, time):
        skip = x
        x = self.groupnorm_feature(x)
        x = F.silu(x)
        x = self.conv_feature(x)
        time = self.linear_time(time)
        x += time.unsqueeze(-1).unsqueeze(-1)
        x = self.groupnorm_merged(x)
        x = F.silu(x)
        x = self.conv_merged(x)
        x += self.conv_skip(skip)
        return x

class ConvAttn(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_heads * n_embed
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, 1, 1, 0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = MHAttn(n_heads, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CSAttn(n_heads, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)

        self.linear_geglu_1 = nn.Linear(channels, channels*8)
        self.linear_geglu_2 = nn.Linear(channels*4, channels)

        self.conv_output = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x, context):
        skip_1 = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h*w))
        x = x.transpose(-1, -2)
        skip_2 = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += skip_2
        skip_2 = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += skip_2
        skip_2 = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x *= F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += skip_2
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x = self.conv_output(x)
        x += skip_1
        return x

class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ConvAttn):
                x = layer(x, context)
            elif isinstance(layer, ConvResd):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, 3, 1, 1)),
            SwitchSequential(ConvResd(320, 320), ConvAttn(8, 40)),
            SwitchSequential(ConvResd(320, 320), ConvAttn(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, 3, 2, 1)),
            SwitchSequential(ConvResd(320, 640), ConvAttn(8, 80)),
            SwitchSequential(ConvResd(640, 640), ConvAttn(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, 3, 2, 1)),
            SwitchSequential(ConvResd(640, 1280), ConvAttn(8, 160)),
            SwitchSequential(ConvResd(1280, 1280), ConvAttn(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, 3, 2, 1)),
            SwitchSequential(ConvResd(1280, 1280)),
            SwitchSequential(ConvResd(1280, 1280)),                             
        ])

        self.bottleneck = SwitchSequential(
            ConvResd(1280, 1280),
            ConvAttn(8, 160),
            ConvResd(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(ConvResd(2560, 1280)),
            SwitchSequential(ConvResd(2560, 1280)),
            SwitchSequential(ConvResd(2560, 1280), UpSample(1280)),
            SwitchSequential(ConvResd(2560, 1280), ConvAttn(8, 160)),
            SwitchSequential(ConvResd(2560, 1280), ConvAttn(8, 160)),
            SwitchSequential(ConvResd(1920, 1280), ConvAttn(8, 160), UpSample(1280)),
            SwitchSequential(ConvResd(1920, 640), ConvAttn(8, 80)),
            SwitchSequential(ConvResd(1280, 640), ConvAttn(8, 80)),
            SwitchSequential(ConvResd(960, 640), ConvAttn(8, 80), UpSample(640)),
            SwitchSequential(ConvResd(960, 320), ConvAttn(8, 40)),
            SwitchSequential(ConvResd(640, 320), ConvAttn(8, 40)),
            SwitchSequential(ConvResd(640, 320), ConvAttn(8, 40)),
        ])

    def forward(self, x, context, time):
        skip = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip.append(x)
        x = self.bottleneck(x, context, time)
        for layers in self.decoders:
            x = torch.cat((x, skip.pop()), dim=1)
            x = layers(x, context, time)
        return x

class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutputLayer(320, 4)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        time = self.time_embed(time)
        x = self.unet(x, context, time)
        x = self.final(x)
        return x
