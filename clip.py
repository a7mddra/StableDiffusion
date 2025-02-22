import torch
from torch import nn
from torch.nn import functional as F
from attention import MultiHeadAttention as MHAttn

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, d_model: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.position_embedding = nn.Parameter(torch.randn((n_token, d_model)) * 0.02)
    
    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        x = self.token_embedding(tokens)
        x = x + self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.attention = MHAttn(n_heads, d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += skip
        skip = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x *= torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x += skip
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output
