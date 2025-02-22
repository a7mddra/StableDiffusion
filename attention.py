import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)  
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scaling = self.d_head ** -0.5
    
    def forward(self, x, causal_mask=False):
        batch_size, seq_len, d_model = x.shape
        multihead_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k, v = (t.view(multihead_shape).transpose(1, 2) for t in (q, k, v))

        weight = (q @ k.transpose(-2, -1)) * self.scaling

        if causal_mask:
            mask = torch.triu(torch.full_like(weight, float('-inf')), diagonal=1)
            weight += mask

        attn = F.softmax(weight, dim=-1)
        output = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_model, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scaling = self.d_head ** -0.5

    def mhs(self, x, y, cross=False):
        batch_size, seq_len_q, _ = x.shape
        seq_len_kv = y.shape[1]
        return (batch_size, seq_len_kv if cross else seq_len_q, self.n_heads, self.d_head)

    def forward(self, x, y):
        batch_size, seq_len_q, _ = x.shape
        q_shape = self.mhs(x, y, False)
        kv_shape = self.mhs(x, y, True)

        q = self.q_proj(x).view(q_shape).transpose(1, 2)
        k = self.k_proj(y).view(kv_shape).transpose(1, 2)
        v = self.v_proj(y).view(kv_shape).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scaling
        attn_probs = F.softmax(attn_weights, dim=-1)
        output = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_len_q, -1)
        return self.out_proj(output)
