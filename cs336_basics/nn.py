import math
import torch
import torch.nn as nn

def softmax(x:torch.Tensor,dim:int)->torch.Tensor:
    x_max=x.max(dim=dim,keepdim=True).values
    e=torch.exp(x-x_max)
    return e/e.sum(dim=dim,keepdim=True)

def silu(x:torch.Tensor)->torch.Tensor:
    return x*torch.sigmoid(x)

class Linear(nn.Module):
    def __init__ (self , in_features:int ,out_features:int ,device=None ,dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=nn.Parameter(
            torch.empty(out_features,in_features,device=device,dtype=dtype)
        )
        self._weight_init()

    def _weight_init(self)->None:
        sigma=math.sqrt(2.0/(self.in_features+self.out_features))
        nn.init.trunc_normal_(self.weight,mean=0.0,std=sigma,a=-3.0*sigma,b=3.0*sigma)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self , num_embeddings:int ,embedding_dim:int,device=None ,dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.weight=nn.Parameter(
            torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3)
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__ (self,d_model,d_ff,device=None,dtype=None):
        super().__init__()
        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.w2(silu(self.w1(x))*self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, token_positions):
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x = x.reshape(*x.shape[:-1], -1, 2)
        x0 = x[..., 0]
        x1 = x[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)
        
def scaled_dot_product_attention(q,k,v,mask=None):
    d_k=q.size(-1)
    scores=q @ k.transpose(-1,-2)/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(~mask,float("-inf"))
    Weights=softmax(scores,dim=-1)
    return Weights@v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model //num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)    

        q=q.reshape(*q.shape[:-1],self.num_heads,self.d_k).transpose(-2,-3)
        k=k.reshape(*k.shape[:-1],self.num_heads,self.d_k).transpose(-2,-3)
        v=v.reshape(*v.shape[:-1],self.num_heads,self.d_k).transpose(-2,-3)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        seq_len=q.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))

        out=scaled_dot_product_attention(q,k,v,causal_mask)
        out = out.transpose(-2, -3)  # (batch, seq, num_heads, d_k)
        out = out.reshape(*out.shape[:-2], self.d_model)  # (batch, seq, d_model)
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff,
                rope=RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length),
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, indices, token_positions=None):
        x = self.token_embeddings(indices)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)          