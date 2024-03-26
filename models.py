from math import sqrt
from typing import Optional
import torch
from torch import nn, Tensor


def glorot(m, zero_bias = True):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            if zero_bias:
                nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(m.bias)
      

class ContinuousValueEmbedding(nn.Module):

    def __init__(self, 
                 embed_dim: int = 32, 
                 use_glorot: bool = False):
        super().__init__()
        hidden_dim = int(sqrt(embed_dim))
        self.emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(), 
                                 nn.Linear(hidden_dim, embed_dim, bias=False))
        if use_glorot:
            self.emb.apply(glorot)

    def forward(self,
                x: Tensor): # (batch_size x seq_len)
        return self.emb(x.unsqueeze(-1)) # (batch_size x seq_len x embed_dim)

   
class FusionSelfAttention(nn.Module):

    def __init__(self, 
                 embed_dim: int = 32, 
                 hidden_dim: int = 32,
                 use_glorot: bool = False):
        super().__init__()
        self.fuser = nn.Sequential(nn.Linear(embed_dim, hidden_dim), 
                                   nn.Tanh(),
                                   nn.Linear(hidden_dim, 1, bias=False)) # Eq 4
        if use_glorot:
            self.fuser.apply(glorot)

    def forward(self, 
                c: Tensor,  # (batch_size x seq_len x embed_dim)
                mask: Tensor):  # (batch_size x seq_len)
        a = self.fuser(c)
        alpha = torch.exp(a)*mask.unsqueeze(-1)
        alpha = alpha/alpha.sum(dim=1, keepdim=True)
        return (alpha*c).sum(dim=1)  # (batch_size x embed_dim)
        

class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 heads: int,
                 dropout_pbb: float = 0.2,
                 use_glorot: bool = False,
                 norm_first: bool = False,
                 activation: str = 'relu',
                 first_residual_connection: bool = True,
                 second_residual_connection: bool = True):
        super().__init__()
        self.MHA = nn.MultiheadAttention(num_heads=heads,
                                         embed_dim=embed_dim,
                                         batch_first=True,
                                         dropout=dropout_pbb)
        self.activation = nn.ReLU()
        if activation == 'gelu':
            self.activation = nn.GELU()
        self.F = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                               self.activation,
                               nn.Dropout(dropout_pbb),
                               nn.Linear(2*embed_dim, embed_dim),
                               nn.Dropout(dropout_pbb))
        if use_glorot:
            self.F.apply(glorot)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first
        self.first_residual = first_residual_connection
        self.second_residual = second_residual_connection

    def forward(self,
                x: Tensor,
                key_mask: Optional[Tensor] = None,
                is_causal: bool = False,
                need_weights: bool = False):
        attn = None
        if self.norm_first:
            y, attn = self._self_attn(self.norm1(x),
                                      key_mask=key_mask,
                                      is_causal=is_causal,
                                      need_weights=need_weights)
            if self.first_residual:
                y += x
            x = self.F(self.norm2(y))
            if self.second_residual:
                x += y
        else:
            y, attn = self._self_attn(x,
                                      key_mask=key_mask,
                                      is_causal=is_causal,
                                      need_weights=need_weights)
            if self.first_residual:
                y += x
            y = self.norm1(y)
            x = self.F(y)
            if self.second_residual:
                x += y
            x = self.norm2(x)
        return x, attn

    def _self_attn(self,
                   emb: Tensor,
                   key_mask: Optional[Tensor] = None,
                   is_causal: bool = False,
                   need_weights: bool = False):
        emb, attn = self.MHA(query=emb,
                             key=emb,
                             value=emb,
                             key_padding_mask=key_mask,
                             need_weights=need_weights,
                             is_causal=is_causal,
                             average_attn_weights=False)
        return emb, attn


class STraTS(nn.Module):

    def __init__(self, 
                 embed_dim: int=32,
                 num_heads: int=4,
                 num_blocks: int=2,
                 dropout_pbb: float=0.2):
        super().__init__()
        self.cve_time = ContinuousValueEmbedding(embed_dim=embed_dim)
        self.cve_value = ContinuousValueEmbedding(embed_dim=embed_dim)
        self.encoder = None
        if num_blocks > 0:
            encoder = [TransformerEncoderBlock(embed_dim=embed_dim,
                                               heads=num_heads,
                                               dropout_pbb=dropout_pbb) 
                       for _ in range(num_blocks)]
            self.encoder = nn.ModuleList(encoder)

        self.fusion = FusionSelfAttention(embed_dim=embed_dim, 
                                          hidden_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, 2) # Eq. 8 (minus the sigmoid)
        
    def forward(self, data: Tensor, mask: Tensor):
        time, value = data[:, 0, :], data[:, 1, :]
        Et = self.cve_time(time)
        Ev = self.cve_value(value)
        E = Ev + Et 
        if self.encoder is not None:
            for layer in self.encoder:
                E, attn = layer(E, key_mask=~mask)
        C = self.fusion(E, mask)
        return self.classifier(C)


class FSTimeModulator(nn.Module):
    
    def __init__(self, 
                 n_harmonics: int, 
                 embed_dim: int, 
                 T_max: float):
        super().__init__()
        self.T_max = T_max
        self.n_harmonics = n_harmonics
        self.scale_cos = nn.Parameter(torch.randn(n_harmonics, embed_dim))
        self.scale_sin = nn.Parameter(torch.randn(n_harmonics, embed_dim))
        self.bias_cos = nn.Parameter(torch.randn(n_harmonics, embed_dim))
        self.bias_sin = nn.Parameter(torch.randn(n_harmonics, embed_dim))
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        self.register_buffer('k', torch.arange(n_harmonics).reshape(1, 1, -1))
   
    def forward(self, time): # BATCH x SEQLEN X 1
        tf = 2 * torch.pi * self.k * time /self.T_max
        fs_sin, fs_cos = torch.sin(tf), torch.cos(tf)
        scale = torch.matmul(fs_sin, self.scale_sin) + torch.matmul(fs_cos, self.scale_cos)
        bias =  torch.matmul(fs_sin, self.bias_sin)  + torch.matmul(fs_cos, self.bias_cos)
        return scale, bias


class ATAT(nn.Module):

    def __init__(self, 
                 T_max: float, 
                 n_harmonics: int = 16,
                 embed_dim: int = 48, 
                 num_heads: int = 4, 
                 num_blocks: int = 2, 
                 dropout_pbb: int = 0.2):
        super().__init__()
        self.tm = FSTimeModulator(embed_dim=embed_dim, 
                                  n_harmonics=n_harmonics, 
                                  T_max=T_max)
        self.ll = nn.Linear(1, embed_dim)
        self.encoder = None
        if num_blocks > 0:
            encoder = [TransformerEncoderBlock(embed_dim=embed_dim,
                                               heads=num_heads,
                                               dropout_pbb=dropout_pbb) 
                       for _ in range(num_blocks)]
            self.encoder = nn.ModuleList(encoder)
        self.token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.classifier = nn.Linear(embed_dim, 2)

                                        
    def forward(self, 
                data: Tensor, 
                mask: Tensor):
        time, value = data[:, 0, :], data[:, 1, :]
        batch_size = time.shape[0]
        gamma_scale, gamma_bias = self.tm(time.unsqueeze(-1))
        x = self.ll(value.unsqueeze(-1))
        x = x*gamma_scale + gamma_bias
        rep_token = self.token.repeat(batch_size, 1, 1)
        x = torch.cat([rep_token, x], dim=1)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        if self.encoder is not None:
            for layer in self.encoder:
                x, attn = layer(x, key_mask=~mask)
        x = x[:, 0, :]  # Extract token
        return self.classifier(x)
