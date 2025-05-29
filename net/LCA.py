import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *

# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
  
  
# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        # self.ffn = CAB(dim, num_heads, bias)
        self.ffn = CrossMambaBlock_Mutli_Heads(dim, num_heads=4, bias=bias)

        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
    
class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        # self.ffn = CAB(dim, num_heads, bias=bias)
        self.ffn = CrossMambaBlock_Mutli_Heads(dim, num_heads=4, bias=bias)

        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x

from mamba_ssm import Mamba

class CrossMambaBlock_Mutli_Heads(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, mamba_config=None):
        super(CrossMambaBlock_Mutli_Heads, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # 为每个头初始化一个 Mamba 模块
        self.mambas = nn.ModuleList([
            Mamba(d_model=self.head_dim, **(mamba_config or {}))
            for _ in range(num_heads)
        ])

        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        """
        x: query 特征 (b, c, h, w)
        y: key/value 特征 (b, c, h, w)
        """
        b, c, h, w = x.shape

        # 生成 Query
        q = self.q_dw(self.q_proj(x))  # (b, c, h, w)
        # 生成 Key 和 Value
        kv = self.kv_dw(self.kv_proj(y))  # (b, 2c, h, w)
        k, v = kv.chunk(2, dim=1)

        # 融合 q 和 k 实现交叉增强
        fused = q + k  # (b, c, h, w)

        # 划分为多个头
        fused = rearrange(fused, 'b (head d) h w -> head b (h w) d', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> head b (h w) d', head=self.num_heads)

        # 分头处理（每个头独立的 Mamba 模块）
        out_heads = []
        for i in range(self.num_heads):
            enhanced = self.mambas[i](fused[i])  # (b, h*w, d)
            out_heads.append(enhanced + v[i])  # 加上对应的 value

        # 拼接所有头
        out = torch.cat(out_heads, dim=-1)  # (b, h*w, c)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        # 输出投影
        out = self.output_proj(out)
        return out
    
class CrossMambaBlock(nn.Module):
    def __init__(self, dim, bias=False, mamba_config=None):
        super(CrossMambaBlock, self).__init__()
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.mamba = Mamba(d_model=dim, **(mamba_config or {}))
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        """
        x: query 特征 (b, c, h, w)
        y: key/value 特征 (b, c, h, w)
        """
        b, c, h, w = x.shape

        # 提取 query 和 key/value 特征
        q = self.q_dw(self.q_proj(x))  # (b, c, h, w)
        kv = self.kv_dw(self.kv_proj(y))  # (b, 2c, h, w)
        k, v = kv.chunk(2, dim=1)

        # 融合 query 和 key（交叉增强）
        fused = q + k  # 简单相加（也可尝试 q * k、q - k 等）

        # 将融合后的特征拉平处理为序列
        fused_seq = rearrange(fused, 'b c h w -> b (h w) c')  # (b, hw, c)

        # 送入 Mamba 结构进行时序建模（注意：这里按空间维度当作序列）
        enhanced_seq = self.mamba(fused_seq)  # (b, hw, c)

        # 还原空间结构
        enhanced = rearrange(enhanced_seq, 'b (h w) c -> b c h w', h=h, w=w)

        # 加上 value 信息
        out = enhanced + v  # 融合 value 信息

        # 输出映射
        out = self.output_proj(out)
        return out