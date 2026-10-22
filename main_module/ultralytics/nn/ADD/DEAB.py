import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
 
    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)  # 修复：使用 torch.cat
        return self.sa(x2)
 
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.Sigmoid()  # 添加sigmoid激活
        )
 
    def forward(self, x):
        return self.ca(self.gap(x))
 
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        return self.sigmoid(self.pa2(x2))

class DEABlock(nn.Module):
    def __init__(self, dim, kernel_size, reduction=8):
        super().__init__()
        self.conv1 = Conv(dim, dim, kernel_size)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = Conv(dim, dim, kernel_size)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        # 初始特征变换
        res = self.conv1(x)
        res = self.act1(res)
        
        # 短连接
        res = res + x
        
        # 二次特征变换
        res = self.conv2(res)
        
        # 注意力机制
        sattn = self.sa(res)  # [B,1,H,W]
        cattn = self.ca(res)  # [B,C,1,1]
        
        # 扩展通道注意力维度
        cattn = cattn.expand_as(res)  # [B,C,H,W]
        
        # 组合注意力
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        
        # 应用像素注意力
        res = res * pattn2
        
        # 最终短连接
        return res + x