import torch
import torch.nn as nn


class Conv(nn.Module):
    """优化的卷积模块：对小特征图/小通道数跳过归一化"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)  # 小通道用偏置替代归一化
        self.act = nn.SiLU() if act else nn.Identity()
        
        # 仅对大特征图+大通道数使用归一化
        self.use_norm = (k > 1) and (c2 > 16)  # 3x3以上卷积且通道>16才用归一化
        if self.use_norm:
            self.norm = nn.BatchNorm2d(c2)  # 大通道用BatchNorm

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm and x.shape[2] > 1 and x.shape[3] > 1:  # 仅对有效空间特征归一化
            x = self.norm(x)
        return self.act(x)

def autopad(k, p=None):
    return p if p is not None else k // 2

class DFF(nn.Module):
    """优化的动态特征融合模块：规避小通道数"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        self.c = max(16, int(c2 * e))  # 确保中间通道数≥16，避免过小
        self.shortcut = shortcut and c1 == c2
        
        # 特征提取分支（3x3和5x5深度卷积）
        self.conv1 = Conv(c1, self.c, 1, 1)  # 1x1降维（无归一化）
        self.dwconv3 = Conv(self.c, self.c, k, 1, g=self.c)  # 深度卷积（大通道才用归一化）
        self.dwconv5 = Conv(self.c, self.c, 5, 1, g=self.c)  # 深度卷积
        self.conv2 = Conv(2 * self.c, c2, 1, 1)  # 1x1升维
        
        # 注意力分支（通道数从2*c→8→2，避免小通道）
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 2*c, 1, 1]
            Conv(2 * self.c, 8, 1, 1, act=True),  # 中间通道=8（避开2）
            Conv(8, 2, 1, 1, act=False),  # 输出2个权重
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        b1 = self.dwconv3(x)
        b2 = self.dwconv5(x)
        
        # 动态融合（注意力权重）
        attn = self.attn(torch.cat([b1, b2], dim=1))  # [B, 2, 1, 1]
        fused = attn[:, 0:1] * b1 + attn[:, 1:2] * b2
        
        x = self.conv2(torch.cat([fused, x], dim=1))
        return x + residual if self.shortcut else x

class C3_DFF(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        c_ = max(16, int(c2 * expansion))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[DFF(c_, c_, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class C3k_DFF(C3_DFF):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=3):
        super().__init__(c1, c2, n, shortcut)
        c_ = max(16, int(c2 * 0.5))
        self.m = nn.Sequential(*[DFF(c_, c_, shortcut, g, k) for _ in range(n)])


class C2f_DFF(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = max(16, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList([DFF(self.c, self.c, shortcut, g) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DFF(C2f_DFF):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_DFF(self.c, self.c, 2, shortcut, g) if c3k else DFF(self.c, self.c, shortcut, g) 
            for _ in range(n)
        )