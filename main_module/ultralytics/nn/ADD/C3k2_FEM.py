import torch
import torch.nn as nn

class FEM(nn.Module):
    """特征增强模块 (FEM)"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        self.cat = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x_1x1 = self.conv1x1(x)
        branch1_out = self.branch1(x_1x1)
        branch2_out = self.branch2(x_1x1)
        branch3_out = self.branch3(x_1x1)
        cat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        return self.cat(cat_out) + x_1x1

class C3_FEM(nn.Module):
    """C3模块使用FEM模块进行特征提取"""
    def __init__(self, c1, c2, n=1, expansion=0.5):
        super().__init__()
        hidden_channels = max(1, int(c2 * expansion))
        self.conv1 = nn.Conv2d(c1, hidden_channels, 1, 1)
        self.conv2 = nn.Conv2d(c1, hidden_channels, 1, 1)
        self.conv3 = nn.Conv2d(2 * hidden_channels, c2, 1)
        self.m = nn.Sequential(*[FEM(hidden_channels) for _ in range(max(1, n))])
 
    def forward(self, x):
        y1 = self.m(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))

class C3k_FEM(nn.Module):
    """C3k模块使用FEM模块进行特征提取"""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        hidden_channels = max(1, int(c2 * 0.5))
        self.cv1 = nn.Conv2d(c1, hidden_channels, 1, 1)
        self.cv2 = nn.Conv2d(c1, hidden_channels, 1, 1)
        self.cv3 = nn.Conv2d(2 * hidden_channels, c2, 1)
        self.m = nn.Sequential(*[FEM(hidden_channels) for _ in range(max(1, n))])
    
    def forward(self, x):
        processed = self.m(self.cv1(x))
        direct = self.cv2(x)
        return self.cv3(torch.cat((processed, direct), dim=1))

class C2f_FEM(nn.Module):
    """C2f模块使用FEM模块进行特征提取"""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.n = max(1, n)
        
        # 确保e是数字类型
        if not isinstance(e, (int, float)):
            e = 0.5
        
        self.c = max(1, int(c2 * e))
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + self.n) * self.c, c2, 1)
        self.m = nn.ModuleList([FEM(self.c) for _ in range(self.n)])
 
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))

class C3k2_FEM(nn.Module):
    """C3k2模块使用FEM模块进行特征提取"""
    def __init__(self, c1, c2, n=1, e=0.5, use_c3k=False):
        super().__init__()
        self.n = max(1, n)
        
        # 确保e是数字类型
        if not isinstance(e, (int, float)):
            e = 0.5
        
        self.c = max(1, int(c2 * e))
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + self.n) * self.c, c2, 1, 1, bias=False)
        
        if use_c3k:
            self.m = nn.ModuleList([C3k_FEM(self.c, self.c, self.n) for _ in range(self.n)])
        else:
            self.m = nn.ModuleList([FEM(self.c) for _ in range(self.n)])
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))