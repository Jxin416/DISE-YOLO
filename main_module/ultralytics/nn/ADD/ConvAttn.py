import torch
import torch.nn as nn
import torch.nn.functional as F

class DEConv(nn.Module):
    """细节增强卷积 - 修复尺寸匹配问题"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.k = k
        self.s = s
        self.g = g
        self.d = d
        self.p = self.autopad(k, p, d)
        
        # 1x1卷积直接使用标准实现
        if k == 1:
            self.conv = nn.Conv2d(c1, c2, 1, s, 0, groups=g, dilation=d, bias=True)
            return
        
        # 计算输出尺寸
        self.cd = Conv2d_cd(c1, c2, k)
        self.hd = Conv2d_hd(c1, c2, k)
        self.vd = Conv2d_vd(c1, c2, k)
        self.ad = Conv2d_ad(c1, c2, k)
        self.std = nn.Conv2d(c1, c2, k, 1, self.p, groups=g, dilation=d, bias=True)
        
        # 冻结特殊卷积权重
        for m in [self.cd, self.hd, self.vd, self.ad]:
            for param in m.parameters():
                param.requires_grad = False
            
    def autopad(self, k, p=None, d=1):
        """自动计算填充值，保持尺寸一致"""
        if d > 1:
            k = d * (k - 1) + 1
        if p is None:
            p = k // 2
        return p

    def forward(self, x):
        # 1x1卷积直接处理
        if self.k == 1:
            return self.conv(x)
        
        # 计算输入尺寸
        batch_size, in_channels, h, w = x.shape
        
        # 计算正确输出尺寸（确保能被s整除）
        out_h = (h + 2 * self.p - self.d * (self.k - 1) - 1) // self.s + 1
        out_w = (w + 2 * self.p - self.d * (self.k - 1) - 1) // self.s + 1
        
        # 尺寸保护机制
        if out_h <= 0 or out_w <= 0 or h <= self.k or w <= self.k:
            return self.std(x)
        
        # 收集并合并权重
        weights = [m.get_weight()[0] for m in [self.cd, self.hd, self.vd, self.ad]]
        weights.append(self.std.weight)
        total_weight = torch.stack(weights).sum(0).to(x.device)
        
        # 收集并合并偏置
        biases = [m.get_weight()[1] for m in [self.cd, self.hd, self.vd, self.ad]]
        biases.append(self.std.bias)
        total_bias = torch.stack(biases).sum(0).to(x.device)
        
        # 执行卷积
        return F.conv2d(x, total_weight, total_bias, 
                        stride=self.s, padding=self.p, 
                        dilation=self.d, groups=self.g)

# 辅助卷积组件
class Conv2d_base(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=True)
    
    def get_weight(self):
        return self.conv.weight.data, self.conv.bias.data

class Conv2d_cd(Conv2d_base):
    """中心差分卷积"""
    def get_weight(self):
        w = self.conv.weight.data
        if self.k == 1:
            return w, self.conv.bias.data
            
        # 中心差分处理
        flat_size = w.size(2) * w.size(3)
        reshaped = w.view(w.size(0), w.size(1), flat_size)
        center_idx = flat_size // 2
        
        weight_diff = reshaped.clone()
        weight_diff[:, :, center_idx] = reshaped[:, :, center_idx] - reshaped.sum(2)
        
        return weight_diff.view_as(w), self.conv.bias.data

class Conv2d_ad(Conv2d_base):
    """角度差分卷积"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size)
        if kernel_size > 1:
            # 预计算角度索引
            k = kernel_size
            indices = torch.arange(k*k).view(k, k)
            self.register_buffer('ad_indices', indices.flip([0, 1]).view(-1))
    
    def get_weight(self):
        w = self.conv.weight.data
        if self.k == 1:
            return w, self.conv.bias.data
            
        # 应用角度差分
        flat_size = w.size(2) * w.size(3)
        reshaped = w.view(w.size(0), w.size(1), flat_size)
        weight_ad = reshaped - reshaped.index_select(2, self.ad_indices)
        
        return weight_ad.view_as(w), self.conv.bias.data

class Conv2d_hd(Conv2d_base):
    """水平差分卷积"""
    def get_weight(self):
        w = self.conv.weight.data
        if self.k == 1:
            return w, self.conv.bias.data
            
        # 创建水平卷积核
        weight_hd = torch.zeros_like(w)
        center_row = self.k // 2
        weight_hd[:, :, center_row:center_row+1, :] = w.mean(dim=2, keepdim=True)
        
        return weight_hd, self.conv.bias.data

class Conv2d_vd(Conv2d_base):
    """垂直差分卷积"""
    def get_weight(self):
        w = self.conv.weight.data
        if self.k == 1:
            return w, self.conv.bias.data
            
        # 创建垂直卷积核
        weight_vd = torch.zeros_like(w)
        center_col = self.k // 2
        weight_vd[:, :, :, center_col:center_col+1] = w.mean(dim=3, keepdim=True)
        
        return weight_vd, self.conv.bias.data