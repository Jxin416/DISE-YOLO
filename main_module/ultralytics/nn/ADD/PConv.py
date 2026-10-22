import torch
import torch.nn as nn
import torch.nn.functional as F

class PConv(nn.Module):
    """完全兼容YOLOv11的PConv实现"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        
        # 自动计算padding（如果未提供）
        if padding is None:
            padding = kernel_size // 2
            
        # 记录参数供forward使用
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 计算活动通道数（仅对部分通道计算）
        self.active_channels = max(1, int(in_channels * 0.25))
        
        # 标准卷积，仅对部分通道计算
        self.conv = nn.Conv2d(
            in_channels=self.active_channels,
            out_channels=self.active_channels,  # 保持输入输出通道数相同
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # 1x1卷积调整通道数
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # He初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        
        # 标识空间下采样的位置（用于步长>1的情况）
        self.downsample = stride > 1
        
    def forward(self, x):
        # 如果步长>1，需要在空间维度匹配特征图尺寸
        if self.downsample:
            # 保存原始高度和宽度
            H_in, W_in = x.shape[2], x.shape[3]
            
            # 计算下采样后的高度和宽度
            H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
            
            # 1. 拆分活跃通道
            active_x = x[:, :self.active_channels, :, :]
            inactive_x = x[:, self.active_channels:, :, :]
            
            # 2. 对活跃通道执行卷积（会改变空间尺寸）
            conv_active = self.conv(active_x)
            
            # 3. 对非活跃通道执行平均池化以匹配空间尺寸
            inactive_x_pooled = F.avg_pool2d(inactive_x, kernel_size=self.stride, stride=self.stride)
            
            # 4. 如果尺寸仍然不匹配，进行裁剪或填充
            if inactive_x_pooled.shape[2] != H_out or inactive_x_pooled.shape[3] != W_out:
                inactive_x_pooled = F.interpolate(
                    inactive_x_pooled, size=(H_out, W_out), mode='nearest'
                )
            
            # 5. 拼接特征图
            x = torch.cat([conv_active, inactive_x_pooled], dim=1)
        else:
            # 当stride=1时，空间尺寸不变
            active_x = x[:, :self.active_channels, :, :]
            inactive_x = x[:, self.active_channels:, :, :]
            
            conv_active = self.conv(active_x)
            x = torch.cat([conv_active, inactive_x], dim=1)
        
        # 用1x1卷积调整通道数
        return self.conv1x1(x)