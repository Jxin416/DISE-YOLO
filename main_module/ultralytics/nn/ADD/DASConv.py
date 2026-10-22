import torch
import torch.nn as nn
import torch.nn.functional as F

class DASConv(nn.Module):
    def __init__(self, *args):
        super().__init__()
        
        # 解析参数
        if len(args) < 2:
            raise ValueError(f"DASConv需要至少2个参数，收到{len(args)}个")
        
        # 基本参数
        c1 = args[0]
        c2 = args[1] if len(args) >= 2 else c1
        k = args[2] if len(args) >= 3 else 3
        s = args[3] if len(args) >= 4 else 1
        g = args[4] if len(args) >= 5 else 1
        
        # 参数验证
        if not isinstance(c1, int) or c1 <= 0:
            raise ValueError(f"无效的输入通道数: {c1}")
        if not isinstance(c2, int) or c2 <= 0:
            raise ValueError(f"无效的输出通道数: {c2}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"无效的卷积核大小: {k}")
        if not isinstance(s, int) or s <= 0:
            raise ValueError(f"无效的步长: {s}")
        
        # 保存关键参数
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = k
        self.stride = s
        
        # 分组通道
        half_channels = c1 // 2
        
        # 分支1：d=1
        self.dw_conv1 = nn.Conv2d(
            half_channels, half_channels,
            kernel_size=k,
            stride=s,
            padding=k//2,
            dilation=1,
            groups=half_channels,
            bias=False
        )
        
        # 分支2：d=3 - 使用精确填充计算
        # 计算膨胀卷积的有效核大小
        effective_kernel = k + (k - 1) * (3 - 1)
        # 计算保持尺寸所需的填充
        padding_needed = (effective_kernel - 1) // 2
        
        self.dw_conv3 = nn.Conv2d(
            c1 - half_channels, c1 - half_channels,
            kernel_size=k,
            stride=s,
            padding=padding_needed,
            dilation=3,
            groups=c1 - half_channels,
            bias=False
        )
        
        # 点卷积
        self.pw_conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
    def forward(self, x):
        # 记录原始尺寸
        B, C, H, W = x.size()
        
        # 计算预期输出尺寸
        expected_H = (H + 2 * self.dw_conv1.padding[0] - 
                      self.dw_conv1.dilation[0] * (self.dw_conv1.kernel_size[0] - 1) - 1
                     ) // self.stride + 1
        expected_W = (W + 2 * self.dw_conv1.padding[1] - 
                      self.dw_conv1.dilation[1] * (self.dw_conv1.kernel_size[1] - 1) - 1
                     ) // self.stride + 1
        
        # 分割通道
        split_point = C // 2
        x1 = x[:, :split_point, :, :]
        x2 = x[:, split_point:, :, :]
        
        # 处理分支1
        x1 = self.dw_conv1(x1)
        
        # 处理分支2
        x2 = self.dw_conv3(x2)
        
        # 确保两个分支尺寸一致且符合预期
        if x1.shape[2] != expected_H or x1.shape[3] != expected_W:
            x1 = F.interpolate(x1, size=(expected_H, expected_W), mode='nearest')
        
        if x2.shape[2] != expected_H or x2.shape[3] != expected_W:
            x2 = F.interpolate(x2, size=(expected_H, expected_W), mode='nearest')
        
        # 连接分支
        x_cat = torch.cat([x1, x2], dim=1)
        out = self.pw_conv(x_cat)
        out = self.bn(out)
        out = self.act(out)
        
        # 验证输出尺寸
        if out.shape[2] != expected_H or out.shape[3] != expected_W:
            # 最终修正
            out = F.interpolate(out, size=(expected_H, expected_W), mode='nearest')
        
        return out