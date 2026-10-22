import torch
import torch.nn as nn

class DWR(nn.Module):
    """完全兼容YOLO解析器的DWR模块"""
    def __init__(self, c1, c2=None, dilation_rates=None, reduction_ratio=8):
        """
        修复参数问题:
        c1: 输入通道数 (由YOLO自动传递)
        c2: 占位参数 (不会被使用，仅用于兼容性)
        dilation_rates: 可选的自定义膨胀率列表
        reduction_ratio: 通道压缩比例
        """
        super().__init__()
        
        # 1. 处理参数兼容性
        if dilation_rates is None:
            dilation_rates = [1, 2, 3]
        elif isinstance(dilation_rates, (int, float)):  # 处理单值参数
            dilation_rates = [int(dilation_rates)]
        
        # 2. 安全通道计算
        branch_channels = max(c1 // 2, 8)  # 防止小通道数
        
        # 3. 多尺度分支
        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            if rate <= 0:  # 跳过无效的膨胀率
                continue
                
            self.branches.append(nn.Sequential(
                nn.Conv2d(c1, branch_channels, 3, 
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.SiLU(inplace=True)
            ))
        
        # 确保有有效的分支
        if not self.branches:
            raise ValueError(f"无效的膨胀率设置: {dilation_rates}")
        
        # 4. 注意力机制
        fused_channels = branch_channels * len(self.branches)
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, max(fused_channels // reduction_ratio, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(fused_channels // reduction_ratio, 8), fused_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(fused_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 5. 输出变换
        self.conv_final = nn.Conv2d(fused_channels, c1, kernel_size=1)
        
    def forward(self, x):
        identity = x
        
        # 执行多尺度处理
        branch_outputs = [branch(x) for branch in self.branches]
        fused = torch.cat(branch_outputs, dim=1)
        
        # 应用通道注意力
        channel_att = self.channel_att(fused)
        channel_weighted = fused * channel_att
        
        # 应用空间注意力
        spatial_att = self.spatial_att(channel_weighted)
        att_weighted = channel_weighted * spatial_att
        
        # 残差连接
        out = self.conv_final(att_weighted)
        return identity + out