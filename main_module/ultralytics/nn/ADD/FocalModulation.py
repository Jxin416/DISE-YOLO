import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        # 确保输出通道数正确
        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.kernel_sizes.append(kernel_size)
            
            # 使用动态填充确保尺寸不变
            padding = kernel_size // 2
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=padding, bias=False),
                    nn.GELU(),
                )
            )
        
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        B, C, H, W = x.shape
        orig_size = (H, W)  # 保存原始尺寸
        
        # 确保输入尺寸兼容
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            H = H + H % 2
            W = W + W % 2

        # pre linear projection
        x_proj = self.f_linear(x)
        q, ctx, gates = torch.split(x_proj, [C, C, self.focal_level + 1], dim=1)

        # 确保所有张量尺寸一致
        if gates.shape[2:] != (H, W):
            gates = F.interpolate(gates, size=(H, W), mode='bilinear', align_corners=False)

        # context aggregation
        ctx_all = 0.0
        for l in range(self.focal_level):
            ctx_layer = self.focal_layers[l](ctx)
            
            # 确保ctx_layer尺寸正确
            if ctx_layer.shape[2:] != (H, W):
                ctx_layer = F.interpolate(ctx_layer, size=(H, W), mode='bilinear', align_corners=False)
            
            ctx_all = ctx_all + ctx_layer * gates[:, l:l + 1]
        
        # 全局上下文 - 确保尺寸正确
        ctx_global = self.act(ctx.mean((2, 3), keepdim=True))
        ctx_global = F.interpolate(ctx_global, size=(H, W), mode='nearest')
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        
        if self.use_postln_in_modulation:
            # 转换为 (B, H, W, C) 格式进行 LayerNorm
            x_out = x_out.permute(0, 2, 3, 1)
            x_out = self.ln(x_out)
            x_out = x_out.permute(0, 3, 1, 2)

        # post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        # 恢复原始尺寸
        return x_out[..., :orig_size[0], :orig_size[1]]