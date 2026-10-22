import math
import torch
from torch import nn
# 你的实现里用到了 einops，这里继续保留；也可用 .view() 替代
from einops.layers.torch import Rearrange
 
 
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
 
    def get_weight(self):
        # —— 设备/数据类型安全：用 new_zeros，避免硬编码 CUDA —— #
        conv_weight = self.conv.weight  # [out_c, in_c, k, k]
        oc, ic, k, _ = conv_weight.shape
        conv_flat = Rearrange('oc ic k1 k2 -> oc ic (k1 k2)')(conv_weight)  # [oc, ic, 9]
        conv_weight_cd = conv_weight.new_zeros(oc, ic, 3 * 3)
 
        conv_weight_cd[:, :, :] = conv_flat
        # 中心差分：中心 = 中心 - 所有元素和  ==>  等价于 -邻域和（总和减去中心）
        conv_weight_cd[:, :, 4] = conv_flat[:, :, 4] - conv_flat.sum(dim=2)
 
        conv_weight_cd = Rearrange('oc ic (k1 k2) -> oc ic k1 k2', k1=k, k2=k)(conv_weight_cd)
        return conv_weight_cd, self.conv.bias
 
 
class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
 
    def get_weight(self):
        conv_weight = self.conv.weight  # [oc, ic, k, k]
        oc, ic, k, _ = conv_weight.shape
        conv_flat = Rearrange('oc ic k1 k2 -> oc ic (k1 k2)')(conv_weight)  # [oc, ic, 9]
        # 角向差分：按你给定的索引重排做相减
        idx = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        conv_weight_ad = conv_flat - self.theta * conv_flat[:, :, idx]
        conv_weight_ad = Rearrange('oc ic (k1 k2) -> oc ic k1 k2', k1=k, k2=k)(conv_weight_ad)
        return conv_weight_ad, self.conv.bias
 
 
class Conv2d_rd(nn.Module):
    # 你这里是 5x5 的“扩展差分”分支，DEConv 里当前没用到，原样保留
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
 
    def forward(self, x):
        if math.fabs(self.theta - 0.0) < 1e-8:
            return self.conv(x)
        else:
            conv_weight = self.conv.weight
            oc, ic, k, _ = conv_weight.shape  # k=3
            # —— 设备/类型安全 —— #
            conv_weight_rd = conv_weight.new_zeros(oc, ic, 5 * 5)
 
            conv_flat = Rearrange('oc ic k1 k2 -> oc ic (k1 k2)')(conv_weight)  # [oc, ic, 9]
            # 把 3x3 的 8 个邻域权重布到 5x5 的外环（你原来的索引）
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_flat[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_flat[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_flat[:, :, 0] * (1 - self.theta)
 
            conv_weight_rd = conv_weight_rd.view(oc, ic, 5, 5)
            out = nn.functional.conv2d(x, conv_weight_rd, bias=self.conv.bias,
                                       stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
            return out
 
 
class Conv2d_hd(nn.Module):
    # 保留你的 Conv1d 生成权重的思路，但用 new_zeros 保证设备安全
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
 
    def get_weight(self):
        w1d = self.conv.weight  # [oc, ic, k], k=3
        oc, ic, k = w1d.shape
        w2d = w1d.new_zeros(oc, ic, 3 * 3)
        # 水平差分：左正右负（列 0/1/2 → 映射到 3x3 的列 0 和 2）
        w2d[:, :, [0, 3, 6]] = w1d
        w2d[:, :, [2, 5, 8]] = -w1d
        w2d = Rearrange('oc ic (k1 k2) -> oc ic k1 k2', k1=k, k2=k)(w2d)  # k=3
        return w2d, self.conv.bias
 
 
class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
 
    def get_weight(self):
        w1d = self.conv.weight  # [oc, ic, k], k=3
        oc, ic, k = w1d.shape
        w2d = w1d.new_zeros(oc, ic, 3 * 3)
        # 垂直差分：上正下负（行 0 和 2）
        w2d[:, :, [0, 1, 2]] = w1d
        w2d[:, :, [6, 7, 8]] = -w1d
        w2d = Rearrange('oc ic (k1 k2) -> oc ic k1 k2', k1=k, k2=k)(w2d)
        return w2d, self.conv.bias
 
 
class DEConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 注意：这里假设所有分支都是 k=3, stride=1, padding=1，才能线性可加
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
 
        self._deployed = False
 
    def _sum_equivalent_weight_bias(self):
        # 从每条分支拿到“等效 3x3 权重和偏置”，逐元素相加
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias
 
        w = w1 + w2 + w3 + w4 + w5
        # 偏置可能有 None，这里统一成张量相加
        zeros = torch.zeros_like(b5) if b5 is not None else torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
        b1 = b1 if b1 is not None else zeros
        b2 = b2 if b2 is not None else zeros
        b3 = b3 if b3 is not None else zeros
        b4 = b4 if b4 is not None else zeros
        b5 = b5 if b5 is not None else zeros
 
        b = b1 + b2 + b3 + b4 + b5
        return w.contiguous(), b.contiguous()
 
    def forward(self, x):
        if self._deployed:
            # 部署：单卷积直走
            return self.conv1_5(x)
        else:
            # 训练：为了与你的原始写法保持一致，这里仍然“在线求和一次”做单次 conv2d
            # 如需严格多分支并行（每支后面再接 非线性/BN），可改为 y=sum(branch(x)) 的实现
            w, b = self._sum_equivalent_weight_bias()
            return nn.functional.conv2d(x, w, b, stride=1, padding=1, groups=1)
 
    @torch.no_grad()
    def switch_to_deploy(self):
        """把 4 条差分分支 + 1 条普通分支折叠到 conv1_5，并删除分支。"""
        if self._deployed:
            return
 
        w, b = self._sum_equivalent_weight_bias()
        # 直接把折叠后的权重写回 conv1_5
        self.conv1_5.weight.copy_(w)
        if self.conv1_5.bias is None:
            self.conv1_5.bias = nn.Parameter(b.clone())
        else:
            self.conv1_5.bias.copy_(b)
 
        # 删除训练期分支以节省显存 & 防止误用
        del self.conv1_1
        del self.conv1_2
        del self.conv1_3
        del self.conv1_4
 
        self._deployed = True
 
 
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
 
    x = torch.randn(1, 32, 64, 64, device=device)
 
    # 1) 训练态输出（在线求和）
    m = DEConv(32).to(device)
    y_train = m(x)
 
    # 2) 折叠到部署态输出（单卷积）
    m.switch_to_deploy()
    m.eval()
    y_deploy = m(x)
 
    print('y_train:', y_train.shape, 'y_deploy:', y_deploy.shape)
    diff = (y_train - y_deploy).abs().max().item()
    rel = (y_train - y_deploy).abs().mean().item() / (y_train.abs().mean().item() + 1e-8)
    print(f'max |Δ| = {diff:.6f},  mean rel error = {rel:.6e}')