import torch
import torch.nn as nn

__all__ = ['C3k2_DFF_1']

def autopad(k, p=None, d=1):
    """自动填充以保持输出形状相同"""
    # 确保d是整数
    d = int(d)
    
    # 处理膨胀率
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = tuple(d * (x - 1) + 1 for x in k)
    
    # 计算填充
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:  # k 是元组
            p = tuple(x // 2 for x in k)
    
    # 确保p是整数或整数元组
    if isinstance(p, tuple):
        p = tuple(int(x) for x in p)
    else:
        p = int(p)
    
    return p

class DFF(nn.Module):
    """动态特征融合模块"""
    def __init__(self, dim):
        super().__init__()
        # 确保dim是整数
        dim = int(dim)
        
        # 通道注意力机制
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        
        # 空间注意力机制
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()
    
    def forward(self, x, skip):
        # 通道注意力
        output = torch.cat([x, skip], dim=1)
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)
        
        # 空间注意力
        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        return output * att

class Bottleneck_DFF(nn.Module):
    """使用DFF的瓶颈模块"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        # 确保所有参数是整数
        c1 = int(c1)
        c2 = int(c2)
        g = int(g)
        
        # 确保组数至少为1
        if g < 1:
            print(f"警告: 组数g={g}无效，已设置为1")
            g = 1
        
        # 确保 e 在合理范围
        e = max(0.1, min(e, 0.9))  # 0.1-0.9之间
        
        # 计算中间通道数，确保最小值为1
        self.c = max(1, int(c2 * e))
        
        # 计算 cv1 输出通道，确保最小值为1
        cv1_out = max(1, 2 * self.c)
        
        # 确保卷积核是整数类型
        k0 = k[0]
        k1 = k[1]
        
        # 处理卷积核类型
        if isinstance(k0, tuple):
            k0 = tuple(int(x) for x in k0)
        else:
            k0 = int(k0)
            
        if isinstance(k1, tuple):
            k1 = tuple(int(x) for x in k1)
        else:
            k1 = int(k1)
        
        # 打印调试信息
        print(f"Bottleneck_DFF参数: c1={c1}, c2={c2}, k0={k0}, k1={k1}, e={e}, self.c={self.c}, cv1_out={cv1_out}")
        
        self.cv1 = Conv(c1, cv1_out, k0, 1)
        self.cv2 = Conv(self.c, c2, k1, 1, g=g)
        self.add = shortcut and c1 == c2
        if self.add:
            self.DFF = DFF(c2)  # 使用DFF模块
    
    def forward(self, x):
        """前向传播"""
        # 通过第一个卷积层
        x1 = self.cv1(x)
        
        # 关键修复：将cv1的输出拆分为两部分
        # 第一部分用于后续处理，第二部分作为残差连接
        part1, part2 = torch.split(x1, [self.c, self.c], dim=1)
        
        # 通过第二个卷积层
        x2 = self.cv2(part1)
        
        if self.add:
            # 使用DFF模块融合原始输入和卷积结果
            return self.DFF(x, x2)
        else:
            return x2

class Conv(nn.Module):
    """标准卷积模块"""
    default_act = nn.SiLU()  # 默认激活函数
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        # 添加通道数有效性检查
        if c1 < 1 or c2 < 1:
            raise ValueError(f"无效的通道数: c1={c1}, c2={c2}. 通道数必须≥1")
        
        # 确保所有参数是整数
        c1 = int(c1)
        c2 = int(c2)
        s = int(s)
        g = int(g)
        d = int(d)
        
        # 确保组数至少为1
        if g < 1:
            print(f"警告: 组数g={g}无效，已设置为1")
            g = 1
        
        # 确保k是整数或整数元组
        if isinstance(k, tuple):
            k = tuple(int(x) for x in k)
        else:
            k = int(k)
        
        # 计算填充
        padding = autopad(k, p, d)
        
        # 确保填充是整数或整数元组
        if isinstance(padding, tuple):
            padding = tuple(int(x) for x in padding)
        else:
            padding = int(padding)
        
        # 打印调试信息
        print(f"Conv参数: c1={c1}, c2={c2}, k={k}, s={s}, padding={padding}, g={g}, d={d}")
        
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        """前向传播"""
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        """融合前向传播"""
        return self.act(self.conv(x))

class C3(nn.Module):
    """CSP瓶颈模块"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        # 确保所有参数是整数
        c1 = int(c1)
        c2 = int(c2)
        n = int(n)
        g = int(g)
        
        # 确保组数至少为1
        if g < 1:
            print(f"警告: 组数g={g}无效，已设置为1")
            g = 1
        
        # 确保 e 在合理范围
        e = max(0.1, min(e, 0.9))  # 0.1-0.9之间
        
        # 计算隐藏通道数，确保最小值为1
        c_ = max(1, int(c2 * e))
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # 可选激活函数
        # 使用DFF瓶颈模块
        self.m = nn.Sequential(*(Bottleneck_DFF(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
    
    def forward(self, x):
        """前向传播"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """自定义核大小的C3模块"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 确保所有参数是整数
        c1 = int(c1)
        c2 = int(c2)
        n = int(n)
        g = int(g)
        k = int(k)
        
        # 确保 e 在合理范围
        e = max(0.1, min(e, 0.9))  # 0.1-0.9之间
        
        # 计算隐藏通道数，确保最小值为1
        c_ = max(1, int(c2 * e))
        
        # 使用DFF瓶颈模块
        self.m = nn.Sequential(*(Bottleneck_DFF(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3kDFF(C3):
    """使用DFF的C3k模块"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 确保所有参数是整数
        c1 = int(c1)
        c2 = int(c2)
        n = int(n)
        g = int(g)
        k = int(k)
        
        # 确保 e 在合理范围
        e = max(0.1, min(e, 0.9))  # 0.1-0.9之间
        
        # 计算隐藏通道数，确保最小值为1
        c_ = max(1, int(c2 * e))
        
        # 使用DFF瓶颈模块
        self.m = nn.Sequential(*(Bottleneck_DFF(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C2f(nn.Module):
    """CSP瓶颈的快速实现"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        # 确保所有参数是整数
        c1 = int(c1)
        c2 = int(c2)
        n = int(n)
        g = int(g)
        
        # 确保 e 在合理范围
        e = max(0.1, min(e, 0.9))  # 0.1-0.9之间
        
        # 计算隐藏通道数，确保最小值为1
        self.c = max(1, int(c2 * e))
        
        # 计算 cv1 输出通道，确保最小值为1
        cv1_out = max(1, 2 * self.c)
        
        self.cv1 = Conv(c1, cv1_out, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 使用DFF瓶颈模块
        self.m = nn.ModuleList(Bottleneck_DFF(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    
    def forward(self, x):
        """前向传播"""
        # 将cv1的输出拆分为两部分
        y = list(self.cv1(x).chunk(2, 1))
        
        # 处理每个瓶颈模块
        for m in self.m:
            y.append(m(y[-1]))
        
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        """分割前向传播"""
        # 将cv1的输出分割为两部分
        y = list(self.cv1(x).split((self.c, self.c), 1))
        
        # 处理每个瓶颈模块
        for m in self.m:
            y.append(m(y[-1]))
        
        return self.cv2(torch.cat(y, 1))

class C3k2_DFF_1(C2f):
    """DFF改进版C3K2模块（类型1）"""
    def __init__(self, c1, c2, n=1, e=0.5, g=1, shortcut=True):
        """初始化模块"""
        # 确保所有参数是整数
        c1 = int(c1)
        c2 = int(c2)
        n = int(n)
        g = int(g)
        
        super().__init__(c1, c2, n, shortcut, g, e)
        # 统一使用DFF瓶颈模块
        self.m = nn.ModuleList(Bottleneck_DFF(self.c, self.c, shortcut, g) for _ in range(n))