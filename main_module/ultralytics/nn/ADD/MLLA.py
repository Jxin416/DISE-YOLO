import torch
import torch.nn as nn
 
__all__ = ['MLLAttention', 'C2PSAMLLA']
 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
 
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
 
 
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
 
    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base
        # 检测PyTorch版本并设置兼容性标志
        self._setup_compatibility()
 
    def _setup_compatibility(self):
        """设置版本兼容性标志"""
        torch_version = torch.__version__
        version_parts = torch_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        # PyTorch 1.10+ 支持indexing参数
        self.supports_indexing = (major > 1) or (major == 1 and minor >= 10)
 
    def generate_rotations(self, x):
        # 获取输入张量的形状
        *channel_dims, feature_dim = x.shape[1:-1][0], x.shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
 
        assert feature_dim % k_max == 0, "Feature dimension must be divisible by 2 * k_max"
 
        # 生成角度
        theta_ks = 1 / (self.base ** (torch.arange(k_max, dtype=x.dtype, device=x.device) / k_max))
        
        # 兼容不同PyTorch版本的meshgrid调用
        if self.supports_indexing:
            # PyTorch 1.10+ 版本
            mesh_tensors = torch.meshgrid(
                [torch.arange(d, dtype=x.dtype, device=x.device) for d in channel_dims],
                indexing='ij'
            )
        else:
            # PyTorch 1.10以下版本
            mesh_tensors = torch.meshgrid(
                [torch.arange(d, dtype=x.dtype, device=x.device) for d in channel_dims]
            )
        
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in mesh_tensors], dim=-1)
 
        # 计算旋转矩阵的实部和虚部
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
 
        return rotations
 
    def forward(self, x):
        # 生成旋转矩阵
        rotations = self.generate_rotations(x)
 
        # 将 x 转换为复数形式
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)
 
        # 应用旋转矩阵
        rotations_complex = torch.view_as_complex(rotations)
        pe_x = rotations_complex * x_complex
 
        # 将结果转换回实数形式并展平最后两个维度
        result_real = torch.view_as_real(pe_x)
        return result_real.flatten(-2)
 
 
class MLLAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
 
    def __init__(self, dim=3, input_resolution=[160, 160], num_heads=4, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()
 
    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        # 重新整形输入
        b, c, h, w = x.shape
        n = h * w
        x_reshaped = x.permute(0, 2, 3, 1).reshape(b, n, c)
        
        num_heads = self.num_heads
        head_dim = c // num_heads
 
        # 计算查询、键、值
        qk = self.qk(x_reshaped).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x_reshaped
 
        # 应用激活函数
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        # 应用旋转位置编码
        q_reshaped = q.reshape(b, h, w, c)
        k_reshaped = k.reshape(b, h, w, c)
        
        q_rope = self.rope(q_reshaped).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k_reshaped).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # 重塑标准查询和键
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
 
        # 计算注意力权重和输出
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x_attn = q_rope @ kv * z
 
        # 重塑输出
        x_attn = x_attn.transpose(1, 2).reshape(b, n, c)
        
        # 应用局部位置增强
        v_spatial = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        lepe_output = self.lepe(v_spatial).permute(0, 2, 3, 1).reshape(b, n, c)
        
        # 合并注意力输出和局部增强
        x_output = x_attn + lepe_output
        x_output = x_output.transpose(2, 1).reshape(b, c, h, w)
        
        return x_output
 
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
 
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
 
class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.
    """
 
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        self.attn = MLLAttention(c)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False)
        )
        self.add = shortcut
 
    def forward(self, x):
        """Executes a forward pass through PSABlock."""
        if self.add:
            x = x + self.attn(x)
            x = x + self.ffn(x)
        else:
            x = self.attn(x)
            x = self.ffn(x)
        return x
 
 
class C2PSAMLLA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction.
    """
 
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, "Input and output channels must be equal"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
 
    def forward(self, x):
        """Processes the input tensor through PSA blocks."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


def test_rope_compatibility():
    """测试RoPE模块的兼容性"""
    print("🧪 测试RoPE模块兼容性...")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 创建测试输入
    x = torch.randn(1, 3, 160, 160)  # B, C, H, W
    rope = RoPE()
    
    try:
        output = rope(x)
        print("✅ RoPE模块修复成功！")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"支持indexing参数: {rope.supports_indexing}")
        return True
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mll_attention():
    """测试MLLAttention模块"""
    print("\n🧪 测试MLLAttention模块...")
    try:
        x = torch.randn(2, 64, 32, 32)  # B, C, H, W
        attn = MLLAttention(dim=64, input_resolution=[32, 32], num_heads=8)
        output = attn(x)
        print("✅ MLLAttention模块测试成功！")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ MLLAttention测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("MLLA模块兼容性测试")
    print("=" * 50)
    
    # 测试RoPE兼容性
    rope_success = test_rope_compatibility()
    
    # 测试MLLAttention
    attn_success = test_mll_attention()
    
    print("\n" + "=" * 50)
    if rope_success and attn_success:
        print("🎉 所有测试通过！模块可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    print("=" * 50)