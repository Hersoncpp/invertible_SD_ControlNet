"""
Invertible 1x1 Convolution Module
用于可逆网络的通道混洗操作
基于 GLOW 和 RealNVP 中的可逆 1x1 卷积
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Invertible1x1Conv(nn.Module):
    """
    Invertible 1x1 Convolution using LU decomposition
    使用 LU 分解实现高效的可逆 1x1 卷积
    参考: GLOW (Kingma & Dhariwal, 2018)
    """
    def __init__(self, num_channels, use_lu=True):
        super(Invertible1x1Conv, self).__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        
        if use_lu:
            # LU 分解方式：W = PL(U + diag(s))
            # 其中 P 是排列矩阵，L 是下三角矩阵，U 是上三角矩阵，s 是对角元素
            w_shape = (num_channels, num_channels)
            w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
            
            # 分解为 P, L, U, s
            # 使用 PyTorch 的 LU 分解（如果可用）或简化版本
            try:
                import scipy.linalg
                np_p, np_l, np_u = scipy.linalg.lu(w_init)
                np_s = np.diag(np_u)
                np_u = np.triu(np_u, 1)
            except ImportError:
                # 如果没有 scipy，使用简化版本
                # 直接使用 QR 分解的结果，手动构造 L 和 U
                np_p = np.eye(num_channels)
                # 使用下三角和上三角分解
                np_l = np.tril(w_init, -1) + np.eye(num_channels)
                np_s = np.diag(w_init)
                np_u = np.triu(w_init, 1)
            
            # 注册为 buffer 和 parameter
            self.register_buffer('p', torch.from_numpy(np_p))
            self.register_buffer('p_inv', torch.from_numpy(np_p).inverse())
            
            # L 和 U 的对角线设为 0（对角线信息在 s 中）
            l_mask = np.tril(np.ones_like(np_l), -1)
            u_mask = np.triu(np.ones_like(np_u), 1)
            
            self.register_parameter('l', nn.Parameter(torch.from_numpy(np_l * l_mask)))
            self.register_parameter('u', nn.Parameter(torch.from_numpy(np_u * u_mask)))
            self.register_parameter('s', nn.Parameter(torch.from_numpy(np_s)))
        else:
            # 直接使用可学习的权重矩阵
            weight = torch.randn(num_channels, num_channels)
            # 使用 QR 分解初始化，确保可逆
            q, r = torch.qr(weight)
            w = q
            self.register_parameter('weight', nn.Parameter(w))
    
    def get_weight(self, reverse=False):
        """
        获取权重矩阵（正向或反向）
        """
        if self.use_lu:
            # 重构权重矩阵: W = PL(U + diag(s))
            l = torch.tril(self.l, diagonal=-1) + torch.eye(self.num_channels, device=self.l.device)
            u = torch.triu(self.u, diagonal=1) + torch.diag(self.s)
            w = torch.matmul(self.p, torch.matmul(l, u))
        else:
            w = self.weight
        
        if reverse:
            # 返回逆矩阵
            w = torch.inverse(w)
        
        return w.view(self.num_channels, self.num_channels, 1, 1)
    
    def forward(self, x, rev=False):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            rev: If True, perform inverse convolution
        Returns:
            Output tensor of same shape (B, C, H, W)
        """
        weight = self.get_weight(reverse=rev)
        out = F.conv2d(x, weight)
        
        # 计算 log determinant for jacobian
        if self.use_lu:
            # log|det(W)| = sum(log|s|)
            self.logdet = torch.sum(torch.log(torch.abs(self.s))) * x.shape[2] * x.shape[3]
        else:
            # log|det(W)| = log|det(weight)| * H * W
            self.logdet = torch.logdet(self.weight.view(self.num_channels, self.num_channels)) * x.shape[2] * x.shape[3]
        
        if rev:
            self.logdet = -self.logdet
        
        return out
    
    def jacobian(self, x, rev=False):
        """
        计算雅可比行列式的对数
        """
        if not hasattr(self, 'logdet'):
            _ = self.forward(x, rev=rev)
        return self.logdet / x.shape[0]


class Invertible1x1ConvSimple(nn.Module):
    """
    Simple Invertible 1x1 Convolution
    使用直接矩阵求逆的简单实现（适合小通道数）
    """
    def __init__(self, num_channels):
        super(Invertible1x1ConvSimple, self).__init__()
        self.num_channels = num_channels
        
        # 初始化可逆矩阵
        weight = torch.randn(num_channels, num_channels)
        # 使用 QR 分解确保可逆性
        q, r = torch.qr(weight)
        self.register_parameter('weight', nn.Parameter(q))
    
    def forward(self, x, rev=False):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            rev: If True, perform inverse convolution
        """
        B, C, H, W = x.shape
        
        if not rev:
            # 正向: y = W @ x
            weight = self.weight.view(C, C, 1, 1)
            out = F.conv2d(x, weight)
            self.logdet = torch.logdet(self.weight) * H * W
        else:
            # 反向: x = W^{-1} @ y
            weight_inv = torch.inverse(self.weight).view(C, C, 1, 1)
            out = F.conv2d(x, weight_inv)
            self.logdet = -torch.logdet(self.weight) * H * W
        
        return out
    
    def jacobian(self, x, rev=False):
        """
        计算雅可比行列式的对数
        """
        if not hasattr(self, 'logdet'):
            _ = self.forward(x, rev=rev)
        return self.logdet / x.shape[0]

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, input, rev):
        w_shape = self.w_shape
        if not rev:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, rev=False):
        weight = self.get_weight(input, rev)
        if not rev:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z

class ConditionalInvertible1x1Conv(nn.Module):
    """
    Conditional Invertible 1x1 Convolution with text embedding
    支持文本嵌入的条件可逆 1x1 卷积
    """
    def __init__(self, num_channels, embedding_dim=768, use_lu=True):
        super(ConditionalInvertible1x1Conv, self).__init__()
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        
        # 基础可逆 1x1 卷积
        self.base_conv = Invertible1x1Conv(num_channels, use_lu=use_lu)
        
        # 条件调制
        self.condition_scale = nn.Linear(embedding_dim, num_channels)
        self.condition_shift = nn.Linear(embedding_dim, num_channels)
    
    def forward(self, x, rev=False, text_embedding=None):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            rev: If True, perform inverse convolution
            text_embedding: Text embedding of shape (B, embedding_dim)
        """
        if text_embedding is not None:
            # 应用条件调制
            scale = self.condition_scale(text_embedding).view(-1, self.num_channels, 1, 1)
            shift = self.condition_shift(text_embedding).view(-1, self.num_channels, 1, 1)
            x = x * (1 + scale) + shift
        
        # 应用可逆 1x1 卷积
        return self.base_conv(x, rev=rev)
    
    def jacobian(self, x, rev=False):
        return self.base_conv.jacobian(x, rev=rev)


# 为了兼容性，如果没有 scipy，提供一个不依赖 scipy 的版本
try:
    import scipy.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using simplified LU decomposition")


if not HAS_SCIPY:
    # 简化版本的 LU 分解（不依赖 scipy）
    class Invertible1x1Conv(nn.Module):
        """
        Invertible 1x1 Convolution (simplified version without scipy)
        """
        def __init__(self, num_channels, use_lu=True):
            super(Invertible1x1Conv, self).__init__()
            self.num_channels = num_channels
            self.use_lu = use_lu
            
            if use_lu:
                # 使用 PyTorch 的 LU 分解
                weight = torch.randn(num_channels, num_channels)
                # 使用 QR 分解初始化
                q, r = torch.qr(weight)
                w = q
                
                # 手动 LU 分解（简化版）
                # 这里我们使用一个可学习的下三角和上三角矩阵
                l_mask = torch.tril(torch.ones(num_channels, num_channels), diagonal=-1)
                u_mask = torch.triu(torch.ones(num_channels, num_channels), diagonal=1)
                
                self.register_parameter('l', nn.Parameter(torch.randn(num_channels, num_channels) * 0.1))
                self.register_parameter('u', nn.Parameter(torch.randn(num_channels, num_channels) * 0.1))
                self.register_parameter('s', nn.Parameter(torch.ones(num_channels)))
                
                # 初始化
                with torch.no_grad():
                    self.l.data = self.l.data * l_mask
                    self.u.data = self.u.data * u_mask
                    self.s.data = torch.ones(num_channels)
            else:
                weight = torch.randn(num_channels, num_channels)
                q, r = torch.qr(weight)
                self.register_parameter('weight', nn.Parameter(q))
        
        def get_weight(self, reverse=False):
            if self.use_lu:
                l = torch.tril(self.l, diagonal=-1) + torch.diag(torch.ones(self.num_channels, device=self.l.device))
                u = torch.triu(self.u, diagonal=1) + torch.diag(self.s)
                w = torch.matmul(l, u)
            else:
                w = self.weight
            
            if reverse:
                w = torch.inverse(w)
            
            return w.view(self.num_channels, self.num_channels, 1, 1)
        
        def forward(self, x, rev=False):
            weight = self.get_weight(reverse=rev)
            out = F.conv2d(x, weight)
            
            if self.use_lu:
                self.logdet = torch.sum(torch.log(torch.abs(self.s))) * x.shape[2] * x.shape[3]
            else:
                self.logdet = torch.logdet(self.weight.view(self.num_channels, self.num_channels)) * x.shape[2] * x.shape[3]
            
            if rev:
                self.logdet = -self.logdet
            
            return out
        
        def jacobian(self, x, rev=False):
            if not hasattr(self, 'logdet'):
                _ = self.forward(x, rev=rev)
            return self.logdet / x.shape[0]

