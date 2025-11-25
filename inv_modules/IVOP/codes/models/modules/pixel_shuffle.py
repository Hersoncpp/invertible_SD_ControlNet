"""
Pixel Shuffling Module for Invertible Networks
支持可逆的上采样和下采样操作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffle(nn.Module):
    """
    Pixel Shuffle for upsampling
    将 (B, C*r^2, H, W) -> (B, C, H*r, W*r)
    """
    def __init__(self, upscale_factor=2):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x, rev=False):
        if not rev:
            # Upsampling: (B, C*r^2, H, W) -> (B, C, H*r, W*r)
            return F.pixel_shuffle(x, self.upscale_factor)
        else:
            # Downsampling: (B, C, H*r, W*r) -> (B, C*r^2, H, W)
            return F.pixel_unshuffle(x, self.upscale_factor)


class InvertiblePixelShuffle(nn.Module):
    """
    Invertible Pixel Shuffle with learnable channel mixing
    结合了 Pixel Shuffle 和可逆通道混洗
    """
    def __init__(self, in_channels, upscale_factor=2):
        super(InvertiblePixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.out_channels = in_channels // (upscale_factor ** 2)
        
        # 可学习的通道混洗权重
        self.channel_mix = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=1, 
            bias=False
        )
        # 初始化为单位矩阵
        nn.init.eye_(self.channel_mix.weight.view(in_channels, in_channels))
        
    def forward(self, x, rev=False):
        if not rev:
            # 先进行通道混洗
            x = self.channel_mix(x)
            # 然后进行 Pixel Shuffle 上采样
            return F.pixel_shuffle(x, self.upscale_factor)
        else:
            # 先进行 Pixel Unshuffle 下采样
            x = F.pixel_unshuffle(x, self.upscale_factor)
            # 然后进行逆通道混洗
            # 使用转置权重
            weight = self.channel_mix.weight.squeeze()
            weight_inv = torch.inverse(weight)
            weight_inv = weight_inv.view(self.in_channels, self.in_channels, 1, 1)
            x = F.conv2d(x, weight_inv, bias=None)
            return x


class ConditionalPixelShuffle(nn.Module):
    """
    Conditional Pixel Shuffle with text embedding support
    支持文本嵌入的条件上采样/下采样
    """
    def __init__(self, in_channels, upscale_factor=2, embedding_dim=768):
        super(ConditionalPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        
        # 条件调制层
        self.condition_scale = nn.Linear(embedding_dim, in_channels)
        self.condition_shift = nn.Linear(embedding_dim, in_channels)
        
    def forward(self, x, rev=False, text_embedding=None):
        if text_embedding is not None:
            # 获取条件参数
            scale = self.condition_scale(text_embedding).view(-1, self.in_channels, 1, 1)
            shift = self.condition_shift(text_embedding).view(-1, self.in_channels, 1, 1)
            # 应用条件调制
            x = x * (1 + scale) + shift
        
        if not rev:
            return F.pixel_shuffle(x, self.upscale_factor)
        else:
            return F.pixel_unshuffle(x, self.upscale_factor)

