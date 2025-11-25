"""
HAAR Wavelet Transform Module for Invertible Networks
使用 HAAR 小波变换实现可逆的下采样和上采样
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HaarTransform(nn.Module):
    """
    HAAR Wavelet Transform for invertible downsampling/upsampling
    使用 HAAR 小波变换实现可逆的多尺度分解
    """
    def __init__(self):
        super(HaarTransform, self).__init__()
        
        # HAAR 小波滤波器
        # Low-pass filter (scaling function)
        ll = np.array([[1, 1], [1, 1]]) / 2.0
        # High-pass filters (wavelet functions)
        lh = np.array([[-1, -1], [1, 1]]) / 2.0
        hl = np.array([[-1, 1], [-1, 1]]) / 2.0
        hh = np.array([[1, -1], [-1, 1]]) / 2.0
        
        # 转换为 PyTorch tensor
        self.register_buffer('ll', torch.from_numpy(ll).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh', torch.from_numpy(lh).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl', torch.from_numpy(hl).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh', torch.from_numpy(hh).float().unsqueeze(0).unsqueeze(0))
    
    def forward(self, x, rev=False):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            rev: If True, perform inverse transform (upsampling)
        Returns:
            If rev=False: (B, 4*C, H/2, W/2) - 4 subbands (LL, LH, HL, HH)
            If rev=True: (B, C, 2*H, 2*W) - reconstructed image
        """
        B, C, H, W = x.shape
        
        if not rev:
            # Forward transform: 2D HAAR decomposition
            # 确保 H 和 W 是偶数
            if H % 2 != 0:
                x = F.pad(x, (0, 0, 0, 1), mode='reflect')
                H += 1
            if W % 2 != 0:
                x = F.pad(x, (0, 1, 0, 0), mode='reflect')
                W += 1
            
            # 对每个通道分别进行 HAAR 变换
            ll_list = []
            lh_list = []
            hl_list = []
            hh_list = []
            
            for c in range(C):
                channel = x[:, c:c+1, :, :]
                
                # 应用 2D 卷积进行小波分解
                ll = F.conv2d(channel, self.ll, stride=2, padding=0)
                lh = F.conv2d(channel, self.lh, stride=2, padding=0)
                hl = F.conv2d(channel, self.hl, stride=2, padding=0)
                hh = F.conv2d(channel, self.hh, stride=2, padding=0)
                
                ll_list.append(ll)
                lh_list.append(lh)
                hl_list.append(hl)
                hh_list.append(hh)
            
            # 拼接所有子带
            ll = torch.cat(ll_list, dim=1)  # (B, C, H/2, W/2)
            lh = torch.cat(lh_list, dim=1)  # (B, C, H/2, W/2)
            hl = torch.cat(hl_list, dim=1)  # (B, C, H/2, W/2)
            hh = torch.cat(hh_list, dim=1)  # (B, C, H/2, W/2)
            
            # 按通道维度拼接: (B, 4*C, H/2, W/2)
            return torch.cat([ll, lh, hl, hh], dim=1)
        
        else:
            # Inverse transform: 2D HAAR reconstruction
            # 分离 4 个子带
            C_out = x.shape[1] // 4
            ll = x[:, 0:C_out, :, :]
            lh = x[:, C_out:2*C_out, :, :]
            hl = x[:, 2*C_out:3*C_out, :, :]
            hh = x[:, 3*C_out:4*C_out, :, :]
            
            # 对每个通道分别进行逆变换
            reconstructed_list = []
            
            for c in range(C_out):
                ll_c = ll[:, c:c+1, :, :]
                lh_c = lh[:, c:c+1, :, :]
                hl_c = hl[:, c:c+1, :, :]
                hh_c = hh[:, c:c+1, :, :]
                
                # 上采样并应用逆滤波器
                ll_up = F.conv_transpose2d(ll_c, self.ll, stride=2, padding=0, output_padding=0)
                lh_up = F.conv_transpose2d(lh_c, self.lh, stride=2, padding=0, output_padding=0)
                hl_up = F.conv_transpose2d(hl_c, self.hl, stride=2, padding=0, output_padding=0)
                hh_up = F.conv_transpose2d(hh_c, self.hh, stride=2, padding=0, output_padding=0)
                
                # 重构
                reconstructed = ll_up + lh_up + hl_up + hh_up
                reconstructed_list.append(reconstructed)
            
            # 拼接所有通道
            return torch.cat(reconstructed_list, dim=1)


class LearnableHaarTransform(nn.Module):
    """
    Learnable HAAR-like Transform
    可学习的类 HAAR 小波变换
    """
    def __init__(self, in_channels):
        super(LearnableHaarTransform, self).__init__()
        self.in_channels = in_channels
        
        # 可学习的滤波器
        self.ll_filter = nn.Parameter(torch.randn(1, 1, 2, 2) * 0.1)
        self.lh_filter = nn.Parameter(torch.randn(1, 1, 2, 2) * 0.1)
        self.hl_filter = nn.Parameter(torch.randn(1, 1, 2, 2) * 0.1)
        self.hh_filter = nn.Parameter(torch.randn(1, 1, 2, 2) * 0.1)
        
        # 初始化接近标准 HAAR
        with torch.no_grad():
            self.ll_filter.data = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
            self.lh_filter.data = torch.tensor([[[[-0.5, -0.5], [0.5, 0.5]]]])
            self.hl_filter.data = torch.tensor([[[[-0.5, 0.5], [-0.5, 0.5]]]])
            self.hh_filter.data = torch.tensor([[[[0.5, -0.5], [-0.5, 0.5]]]])
    
    def forward(self, x, rev=False):
        B, C, H, W = x.shape
        
        if not rev:
            # 确保 H 和 W 是偶数
            if H % 2 != 0:
                x = F.pad(x, (0, 0, 0, 1), mode='reflect')
                H += 1
            if W % 2 != 0:
                x = F.pad(x, (0, 1, 0, 0), mode='reflect')
                W += 1
            
            # 对每个通道应用可学习滤波器
            ll_list, lh_list, hl_list, hh_list = [], [], [], []
            
            for c in range(C):
                channel = x[:, c:c+1, :, :]
                ll = F.conv2d(channel, self.ll_filter, stride=2, padding=0)
                lh = F.conv2d(channel, self.lh_filter, stride=2, padding=0)
                hl = F.conv2d(channel, self.hl_filter, stride=2, padding=0)
                hh = F.conv2d(channel, self.hh_filter, stride=2, padding=0)
                
                ll_list.append(ll)
                lh_list.append(lh)
                hl_list.append(hl)
                hh_list.append(hh)
            
            ll = torch.cat(ll_list, dim=1)
            lh = torch.cat(lh_list, dim=1)
            hl = torch.cat(hl_list, dim=1)
            hh = torch.cat(hh_list, dim=1)
            
            return torch.cat([ll, lh, hl, hh], dim=1)
        
        else:
            C_out = x.shape[1] // 4
            ll = x[:, 0:C_out, :, :]
            lh = x[:, C_out:2*C_out, :, :]
            hl = x[:, 2*C_out:3*C_out, :, :]
            hh = x[:, 3*C_out:4*C_out, :, :]
            
            reconstructed_list = []
            for c in range(C_out):
                ll_c = ll[:, c:c+1, :, :]
                lh_c = lh[:, c:c+1, :, :]
                hl_c = hl[:, c:c+1, :, :]
                hh_c = hh[:, c:c+1, :, :]
                
                ll_up = F.conv_transpose2d(ll_c, self.ll_filter, stride=2, padding=0)
                lh_up = F.conv_transpose2d(lh_c, self.lh_filter, stride=2, padding=0)
                hl_up = F.conv_transpose2d(hl_c, self.hl_filter, stride=2, padding=0)
                hh_up = F.conv_transpose2d(hh_c, self.hh_filter, stride=2, padding=0)
                
                reconstructed = ll_up + lh_up + hl_up + hh_up
                reconstructed_list.append(reconstructed)
            
            return torch.cat(reconstructed_list, dim=1)


class ConditionalHaarTransform(nn.Module):
    """
    Conditional HAAR Transform with text embedding support
    支持文本嵌入的条件 HAAR 变换
    """
    def __init__(self, in_channels, embedding_dim=768):
        super(ConditionalHaarTransform, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        
        # 条件调制
        self.condition_scale = nn.Linear(embedding_dim, in_channels)
        self.condition_shift = nn.Linear(embedding_dim, in_channels)
        
        # HAAR 变换
        self.haar = HaarTransform()
    
    def forward(self, x, rev=False, text_embedding=None):
        if text_embedding is not None:
            # 应用条件调制
            scale = self.condition_scale(text_embedding).view(-1, self.in_channels, 1, 1)
            shift = self.condition_shift(text_embedding).view(-1, self.in_channels, 1, 1)
            x = x * (1 + scale) + shift
        
        return self.haar(x, rev=rev)

