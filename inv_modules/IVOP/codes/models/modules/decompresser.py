import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.Subnet_constructor import subnet

# class AttentionGate(nn.Module):
#     def __init__(self, F_g, F_l, F_int, init='xavier', bias=True):
#         super(AttentionGate, self).__init__()
#         self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=bias)
#         self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=bias)
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=bias),
#             nn.Sigmoid()
#         )
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         if init == 'xavier':
#             mutil.initialize_weights_xavier([self.W_g, self.W_x, self.psi[0]], 0.1)
#         else:
#             mutil.initialize_weights([self.W_g, self.W_x, self.psi[0]], 0.1)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x1 = self.W_x(x)
#         psi = self.lrelu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class UNet(nn.Module):
    # def __init__(self, n_channels=3, n_classes=3, block_type='ConvNet', init='xavier'):
    #     super(UNet, self).__init__()
    #     self.block = subnet(block_type, init)
        
    #     # Encoder
    #     self.enc1 = self.block(n_channels, 64)
    #     self.pool1 = nn.MaxPool2d(2)
    #     self.enc2 = self.block(64, 128)
    #     self.pool2 = nn.MaxPool2d(2)
    #     self.enc3 = self.block(128, 256)
    #     self.pool3 = nn.MaxPool2d(2)
    #     self.enc4 = self.block(256, 512)
    #     self.pool4 = nn.MaxPool2d(2)
    #     self.bottleneck = self.block(512, 1024)
        
    #     # Decoder
    #     self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
    #     self.dec4 = self.block(1024, 512)  # 512 (up) + 512 (att_skip)
    #     self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
    #     self.dec3 = self.block(512, 256)
    #     self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
    #     self.dec2 = self.block(256, 128)
    #     self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    #     self.dec1 = self.block(128, 64)
        
    #     self.outc = nn.Conv2d(64, n_classes, 1)
        
    #     # Attention gates
    #     self.att4 = AttentionGate(512, 512, 256, init)
    #     self.att3 = AttentionGate(256, 256, 128, init)
    #     self.att2 = AttentionGate(128, 128, 64, init)
    #     self.att1 = AttentionGate(64, 64, 32, init)
        
    #     mutil.initialize_weights(self.outc, 0)
    
    # def forward(self, x):
    #     # Encoder
    #     e1 = self.enc1(x)
    #     p1 = self.pool1(e1)
    #     e2 = self.enc2(p1)
    #     p2 = self.pool2(e2)
    #     e3 = self.enc3(p2)
    #     p3 = self.pool3(e3)
    #     e4 = self.enc4(p3)
    #     p4 = self.pool4(e4)
    #     b = self.bottleneck(p4)
        
    #     # Decoder with attention
    #     u4 = self.up4(b)
    #     if u4.shape[2:] != e4.shape[2:]:
    #         u4 = F.interpolate(u4, size=e4.shape[2:], mode='bilinear', align_corners=False)
    #     att_e4 = self.att4(u4, e4)
    #     cat4 = torch.cat([u4, att_e4], dim=1)
    #     d4 = self.dec4(cat4)
        
    #     u3 = self.up3(d4)
    #     if u3.shape[2:] != e3.shape[2:]:
    #         u3 = F.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=False)
    #     att_e3 = self.att3(u3, e3)
    #     cat3 = torch.cat([u3, att_e3], dim=1)
    #     d3 = self.dec3(cat3)
        
    #     u2 = self.up2(d3)
    #     if u2.shape[2:] != e2.shape[2:]:
    #         u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
    #     att_e2 = self.att2(u2, e2)
    #     cat2 = torch.cat([u2, att_e2], dim=1)
    #     d2 = self.dec2(cat2)
        
    #     u1 = self.up1(d2)
    #     if u1.shape[2:] != e1.shape[2:]:
    #         u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
    #     att_e1 = self.att1(u1, e1)
    #     cat1 = torch.cat([u1, att_e1], dim=1)
    #     d1 = self.dec1(cat1)
        
    #     out = self.outc(d1)
    #     return out

# Example usage:
# model = UNetWithAttention(n_channels=3, n_classes=3, block_type='ConvNet', init='xavier')
# input_tensor = torch.randn(1, 3, 256, 256)
# output = model(input_tensor)
# print(output.shape)  # Should be torch.Size([1, 3, 256, 256])

class SequentialBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_in=3, channel_out=3, channel_base=20, gc=20, num_layers=4):
        super(SequentialBlock, self).__init__()
        
        channel = channel_in
        subnets, redisual = [], []
        for _ in range(num_layers - 1):
            block = subnet_constructor(channel, channel_base, gc)
            res_connection = nn.Conv2d(channel, channel_base, 1, bias=True)
            mutil.initialize_weights_zeros(res_connection)
            subnets.append(block)
            redisual.append(res_connection)
            channel = channel_base
        subnets.append(subnet_constructor(channel, channel_out, gc))
        redisual.append(nn.Conv2d(channel, channel_out, 1, bias=True))
        self.subnets = nn.ModuleList(subnets)
        self.redisual = nn.ModuleList(redisual)
        
        
    def forward(self, x):
        ori = x
        for i in range(len(self.subnets)):
            res = self.redisual[i](x)
            x = self.subnets[i](x) + res
        return x + ori

class Decompresser(nn.Module):
    def __init__(self, channel=3, block_type='CBAM', init='xavier'):
        super(Decompresser, self).__init__()
        subnet_constructor = subnet(block_type, init)
        self.net = SequentialBlock(subnet_constructor, channel, channel, channel_base=25, gc=25, num_layers=16)

    def forward(self, x):
        return self.net(x)