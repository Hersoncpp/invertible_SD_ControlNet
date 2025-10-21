import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=20, bias=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d( gc, gc, 3, 1, 1, bias=bias)
        #self.conv3 = nn.Conv2d( gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d( gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d( gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2,  self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1) )
        #x3 = self.lrelu(self.conv3(x2) )
        x4 = self.lrelu(self.conv4(x2) + x1)
        x5 = self.conv5(x4)

        return x5

class ChannelAttention(nn.Module):
    def __init__(self, channel, bias=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Conv2d(channel, channel, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.lrelu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.lrelu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bias=True):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class AttentionBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=20, bias=True):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.ca = ChannelAttention(gc, bias)
        self.sa = SpatialAttention(bias=bias)
        self.conv_out = nn.Conv2d(gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv_out, self.ca.fc1, self.ca.fc2, self.sa.conv], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv_out, self.ca.fc1, self.ca.fc2, self.sa.conv], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1) + x1)
        x3 = self.lrelu(self.conv3(x2) + x1)
        x4 = self.lrelu(self.conv4(x3) + x1)
        
        x4 = x4 * self.ca(x4) * self.sa(x4)
        out = self.conv_out(x4)
        return out
        

def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        if net_structure == 'ConvNet':
            if init == 'xavier':
                return ConvBlock(channel_in, channel_out, init)
            else:
                return ConvBlock(channel_in, channel_out)
        if net_structure == 'CBAM':
            if init == 'xavier':
                return AttentionBlock(channel_in, channel_out, init)
            else:
                return AttentionBlock(channel_in, channel_out)

    return constructor
