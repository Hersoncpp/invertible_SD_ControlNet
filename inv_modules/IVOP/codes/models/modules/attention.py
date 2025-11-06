import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

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

class SelfAttention(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=24, bias=True):
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.skip = nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=bias)
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
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        
        x4 = x4 * self.ca(x4) * self.sa(x4)
        out = self.conv_out(x4) + self.skip(x)
        return out
    
class CrossChannelAttention(nn.Module):
    def __init__(self, channel, bias=True):
        super(CrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel * 2, channel, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2_x = nn.Conv2d(channel, channel, 1, bias=bias)
        self.fc2_y = nn.Conv2d(channel, channel, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_y = self.avg_pool(y)
        max_y = self.max_pool(y)
        
        cat_avg = torch.cat([avg_x, avg_y], dim=1)
        cat_max = torch.cat([max_x, max_y], dim=1)
        
        shared_avg = self.lrelu(self.fc1(cat_avg))
        shared_max = self.lrelu(self.fc1(cat_max))
        
        ca_x = self.sigmoid(self.fc2_x(shared_avg) + self.fc2_x(shared_max))
        ca_y = self.sigmoid(self.fc2_y(shared_avg) + self.fc2_y(shared_max))
        
        return ca_x, ca_y

class CrossSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bias=True):
        super(CrossSpatialAttention, self).__init__()
        self.conv_x = nn.Conv2d(4, 1, kernel_size, padding=kernel_size//2, bias=bias)
        self.conv_y = nn.Conv2d(4, 1, kernel_size, padding=kernel_size//2, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        avg_y = torch.mean(y, dim=1, keepdim=True)
        max_y, _ = torch.max(y, dim=1, keepdim=True)
        
        cat_x = torch.cat([avg_x, max_x, avg_y, max_y], dim=1)
        cat_y = torch.cat([avg_y, max_y, avg_x, max_x], dim=1)
        
        sa_x = self.sigmoid(self.conv_x(cat_x))
        sa_y = self.sigmoid(self.conv_y(cat_y))
        
        return sa_x, sa_y

class CrossAttention(nn.Module):
    def __init__(self, channel_in_x, channel_in_y, channel_out, init='xavier', gc=24, bias=True):
        super(CrossAttention, self).__init__()
        self.conv1_x = nn.Conv2d(channel_in_x, gc, 3, 1, 1, bias=bias)
        self.conv2_x = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv3_x = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv4_x = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        
        self.conv1_y = nn.Conv2d(channel_in_y, gc, 3, 1, 1, bias=bias)
        self.conv2_y = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv3_y = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv4_y = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        
        self.cca = CrossChannelAttention(gc, bias)
        self.csa = CrossSpatialAttention(bias=bias)
        self.conv_out = nn.Conv2d(gc * 2, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        layers_to_init = [self.conv1_x, self.conv2_x, self.conv3_x, self.conv4_x,
                          self.conv1_y, self.conv2_y, self.conv3_y, self.conv4_y,
                          self.conv_out,
                          self.cca.fc1, self.cca.fc2_x, self.cca.fc2_y,
                          self.csa.conv_x, self.csa.conv_y]
        if init == 'xavier':
            mutil.initialize_weights_xavier(layers_to_init, 0.1)
        else:
            mutil.initialize_weights(layers_to_init, 0.1)

    def forward(self, x, y):
        x1 = self.lrelu(self.conv1_x(x))
        x2 = self.lrelu(self.conv2_x(x1))
        x3 = self.lrelu(self.conv3_x(x2))
        x4 = self.lrelu(self.conv4_x(x3))
        
        y1 = self.lrelu(self.conv1_y(y))
        y2 = self.lrelu(self.conv2_y(y1))
        y3 = self.lrelu(self.conv3_y(y2))
        y4 = self.lrelu(self.conv4_y(y3))
        
        ca_x, ca_y = self.cca(x4, y4)
        sa_x, sa_y = self.csa(x4, y4)
        
        x_att = x4 * ca_x * sa_x
        y_att = y4 * ca_y * sa_y
        
        fused = torch.cat([x_att, y_att], dim=1)
        out = self.conv_out(fused)
        return out