import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.Subnet_constructor import subnet

class SequentialBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_in=3, channel_out=3, channel_base=20, gc=20, num_layers=4):
        super(SequentialBlock, self).__init__()
        
        channel = channel_in
        subnets = []
        for _ in range(num_layers - 1):
            block = subnet_constructor(channel, channel_base, gc)
            subnets.append(block)
            channel = channel_base
        subnets.append(subnet_constructor(channel, channel_out, gc))
        self.subnets = nn.ModuleList(subnets)
        
        
    def forward(self, x):
        # ori = x
        for i in range(len(self.subnets)):
            x = self.subnets[i](x)
        return x

class Decompresser(nn.Module):
    def __init__(self, channel=3, block_type='CBAM', init='xavier'):
        super(Decompresser, self).__init__()
        subnet_constructor = subnet(block_type, init)
        self.net = SequentialBlock(subnet_constructor, channel, channel, channel_base=25, gc=25, num_layers=16)

    def forward(self, x):
        return self.net(x)
    
class DifferenceEncoder(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, block_type='CBAM', init='xavier'):
        super(DifferenceEncoder, self).__init__()
        subnet_constructor = subnet(block_type, init)
        self.net = SequentialBlock(subnet_constructor, channel_in, channel_out, channel_base=20, gc=20, num_layers=8)

    def forward(self, x):
        return self.net(x)