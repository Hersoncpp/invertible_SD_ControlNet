import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiffEncoder(nn.Module):
    def __init__(self, subnet_constructor, in_channels=3, out_channels=2, base_channels=20, init='xavier', gc=20, num_layers=4):
        super(DiffEncoder, self).__init__()
        
        channels = in_channels
        encoder = []
        for _ in range(num_layers - 1):
            block = subnet_constructor(channels, base_channels, init, gc)
            encoder.append(block)
            channels = base_channels
        encoder.append(subnet_constructor(channels, out_channels, init, gc))
        self.encoder = nn.Sequential(*encoder)
        
    def forward(self, x):
        return self.encoder(x)