import torch
import torch.nn as nn

class GIBS(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = channels // num_groups
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, self.num_groups, self.group_size, h, w)
        
        mean = x.mean([2, 3, 4], keepdim=True)
        var = x.var([2, 3, 4], keepdim=True)
        
        x_even = (x[:, 0::2] - mean[:, 0::2]) / (var[:, 0::2] + 1e-5).sqrt()  # Batch Standardization
        x_odd = (x[:, 1::2] - x[:, 1::2].mean([3, 4], keepdim=True)) / (x[:, 1::2].var([3, 4], keepdim=True) + 1e-5).sqrt()  # Instance Standardization
        
        x = torch.stack([x_even, x_odd], dim=2).view(b, c, h, w)
        return x