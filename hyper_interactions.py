import torch
import torch.nn as nn
from hyper_zzw import HyperZZW

class HyperChannelInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.hyper_zzw = HyperZZW(channels, 1, is_global=True)
        
    def forward(self, x):
        z_c = self.avg_pool(x)
        weights = torch.sigmoid(self.hyper_zzw(z_c))
        return x * weights

class HyperInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_interaction = HyperChannelInteraction(channels)
        self.spatial_hyper_zzw = HyperZZW(1, 1, is_global=True)
        
    def forward(self, x):
        z_c = self.channel_interaction(x)
        z_s = x.mean(1, keepdim=True)
        spatial_weights = torch.sigmoid(self.spatial_hyper_zzw(z_s))
        return x * z_c * spatial_weights