import torch
import torch.nn as nn
from hyper_zzw import HyperZZW
from recursive_gated_unit import RecursiveGatedUnit
from hyper_interactions import HyperInteraction
from gibs import GIBS

class SFNEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_mixer = nn.Conv2d(in_channels, out_channels, 1)
        self.rgu = RecursiveGatedUnit(out_channels)
        self.global_hyper_zzw = HyperZZW(out_channels, 1, is_global=True)
        self.local_hyper_zzw = HyperZZW(out_channels, 7)
        self.hyper_interaction = HyperInteraction(out_channels)
        self.bottleneck = nn.Conv2d(out_channels * 4, out_channels, 1)
        self.gibs = GIBS(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.channel_mixer(x)
        rgu_out = self.rgu(x)
        global_zzw_out = self.global_hyper_zzw(x)
        local_zzw_out = self.local_hyper_zzw(x)
        hyper_int_out = self.hyper_interaction(x)
        combined = torch.cat([rgu_out, global_zzw_out, local_zzw_out, hyper_int_out], dim=1)
        out = self.bottleneck(combined)
        out = self.gibs(out)
        out = self.gelu(out)
        out = self.dropout(out)
        return out