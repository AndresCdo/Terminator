import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveGatedUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.wk = nn.Conv2d(channels, channels, 1)
        self.wv = nn.Conv2d(channels, channels, 1)
        self.wy = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        k = self.wk(x)
        v = self.wv(x)
        q = F.gelu(self._instance_standardization(k * self.wv(x)))
        return self.wy(q)

    def _instance_standardization(self, x):
        return (x - x.mean([2, 3], keepdim=True)) / (x.std([2, 3], keepdim=True) + 1e-5)