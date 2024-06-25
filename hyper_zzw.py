import torch
import torch.nn as nn
import torch.nn.functional as F
from coordinate_mlp import CoordinateBasedMLP

class HyperZZW(nn.Module):
    def __init__(self, channels, kernel_size, is_global=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.is_global = is_global
        
        input_dim = 2 if is_global else 2
        output_dim = channels * (kernel_size * kernel_size if not is_global else 1)
        self.mlp = CoordinateBasedMLP(input_dim, 64, output_dim, 4)
        
    def forward(self, z):
        b, c, h, w = z.shape
        if self.is_global:
            kernel = self.mlp(self._get_coords(h, w)).view(1, c, h, w)
            return z * kernel
        else:
            kernel = self.mlp(self._get_coords(self.kernel_size, self.kernel_size)).view(c, 1, self.kernel_size, self.kernel_size)
            return F.conv2d(z, kernel, padding=self.kernel_size//2, groups=c)

    def _get_coords(self, h, w):
        coords_h = torch.linspace(-1, 1, h)
        coords_w = torch.linspace(-1, 1, w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        return coords.view(-1, 2)