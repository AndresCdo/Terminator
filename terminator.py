import torch
import torch.nn as nn
import torch.nn.functional as F
from sfne_block import SFNEBlock

class Terminator(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            SFNEBlock(input_channels, 64),
            SFNEBlock(64, 128),
            SFNEBlock(128, 256),
            SFNEBlock(256, 512)
        ])
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return self.classifier(x)

    def slow_neural_loss(self):
        loss = 0
        for i in range(1, len(self.blocks)):
            loss += F.mse_loss(
                self.blocks[i].global_hyper_zzw.mlp.layers[-1].weight,
                self.blocks[i-1].global_hyper_zzw.mlp.layers[-1].weight
            )
        return loss