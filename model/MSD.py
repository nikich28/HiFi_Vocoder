import torch
from torch import nn
from configs.config import TaskConfig
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F


class MSDBlock(nn.Module):
    def __init__(self, norm='weight'):
        super().__init__()
        if norm == 'weight':
            self.norm = weight_norm
        else:
            self.norm = spectral_norm
        self.convs = nn.ModuleList([
            self.norm(nn.Conv1d(1, 128, kernel_size=15, stride=1)),
            self.norm(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4)),
            self.norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16)),
            self.norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16)),
            self.norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16)),
            self.norm(nn.Conv1d(1024, 1024, kernel_size=41, groups=16)),
            self.norm(nn.Conv1d(1024, 1024, kernel_size=5)),
            self.norm(nn.Conv1d(1024, 1, kernel_size=3))
        ])

    def forward(self, x):
        res = []
        for i in range(len(self.convs) - 1):
            x = F.leaky_relu(x, TaskConfig.leaky)
            res.append(x)
        out = self.convs[-1](x)
        res.append(out)
        return res

class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSDBlock('spectral'),
            MSDBlock(),
            MSDBlock()
        ])
        self.pool1 = nn.AvgPool1d(4, 2, 2)
        self.pool2 = nn.AvgPool1d(4, 2, 2)

    def forward(self, x):
        out_maps = []
        out_scores = []

        out1 = self.blocks[0](x)
        out_scores.append(out1[-1])
        out1.pop()
        out_maps.append(out1)

        x = self.pool1(x)

        out2 = self.blocks[1](x)
        out_scores.append(out2[-1])
        out2.pop()
        out_maps.append(out2)

        x = self.pool2(x)

        out3 = self.blocks[2](x)
        out_scores.append(out3[-1])
        out3.pop()
        out_maps.append(out3)

        return out_maps, out_scores
