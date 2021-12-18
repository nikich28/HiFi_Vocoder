import torch
from torch import nn
from configs.config import TaskConfig
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class MPDBlock(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([])
        chs = 1
        for i in range(1, 5):
            self.convs.append(nn.Sequential(
                weight_norm(nn.Conv1d(chs, 2 ** (5 + i), kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(TaskConfig.leaky)
            ))
            chs = 2 ** (5 + i)

        self.convs.append(nn.Sequential(
            weight_norm(nn.Conv1d(chs, 1024, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))),
            nn.LeakyReLU(TaskConfig.leaky)
        ))

        self.convs.append(weight_norm(nn.Conv1d(1024, 1, kernel_size=(3, 1), stride=(1, 1))))

    def forward(self, x):
        x = F.pad(x, (0, -x.shape[-1] % self.period), 'reflect')
        x = x.view(x.size(0), x.size(1), -1, self.period)

        res = []
        for l in self.convs:
            x = l(x)
            res.append(x)

        return res


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = TaskConfig.disc_periods
        self.blocks = nn.ModuleList([])
        for p in self.periods:
            self.blocks.append(MPDBlock(p))

    def forward(self, x):
        out_maps = []
        out_scores = []
        for b in self.blocks:
            out = b(x)
            s = out[-1]
            out.pop()
            out_maps.append(out)
            out_scores.append(s)
        return out_maps, out_scores
