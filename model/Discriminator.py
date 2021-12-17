import torch
from torch import nn
from .MPD import MPD
from .MSD import MSD


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = MSD
        self.mpd = MPD

    def forward(self, x):
        return self.mpd(x), self.msd(x)
