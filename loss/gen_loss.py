import torch
from torch import nn


class GenLoss(nn.Module):
    def __init__(self):
        self.mse = nn.L1Loss()

    def forward(self, real, fake):
        if real.size(-1) > fake.size(-1):
            padded = nn.functional.pad(fake, (0, real.size(-1) - fake.size(-1)))
            return self.mse(padded, real)
        padded = nn.functional.pad(real, (0, fake.size(-1) - real.size(-1)))
        return self.mse(fake, padded)
