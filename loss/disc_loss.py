import torch
from torch import nn


class DiscLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, msd_fake_score, msd_real_score, mpd_fake_score, mpd_real_score):
        loss = 0
        for i in range(len(msd_fake_score)):
            fake, real = msd_fake_score[i], msd_real_score[i]
            loss += self.mse(real, torch.ones_like(real)) + self.mse(fake, torch.zeros_like(fake))

        for i in range(len(mpd_fake_score)):
            fake, real = mpd_fake_score[i], mpd_real_score[i]
            loss += self.mse(real, torch.ones_like(real)) + self.mse(fake, torch.zeros_like(fake))

        return loss
