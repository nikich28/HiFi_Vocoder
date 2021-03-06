from torch import nn
import torch
from utils.featurizer import MelSpectrogramConfig


class GenLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, msd_fake_score, msd_fake_map, msd_real_map, mpd_fake_score, mpd_fake_map, mpd_real_map):

        scores_loss = 0
        spec_loss = 0

        for i in range(len(msd_fake_map)):
            fake, real = msd_fake_map[i], msd_real_map[i] #maps
            for j in range(len(fake)):
                f, r = fake[j], real[j]
                spec_loss += self.loss(f, r)

            scores_loss += (msd_fake_score[i] - 1).square().mean()

        for i in range(len(mpd_fake_map)):
            fake, real = mpd_fake_map[i], mpd_real_map[i]  # maps
            for j in range(len(fake)):
                f, r = fake[j], real[j]
                spec_loss += self.loss(f, r)

            scores_loss += (mpd_fake_score[i] - 1).square().mean()

        return scores_loss + 2 * spec_loss


def L1LOSS(real, fake):
    loss = nn.L1Loss()
    if real.size(-1) > fake.size(-1):
        padded = nn.functional.pad(fake, (0, real.size(-1) - fake.size(-1)), value=MelSpectrogramConfig.pad_value)
        return loss(padded, real)
    padded = nn.functional.pad(real, (0, fake.size(-1) - real.size(-1)), value=MelSpectrogramConfig.pad_value)
    return loss(fake, padded)
