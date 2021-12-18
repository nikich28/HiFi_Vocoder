from torch import nn
from utils.featurizer import MelSpectrogramConfig


class GenLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, msd_fake_score, msd_fake_map, msd_real_map, mpd_fake_score, mpd_fake_map, mpd_real_map):
        # if real.size(-1) > fake.size(-1):
        #     padded = nn.functional.pad(fake, (0, real.size(-1) - fake.size(-1)), value=MelSpectrogramConfig.pad_value)
        #     return self.mse(padded, real)
        # padded = nn.functional.pad(real, (0, fake.size(-1) - real.size(-1)), value=MelSpectrogramConfig.pad_value)
        # return self.loss(fake, padded)

        scores_loss = 0
        spec_loss = 0

        for i in range(len(msd_fake_map)):
            fake, real = msd_fake_map[i], msd_real_map[i] #maps
            for j in range(len(fake)):
                f, r = fake[j], real[j]
                spec_loss += self.loss(f, r)

            scores_loss += self.mse(msd_fake_score[i], torch.ones_like(msd_fake_score[i]))

        for i in range(len(mpd_fake_map)):
            fake, real = mpd_fake_map[i], mpd_real_map[i]  # maps
            for j in range(len(fake)):
                f, r = fake[j], real[j]
                spec_loss += self.loss(f, r)

            scores_loss += self.mse(mpd_fake_score[i], torch.ones_like(mpd_fake_score[i]))

        return scores_loss + 2 * spec_loss
