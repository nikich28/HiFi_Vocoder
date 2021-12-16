from torch import nn
from utils.featurizer import MelSpectrogramConfig


class GenLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, real, fake):
        if real.size(-1) > fake.size(-1):
            padded = nn.functional.pad(fake, (0, real.size(-1) - fake.size(-1)), value=MelSpectrogramConfig.pad_value)
            return self.mse(padded, real)
        padded = nn.functional.pad(real, (0, fake.size(-1) - real.size(-1)), value=MelSpectrogramConfig.pad_value)
        return self.loss(fake, padded)
