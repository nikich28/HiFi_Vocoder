from torch import nn
from torch.nn.utils import weight_norm
from .MRF import MRF
from configs.config import TaskConfig


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = TaskConfig.mel_channels
        hidden_ch = TaskConfig.chs
        self.first_conv = weight_norm(nn.Conv1d(channels, hidden_ch, kernel_size=7, dilation=1, padding=3)
        self.act = nn.Tanh()
        self.last_act = nn.LeakyReLU(TaskConfig.leaky)

        tmp_channels = hidden_ch
        self.net = []
        for kernel in TaskConfig.kernels_up:
            new_block = [
                nn.LeakyReLU(TaskConfig.leaky),
                weight_norm(nn.Conv1d(tmp_channels, tmp_channels // 2, kernel_size=kernel, stride=kernel // 2)),
                MRF(tmp_channels // 2)
            ]
            self.net.append(nn.Sequential(*new_block))
            tmp_channels //= 2

        self.last_conv = weight_norm(nn.Conv1d(hidden_ch, 1, kernel_size=7, dilation=1, padding=3)

    def forward(self, x):
        x = self.first_conv(x)
        for b in self.net:
            x = b(x)
        x = self.last_conv(self.last_act(x))
        return self.act(x).squeeze(1)
