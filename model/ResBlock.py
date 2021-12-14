from torch import nn
from torch.nn.utils import weight_norm
from configs.config import TaskConfig


class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size, i, dilations=TaskConfig.dilations, leaky_coef=TaskConfig.leaky):
        super().__init__()
        self.convs = nn.ModuleList([])
        for j in range(len(dilations[i])):
            tmp_conv = [
                nn.LeakyReLU(leaky_coef),
                weight_norm(nn.Conv1d(ch, ch, kernel_size, dilation=dilations[i][j],
                                        padding=dilations[i][j] * (kernel_size - 1) // 2)),
                nn.LeakyReLU(leaky_coef),
                weight_norm(nn.Conv1d(ch, ch, kernel_size, dilation=1,
                                        padding=(kernel_size - 1) // 2))
            ]

            self.convs.append(nn.Sequential(*tmp_conv))

    def forward(self, x):
        for c in self.convs:
            x = x + c(x)
        return x
