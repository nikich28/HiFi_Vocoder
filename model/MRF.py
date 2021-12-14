from torch import nn
from .ResBlock import ResBlock
from configs.config import TaskConfig


class MRF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(channels, kernel) for kernel in TaskConfig.kernels])

    def forward(self, x):
        for i in range(len(self.blocks)):
            if i != 0:
                x = x + self.blocks[i](x)
            else:
                x = self.blocks[i](x)
        return x / len(self.blocks)
