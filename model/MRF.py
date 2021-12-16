from torch import nn
from .ResBlock import ResBlock
from configs.config import TaskConfig


class MRF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(channels, kernel, i) for i, kernel in enumerate(TaskConfig.kernels)])

    def forward(self, x):
        ans = 0
        for i in range(len(self.blocks)):
            if i != 0:
                ans = ans + self.blocks[i](x)
            else:
                ans = self.blocks[i](x)
        return ans / len(self.blocks)
