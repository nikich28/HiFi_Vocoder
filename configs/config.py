from dataclasses import dataclass
import torch
from typing import List


@dataclass
class TaskConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 64
    kernels: List[int] = [3, 7, 11]
    dilations: List[List[int]] = [[1, 3, 5]] * 3
    leaky: float = 0.1
    chs: int = 512
    num_blocks: int = 3
    mel_channels: int = 80
    gen_hidden_size: int = 32
    kernels_up: List[int] = [16, 16, 4, 4]

    warmup: int = 4000
    n_epochs: int = 100
    show_every: int = 20
    lr: float = 1e-4
    project_name: str = 'vocoder'
