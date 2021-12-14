from dataclasses import dataclass
import torch


@dataclass
class TaskConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 8
    kernels = [3, 7, 11]
    dilations = [[1, 3, 5]] * 3
    leaky: float = 0.1
    chs: int = 512
    num_blocks: int = 3
    mel_channels: int = 80
    kernels_up = [16, 16, 4, 4]

    warmup: int = 4000
    n_epochs: int = 5000
    show_every: int = 100
    lr: float = 1e-4
    project_name: str = 'vocoder'
