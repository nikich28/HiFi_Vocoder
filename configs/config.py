from dataclasses import dataclass
import torch


@dataclass
class TaskConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 64
    kernels = [3, 7, 11]
    dilations = [[1, 3, 5]] * 3
    leaky: float = 0.1
    chs: int = 512
    num_blocks: int = 3
    mel_channels: int = 80
    kernels_up = [16, 16, 4, 4]
    disc_periods = [2, 3, 5, 7, 11]

    warmup: int = 4000
    n_epochs: int = 80
    show_every: int = 2
    save_every: int = 10
    lr: float = 2e-4
    project_name: str = 'hifi_vocoder'
