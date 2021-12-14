import torch
from torch.nn.utils.rnn import pad_sequence
from .LJSpeechDataset import Batch
from typing import List, Tuple, Dict


class LJSpeechCollator:
    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        return Batch(waveform, waveform_length)
