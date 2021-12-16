import torch
import torchaudio
from dataclasses import dataclass
import random


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor

    def to(self, device: torch.device, non_blocking=False) -> 'Batch':
        self.waveform = self.waveform.to(device, non_blocking=non_blocking)

        return self


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, root, size_of_wav=8912):
        super().__init__(root=root)
        self.wav_size = size_of_wav

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        if waveform.shape[1] <= self.wav_size:
            return waveform, waveform_length
        idx = random.randint(0, waveform.shape[1] - self.wav_size)
        return waveform[:, idx: idx + self.wav_size], waveform_length

    # def decode(self, tokens, lengths):
    #     result = []
    #     for tokens_, length in zip(tokens, lengths):
    #         text = "".join([
    #             self._tokenizer.tokens[token]
    #             for token in tokens_[:length]
    #         ])
    #         result.append(text)
    #     return result
