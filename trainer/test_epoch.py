from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import torchaudio


@torch.no_grad()
def test_epoch(model, featurizer, logger, epoch, melspec_config, config):
    model.eval()

    path = 'test_data'

    files = os.listdir(path)
    for file in files:
        path_ = os.path.join(path, file)
        wav, sr = torchaudio.load(path_)
        waveform = wav.to(config.device)

        spec = featurizer(waveform)

        gen_out = model(spec)
        predicted_spec = featurizer(gen_out)

        # log all loses
        tmp_step = logger.step
        logger.set_step(tmp_step, mode='test')

        logger.add_audio(f"Ground_truth_{file}", waveform[0], sample_rate=melspec_config.sr)
        logger.add_audio(f"Predicted_{file}", gen_out[0], sample_rate=melspec_config.sr)

        # logger.add_image(f"Ground_truth_spect_{file}", plt.imshow(spec[0].detach().cpu().numpy()))
        # logger.add_image(f"Predicted_spect_{file}", plt.imshow(predicted_spec[0].detach().cpu().numpy()))
