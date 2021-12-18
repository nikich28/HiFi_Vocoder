from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def valid_epoch(model, disc, dataloader, criterions, featurizer, logger, epoch, melspec_config, config):
    model.eval()
    disc.eval()
    gen_cr, disc_cr = criterions
    for i, batch in tqdm(enumerate(dataloader), position=0, leave=True):
        batch = batch.to(config.device)

        spec = featurizer(batch.waveform)

        output = model(spec)
        # output - predicted wav

        mpd_fake, msd_fake = disc(output)
        mpd_real, msd_real = disc(batch.waveform)
        disc_loss = disc_cr(msd_fake[1], msd_real[1], mpd_fake[1], mpd_real[1])

        gen_out = model(spec)
        predicted_spec = featurizer(gen_out)
        mpd_fake, msd_fake = disc(gen_out)
        mpd_real, msd_real = disc(batch.waveform)

        spec_loss = nn.L1Loss(spec, predicted_spec)
        gen_loss = gen_cr(msd_fake[1], msd_fake[0], msd_real[0], mpd_fake[1], mpd_fake[0], mpd_real[0])
        gen_loss += 45 * spec_loss

        # log all loses
        logger.set_step(i + epoch * len(dataloader))
        logger.add_scalar('generator_loss', gen_loss.item())
        logger.add_scalar('spectrogram_loss', spec_loss.item())
        logger.add_scalar('discriminator_loss', disc_loss.item())

    if (epoch + 1) % config.show_every == 0:
        logger.add_audio("Ground_truth", batch.waveform[0], sample_rate=melspec_config.sr)
        logger.add_audio("predicted", output[0], sample_rate=melspec_config.sr)

        logger.add_image("Ground_truth_spect", plt.imshow(spec[0].detach().cpu().numpy()))
        logger.add_image("Predicted_spect", plt.imshow(predicted_spec[0].detach().cpu().numpy()))
