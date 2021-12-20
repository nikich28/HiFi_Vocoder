from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
from loss.gen_loss import L1LOSS
import torch


def train_epoch(model, disc, optims, schedulers, dataloader, criterions, featurizer, logger, epoch,
                melspec_config, config):
    model.train()
    disc.train()
    gen_cr, disc_cr = criterions
    gen_opt, disc_opt = optims
    gen_sch, disc_sch = schedulers
    l1loss = nn.L1Loss()
    for i, batch in tqdm(enumerate(dataloader), position=0, leave=True, total=len(dataloader)):
        batch = batch.to(config.device)
        spec = featurizer(batch.waveform)

        output = model(spec)
        # output - predicted wav
        out_d = output.detach()
        output = output[:, :batch.waveform.size(1)]

        disc_opt.zero_grad()

        mpd_fake, msd_fake = disc(out_d)
        mpd_real, msd_real = disc(batch.waveform)
        disc_loss = disc_cr(msd_fake[1], msd_real[1], mpd_fake[1], mpd_real[1])
        disc_loss.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        predicted_spec = featurizer(output)

        mpd_fake, msd_fake = disc(output)
        mpd_real, msd_real = disc(batch.waveform)

        spec_loss = l1loss(spec, predicted_spec)
        gen_loss = gen_cr(msd_fake[1], msd_fake[0], msd_real[0], mpd_fake[1], mpd_fake[0], mpd_real[0])
        gen_loss += 45 * spec_loss
        gen_loss.backward()
        gen_opt.step()

        # log all loses
        logger.set_step(i + epoch * len(dataloader))
        logger.add_scalar('generator_loss', gen_loss.item())
        logger.add_scalar('spectrogram_loss', spec_loss.item())
        logger.add_scalar('discriminator_loss', disc_loss.item())

    gen_sch.step()
    disc_sch.step()
    if (epoch + 1) % config.save_every == 0:
        torch.save({
                'generator': model.state_dict(),
                'discriminator': disc.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc_opt': disc_opt.state_dict(),
                'gen_sch': gen_sch.state_dict(),
                'disc_sch': disc_sch.state_dict()
        }, f"best_model_{epoch + 1}.pth")

    if (epoch + 1) % config.show_every == 0:
        logger.add_audio("Ground_truth", batch.waveform[0], sample_rate=melspec_config.sr)
        logger.add_audio("predicted", output[0], sample_rate=melspec_config.sr)

        logger.add_image("Ground_truth_spect", plt.imshow(spec[0].detach().cpu().numpy()))
        logger.add_image("Predicted_spect", plt.imshow(predicted_spec[0].detach().cpu().numpy()))
