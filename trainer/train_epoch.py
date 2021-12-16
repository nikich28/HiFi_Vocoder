from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def train_epoch(model, scheduler, dataloader, criterion, featurizer, logger, epoch, melspec_config, config):
    model.train()
    for i, batch in tqdm(enumerate(dataloader), position=0, leave=True):
        batch = batch.to(config.device)

        spect = featurizer(batch.waveform)

        scheduler.zero_grad()
        output = model(spect)
        # print(batch.waveform.size())
        # print(spect.size())
        # print(output.size())

        predicted_spect = featurizer(output)

        loss = criterion(spect, predicted_spect)
        loss.backward()
        scheduler.step()
        # log all loses
        logger.set_step(i + epoch * len(dataloader))
        logger.add_scalar('loss', loss.item())
        # logger.add_scalar('spect_loss', losses[0].item())
        # logger.add_scalar('duration_loss', losses[1].item())
        # logger.add_scalar('combined loss', loss.item())

    # if (epoch + 1) % config.show_every == 0:
    #     torch.save(model.state_dict(), f"best_model_{epoch + 1}.pth")

    if (epoch + 1) % config.show_every == 0:
        logger.add_audio("Ground_truth", batch.waveform[0], sample_rate=melspec_config.sr)
        logger.add_audio("predicted", output[0], sample_rate=melspec_config.sr)

        logger.add_image("Ground_truth_spect", plt.imshow(spect[0].detach().cpu().numpy()))
        logger.add_image("Predicted_spect", plt.imshow(predicted_spect[0].detach().cpu().numpy()))
