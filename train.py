import torch

from model.Generator import Generator
from model.Discriminator import Discriminator
from configs.config import TaskConfig
from loss.gen_loss import GenLoss
from loss.disc_loss import DiscLoss
from dataset.LJSpeechDataset import LJSpeechDataset
from dataset.DatasetCollator import LJSpeechCollator
from utils.featurizer import MelSpectrogramConfig, MelSpectrogram
import numpy as np
from trainer.train_epoch import train_epoch
from trainer.test_epoch import test_epoch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from logger.logger import WanDBWriter

torch.manual_seed(57)
torch.cuda.manual_seed(57)
torch.cuda.manual_seed_all(57)
np.random.seed(57)
torch.backends.cudnn.deterministic = True

root = '../'


def train(model, disc, dataloader, optims, schedulers, criterions, featurizer, logger, melspec_config, config):
    for epoch in range(config.n_epochs):
        print(f'Start of the epoch {epoch}')
        train_epoch(model, disc, optims, schedulers, dataloader, criterions, featurizer, logger, epoch,
                    melspec_config, config)

        if (epoch + 1) % config.show_every == 0:
            test_epoch(model, featurizer, logger, epoch, melspec_config, config)


if __name__ == '__main__':
    # create config
    config = TaskConfig()

    # create utils
    melspec_config = MelSpectrogramConfig()
    featurizer = MelSpectrogram(melspec_config).to(config.device)

    # data
    dataset = LJSpeechDataset(root=root, size_of_wav=8192)
    dataloader = DataLoader(dataset, num_workers=4, pin_memory=True, batch_size=config.batch_size,
                            collate_fn=LJSpeechCollator()
                            )

    # model
    model = Generator().to(config.device)
    disc = Discriminator().to(config.device)

    # optmizations
    criterions = [GenLoss(), DiscLoss()]
    g_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.8, 0.99), eps=1e-9)
    d_optimizer = torch.optim.AdamW(disc.parameters(), lr=config.lr, betas=(0.8, 0.99), eps=1e-9)
    schedulers = [ExponentialLR(g_optimizer, gamma=0.999),
                  ExponentialLR(d_optimizer, gamma=0.999)]

    # wandb
    logger = WanDBWriter(config)

    # training
    train(model, disc, dataloader, [g_optimizer, d_optimizer], schedulers, criterions, featurizer, logger,
          melspec_config, config)
