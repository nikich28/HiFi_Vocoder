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
from trainer.valid_epoch import valid_epoch
from torch.utils.data import DataLoader
import wandb
from logger.logger import WanDBWriter

torch.manual_seed(57)
torch.cuda.manual_seed(57)
torch.cuda.manual_seed_all(57)
np.random.seed(57)
torch.backends.cudnn.deterministic = True

root = '../'


# https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer

class CustomScheduler:
    def __init__(self, model_size, optimizer, warmup, factor=2):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._step = 0

    def rate(self, step):
        return 1 / self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        rate = self.rate(self._step)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()


def train(model, disc, dataloader, test_texts, optims, schedulers, criterions, featurizer, logger,
          melspec_config, config):
    gen_sch, disc_sch = schedulers
    for epoch in range(config.n_epochs):
        print(f'Start of the epoch {epoch}')
        train_epoch(model, disc, optims, dataloader, criterions, featurizer, logger, epoch, melspec_config, config)
        gen_sch.step()
        disc_sch.step()
        # if (epoch + 1) % config.show_every == 0:
        #     valid(model, disc, test_texts, criterions, featurizer, logger, epoch, melspec_config, config)


if __name__ == '__main__':
    # create config
    config = TaskConfig()

    test_texts = [
        'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
        'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
        'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space'
    ]

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
    train(model, disc, dataloader, test_texts, [g_optimizer, d_optimizer], schedulers, criterions, featurizer, logger,
          melspec_config, config)
