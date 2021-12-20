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


if __name__ == '__main__':
    # create config
    config = TaskConfig()

    # create utils
    melspec_config = MelSpectrogramConfig()
    featurizer = MelSpectrogram(melspec_config).to(config.device)

    # model
    model = Generator().to(config.device)
    model.load_state_dict(torch.load('best_model.pth')['generator'])

    # wandb
    logger = WanDBWriter(config)

    # training
    test_epoch(model, featurizer, logger, 0, melspec_config, config)
