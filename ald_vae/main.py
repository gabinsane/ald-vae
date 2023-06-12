import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from models.trainer import ALDVAE
from models.config_cls import Config
from models.dataloader import DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg", help="Specify config file", metavar="FILE")
parser.add_argument('--viz_freq', type=int, default=None,
                    help='frequency of visualization savings (number of iterations)')
parser.add_argument('--compare', action='store_true', help='Whether to perform grid search')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Size of the training batch')
parser.add_argument('--obj', type=str, metavar='O', default=None,
                    help='objective to use (moe_elbo/poe_elbo_semi)')
parser.add_argument('--loss', type=str, metavar='O', default=None,
                    help='loss to use (lprob/bce)')
parser.add_argument('--n_latents', type=int, default=None,
                    help='latent vector dimensionality')
parser.add_argument('--pre_trained', type=str, default=None,
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--seed', type=int, metavar='S', default=None,
                    help='seed number')
parser.add_argument('--exp_name', type=str, default=None,
                    help='name of folder')
parser.add_argument('--optimizer', type=str, default=None,
                    help='optimizer')

def main():
    config = Config(parser)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    data_module = DataModule(config)
    model_wrapped = ALDVAE(config, data_module.get_dataset_class().feature_dims)
    logger1 = TensorBoardLogger(config.mPath, log_graph=True, name="tensorboard")
    logger2 = CSVLogger(save_dir=config.mPath, name="csv", flush_logs_every_n_steps=1)
    trainer_kwargs = {"accelerator":"gpu",
                      "default_root_dir": config.mPath, "max_epochs": config.epochs, "check_val_every_n_epoch": 1,
                      "logger":[logger1, logger2]}
    pl_trainer = pl.Trainer(**trainer_kwargs)
    pl_trainer.fit(model_wrapped, datamodule=data_module)
    pl_trainer.test(ckpt_path="best", datamodule=data_module)

def identity(string):
    return string


if __name__ == '__main__':
    parser.register('type', None, identity)
    args = parser.parse_args()
    main()
