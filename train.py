from ast import expr_context
from lib2to3.refactor import get_all_fix_names
import os.path as osp
from argparse import ArgumentParser
# import wandb
from mmcv import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import MODELS
import random
import numpy as np
import torch

from tensorboardX import SummaryWriter
from datasets.nusc_test import Testloader
from datasets.common import NUSCENES_ROOT


def parse_args():
    parser = ArgumentParser(description='Training with DDP.')
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--work_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--test', type=int, default=False)
    args = parser.parse_args()
    return args


def main():
    # parse args
    args = parse_args()
    cfg = Config.fromfile(osp.join(f'configs/{args.config}.yaml'))
    cfg.test = args.test
    cfg.seed = args.seed
    exp_name = "joint-depth: lr:{}, s_weight:{}, adversarial:{}, illu_mask{}". \
                format(cfg.model.G_learning_rate, cfg.model.S_weight, cfg.model.day_check_point, cfg.model.use_illu_mask)
    # wandb.init(project="joint-depth", config=cfg, name=exp_name)
    # parse cfg


    # show information
    print(f'Now training with {args.config}...')

    # configure seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    seed_everything(args.seed)

    # prepare data loader
    dataset = build_dataset(cfg.dataset)
    loader = DataLoader(dataset, cfg.imgs_per_gpu, shuffle=True, num_workers=cfg.workers_per_gpu, drop_last=True)
    
    test_dir = NUSCENES_ROOT["test_color"]
    gt_dir = NUSCENES_ROOT["test_gt"]
    val_dataset = Testloader(test_dir, gt_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False) 

    if cfg.model.name == 'rnw':
        cfg.data_link = dataset

    # define model
    model = MODELS.build(name=cfg.model.name, option=cfg)

    # define trainer
    work_dir = osp.join(args.work_dir, args.config)
    # save checkpoint every 'cfg.checkpoint_epoch_interval' epochs
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          save_weights_only=True,
                                          save_top_k=-1,
                                          filename='checkpoint_{epoch}',
                                          every_n_epochs=1)
    trainer = Trainer(accelerator='ddp',
                      default_root_dir=work_dir,
                      gpus=args.gpus,
                      num_nodes=1,
                      max_epochs=cfg.total_epochs,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=0,
                      deterministic=False)

    # training
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
