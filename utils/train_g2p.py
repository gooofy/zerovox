#!/bin/env python3

import os
import argparse
import shutil

from pathlib import Path
from typing import Dict
import yaml

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from zerovox.g2p.model import ModelType, LightningTransformer
from zerovox.g2p.data import G2PDataModule

class ForgivingModelCheckpoint(ModelCheckpoint):

    def _save_topk_checkpoint(self, trainer: Trainer, monitor_candidates: Dict[str, torch.Tensor]) -> None:

        if self.monitor not in monitor_candidates:
            return

        return super(ForgivingModelCheckpoint, self)._save_topk_checkpoint(trainer, monitor_candidates)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Path to config.yaml")

    choices = ['cpu', 'gpu']
    parser.add_argument("--accelerator", type=str, default=choices[0], choices=choices)
    parser.add_argument("--lang", type=str, default="de", help="language")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--val_epochs", type=int, default=1)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate for AdamW.')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-5,
                        metavar='N',
                        help='Optimizer weight decay')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        help='Batch size')
    parser.add_argument('--out-dir',
                        default="mymodel1",
                        type=str,
                        help="Output directory for checkpoints, modelcfg and validation data",)
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="Path to model checkpoint file to continue training",)

    args = parser.parse_args()

    config = yaml.load( open(args.config, "r"), Loader=yaml.FullLoader)

    datamodule = G2PDataModule (config, args.lang, num_workers=args.num_workers, batch_size=args.batch_size)

    model_type = ModelType(config['model']['type'])

    if args.checkpoint:
        model = LightningTransformer.load_from_checkpoint(args.checkpoint,
                                                          model_type=model_type,
                                                          config=config,
                                                          symbols=datamodule.symbols,
                                                          val_dir=Path(args.out_dir) / 'validation',
                                                          lr=args.lr,
                                                          weight_decay=args.weight_decay,
                                                          max_epochs=args.max_epochs,
                                                          warmup_epochs=args.warmup_epochs)

    else:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        model = LightningTransformer(model_type=model_type,
                                     config=config,
                                     symbols=datamodule.symbols,
                                     val_dir=Path(args.out_dir) / 'validation',
                                     lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     max_epochs=args.max_epochs,
                                     warmup_epochs=args.warmup_epochs)

    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint_callback = ForgivingModelCheckpoint(
        monitor='val',
        dirpath=Path(args.out_dir) / 'checkpoints',
        #filename='epoch={epoch:02d}-loss={val:.5f}',
        filename='best',
        auto_insert_metric_name=False,
        #save_top_k=2,
        save_top_k=1,
        verbose=True,
        save_on_train_epoch_end=True,
    )

    trainer = Trainer(accelerator=args.accelerator,
                      check_val_every_n_epoch=args.val_epochs,
                      max_epochs=args.max_epochs,
                      callbacks=[checkpoint_callback],
                      default_root_dir=args.out_dir)
    #                   gradient_clip_val=args.grad_clip)

    trainer.fit(model, datamodule=datamodule)

