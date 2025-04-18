#!/bin/env python3

'''
zerovox

    Apache 2.0 License
    2024 by Guenter Bartsch

is based on:

    EfficientSpeech: An On-Device Text to Speech Model
    https://ieeexplore.ieee.org/abstract/document/10094639
    Rowel Atienza
    Apache 2.0 License
'''

import sys
import os
import yaml
import torch
import datetime
import argparse
import json
import random

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from zerovox.tts.data import LJSpeechDataModule, PREPROCESSED_DATA_PATH
from zerovox.tts.model import ZeroVox
from zerovox.tts.symbols import Symbols
from zerovox.tts.model import DEFAULT_MELDEC_MODEL_NAME

from pathlib import Path

from typing import Dict

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()

    choices = ['cpu', 'gpu']
    parser.add_argument("--accelerator", type=str, default=choices[1], choices=choices)

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--threads", type=int, default=24)

    #choices = ["bf16-mixed", "16-mixed", 16, 32, 64]
    parser.add_argument("--precision", default="16-mixed")

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("-c", "--model-config",
                        type=str,
                        help="Path to model config.yaml",
                        required=True)

    parser.add_argument("corpora",
                        type=str,
                        help="Path to corpus .yamls",
                        nargs='+')

    parser.add_argument('--out-folder',
                        default="mymodel1",
                        type=str,
                        help="Output folder for checkpoints, modelcfg and validation data",)

    parser.add_argument("--meldec-model",
                        default=DEFAULT_MELDEC_MODEL_NAME,
                        type=str,
                        help="Multi-Band MELGAN model",)

    parser.add_argument("--name",
                        type=str,
                        help="run name (optional)")

    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="Path to model checkpoint file",)

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print out debug information')

    parser.add_argument('--train-decoder-only',
                        action='store_true',
                        help='Train only the decoder part of the model (useful in incremental mode only, keeps all other module weights fixed)')

    parser.add_argument('--max-epochs',
                        type=int,
                        default=40,
                        help='Train for number of epochs, default: 40')
    parser.add_argument('--warmup-epochs',
                        type=int,
                        default=2,
                        help='Number of warmup epochs, default: 2')
    parser.add_argument('--batch-size',
                        type=int,
                        default=24,
                        help='Training batch size, default: 24')


    args = parser.parse_args()

    args.num_workers *= args.devices

    return args

class ZVModelCheckpointCheckpoint(ModelCheckpoint):

    def _save_topk_checkpoint(self, trainer: Trainer, monitor_candidates: Dict[str, torch.Tensor]) -> None:

        if self.monitor not in monitor_candidates:
            return

        return super(ZVModelCheckpointCheckpoint, self)._save_topk_checkpoint(trainer, monitor_candidates)

if __name__ == "__main__":

    from setproctitle import setproctitle

    setproctitle("ZeroTTS_train")
    random.seed(42)

    args = get_args()

    print ("collecting .yaml files from specified paths...")

    corpora = []
    for cfgfn in tqdm(args.corpora):
        if os.path.isdir(cfgfn):
            for cfn in os.listdir(cfgfn):

                _, ext = os.path.splitext(cfn)
                if ext != '.yaml':
                    continue

                cfgpath = os.path.join(cfgfn, cfn)
                corpora.append(yaml.load(open(cfgpath, "r"), Loader=yaml.FullLoader))
        else:
            corpora.append(yaml.load(open(cfgfn, "r"), Loader=yaml.FullLoader))

    if not corpora:
        raise Exception ("*** error: no .yaml files found!")
    print (f"{len(corpora)} corpus .yaml files found.")

    modelcfg = yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader)
    
    modelcfg['stats'] = {
        'energy_min' : sys.float_info.max,
        'energy_max' : sys.float_info.min,
        'pitch_min'  : sys.float_info.max,
        'pitch_max'  : sys.float_info.min
    }
    modelcfg['lang']                 = []

    for corpus in tqdm(corpora, desc="compute corpus stats"):
        if corpus['language'] not in modelcfg['lang']:
            modelcfg['lang'].append(corpus['language'])

        with open(os.path.join(PREPROCESSED_DATA_PATH, corpus['path']['preprocessed_path'], 'stats.json')) as statsf:
            stats = json.load(statsf)
            pitch_min, pitch_max   = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

            if energy_min < modelcfg['stats']['energy_min']:
                modelcfg['stats']['energy_min'] = energy_min
            if energy_max > modelcfg['stats']['energy_max']:
                modelcfg['stats']['energy_max'] = energy_max

            if pitch_min < modelcfg['stats']['pitch_min']:
                modelcfg['stats']['pitch_min'] = pitch_min
            if pitch_max > modelcfg['stats']['pitch_max']:
                modelcfg['stats']['pitch_max'] = pitch_max

    symbols       = Symbols (modelcfg['model']['phones'], modelcfg['model']['puncts'])

    os.makedirs (args.out_folder, exist_ok=True)

    if args.name:
        modelcfg_path = Path(args.out_folder) / f"modelcfg_{args.name}.yaml"
        checkpoint_path = Path(args.out_folder) / 'checkpoints' / args.name
    else:
        modelcfg_path = Path(args.out_folder) / "modelcfg.yaml"
        checkpoint_path = Path(args.out_folder) / 'checkpoints'

    with open (modelcfg_path, 'w') as modelcfgf:
        yaml.dump(modelcfg, modelcfgf, default_flow_style=False)

    args.num_workers *= args.devices

    datamodule = LJSpeechDataModule(corpora=corpora,
                                    symbols=symbols,
                                    stats=modelcfg['stats'],
                                    num_bins=modelcfg['model']['encoder']['ve_n_bins'],
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    model = ZeroVox ( symbols=symbols,
                      meldec_model=args.meldec_model,
                      sampling_rate=modelcfg['audio']['sampling_rate'],
                      hop_length=modelcfg['audio']['hop_size'],
                      n_mels=modelcfg['audio']['num_mels'],
                      lr=modelcfg['training']['learning_rate'],
                      weight_decay=modelcfg['training']['weight_decay'],
                      betas=modelcfg['training']['betas'],
                      eps=modelcfg['training']['eps'],
                      max_epochs=args.max_epochs,
                      warmup_epochs=args.warmup_epochs,

                      embed_dim=modelcfg['model']['emb_dim'],
                      punct_embed_dim=modelcfg['model']['punct_emb_dim'],
                      dpe_embed_dim=modelcfg['model']['dpe_emb_dim'],
                      emb_reduction=modelcfg['model']['emb_reduction'],
                      max_txt_len=modelcfg['model']['max_txt_len'],
                      max_mel_len=modelcfg['model']['max_mel_len'],

                      fs2enc_layer=modelcfg['model']['encoder']['fs2_layer'],
                      fs2enc_head=modelcfg['model']['encoder']['fs2_head'],
                      fs2enc_dropout=modelcfg['model']['encoder']['fs2_dropout'],
                      vp_filter_size=modelcfg['model']['encoder']['vp_filter_size'],
                      vp_kernel_size=modelcfg['model']['encoder']['vp_kernel_size'],
                      vp_dropout=modelcfg['model']['encoder']['vp_dropout'],
                      ve_n_bins=modelcfg['model']['encoder']['ve_n_bins'],

                      resnet_layers=modelcfg['model']['resnet']['layers'],
                      resnet_num_filters=modelcfg['model']['resnet']['num_filters'],
                      resnet_encoder_type=modelcfg['model']['resnet']['encoder_type'],

                      decoder_kind=modelcfg['model']['decoder']['kind'],
                      decoder_n_layers=modelcfg['model']['decoder']['n_layers'],
                      decoder_n_head=modelcfg['model']['decoder']['n_head'],
                      decoder_conv_filter_size=modelcfg['model']['decoder']['conv_filter_size'],
                      decoder_conv_kernel_size=modelcfg['model']['decoder']['conv_kernel_size'],
                      decoder_dropout=modelcfg['model']['decoder']['dropout'],
                      decoder_scln=modelcfg['model']['decoder']['scln'],

                      verbose=args.verbose)

    # we restore the checkpoint manually to support partial training
    # https://github.com/Lightning-AI/pytorch-lightning/issues/2656

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        #torch.serialization.add_safe_globals([G2PSymbols])

        print(f"incremental training mode: restoring model weights from {args.checkpoint}")

        checkpoint = torch.load(ckpt_path, weights_only=False)

        state_dict = checkpoint['state_dict']

        if args.train_decoder_only:

            keys = list(state_dict.keys())

            for key in keys:
                if key.startswith('_mel_decoder'):
                    print (f"decoder only training mode: removing key {key} from state dict")
                    del state_dict[key]

            random_state_dict = model.state_dict()
            for key in random_state_dict.keys():
                if key.startswith('_mel_decoder'):
                    print (f"decoder only training mode: adding weigths for {key} from default state")
                    state_dict[key] = random_state_dict[key]

            for name, p in model.named_parameters():
                if not name.startswith('_mel_decoder'):
                    print (f"decoder only training mode: freezing weights for {name}")
                    p.requires_grad = False
            model._phoneme_encoder.eval()
            model._spkemb.eval()
            
        model.load_state_dict(state_dict)

    checkpoint_callback = ZVModelCheckpointCheckpoint(
        monitor='loss',
        dirpath=str(checkpoint_path),
        # filename='epoch={epoch:02d}-loss={loss:.2f}',
        # filename='best',
        filename='{epoch:04d}',
        auto_insert_metric_name=False,
        save_top_k=args.max_epochs, # keep all checkpoints
        # save_top_k=1,
        verbose=True,
        save_on_train_epoch_end=True,
    )

    if args.name:
        logger=TensorBoardLogger(Path(args.out_folder) / "lightning_logs", name=args.name)
    else:
        logger=TensorBoardLogger(Path(args.out_folder))

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      #check_val_every_n_epoch=cfg['training']['val_epochs'],
                      max_epochs=args.max_epochs,
                      default_root_dir=args.out_folder,
                      callbacks=[checkpoint_callback],
                      gradient_clip_val=modelcfg['training']['grad_clip'],
                      #num_sanity_val_steps=0,
                      logger=logger,
                      log_every_n_steps=1)

    trainer.fit(model, datamodule=datamodule)
