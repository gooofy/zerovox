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

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from zerovox.tts.data import LJSpeechDataModule
from zerovox.tts.model import ZeroVox
from zerovox.g2p.data import G2PSymbols
from zerovox.lexicon import Lexicon
from zerovox.tts.model import DEFAULT_HIFIGAN_MODEL_NAME

from pathlib import Path

from typing import Dict

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

    parser.add_argument("configs",
                        type=str,
                        help="Path to config.yamls",
                        nargs='+')

    parser.add_argument('--out-folder',
                        default="mymodel1",
                        type=str,
                        help="Output folder for checkpoints, modelcfg and validation data",)

    parser.add_argument("--hifigan-model",
                        default=DEFAULT_HIFIGAN_MODEL_NAME,
                        type=str,
                        help="HiFiGAN model",)

    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default=choices[1],
                        choices=choices,
                        type=str,
                        help="Inference device",)

    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="Path to model checkpoint file",)

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print out debug information')

    parser.add_argument('--compile',
                        action='store_true',
                        help='Train using the compiled model')


    args = parser.parse_args()

    args.num_workers *= args.devices

    return args

class ZVModelCheckpointCheckpoint(ModelCheckpoint):

    def _save_topk_checkpoint(self, trainer: Trainer, monitor_candidates: Dict[str, torch.Tensor]) -> None:

        if self.monitor not in monitor_candidates:
            return

        return super(ZVModelCheckpointCheckpoint, self)._save_topk_checkpoint(trainer, monitor_candidates)

if __name__ == "__main__":
    args = get_args()

    print ("collecting .yaml files from specified paths...")

    cfgfns = []
    for cfgfn in args.configs:
        if os.path.isdir(cfgfn):
            for cfn in os.listdir(cfgfn):

                _, ext = os.path.splitext(cfn)
                if ext != '.yaml':
                    continue

                cfpath = os.path.join(cfgfn, cfn)
                #print (f"{cfpath} ...")
                cfgfns.append(cfpath)
        else:
            #print (f"{cfgfn} ...")
            cfgfns.append(cfgfn)

    if not cfgfns:
        print ("*** error: no .yaml files found!")
        sys.exit(1)
    else:
        print (f"{len(cfgfns)} .yaml files found.")

    preprocess_configs = [yaml.load(open(fn, "r"), Loader=yaml.FullLoader) for fn in cfgfns]

    cfg = yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader)

    modelcfg = {
        'lang'          : None,
        'audio': {
            'sampling_rate' : None,
            'hop_length'    : None,
            'n_mel_channels': None,
            'filter_length' : None,
            'win_length'    : None,
            'mel_fmin'      : None,
            'mel_fmax'      : None,
        },
        'model': {
            'emb_dim'       : cfg['model']['emb_dim'],
            'emb_reduction' : cfg['model']['emb_reduction'],
            'punct_emb_dim' : cfg['model']['punct_emb_dim'],
            'encoder'       : {
                'depth'       : cfg['model']['encoder']['depth'],
                'n_heads'     : cfg['model']['encoder']['n_heads'],
                'kernel_size' : cfg['model']['encoder']['kernel_size'],
                'expansion'   : cfg['model']['encoder']['expansion'],
            },
            'decoder'       : {
                'block_depth' : cfg['model']['decoder']['block_depth'],
                'n_blocks'    : cfg['model']['decoder']['n_blocks'],
                'kernel_size' : cfg['model']['decoder']['kernel_size'],
                'x2_fix'      : True,
            },
            'gst'           : {
                'n_style_tokens' : cfg['model']['gst']['n_style_tokens'],
                'n_heads'        : cfg['model']['gst']['n_heads'],
                'ref_enc_filters': cfg['model']['gst']['ref_enc_filters'],
            },
        },
        'stats': {
            'energy_min'    : sys.float_info.max,
            'energy_max'    : sys.float_info.min,
            'pitch_min'     : sys.float_info.max,
            'pitch_max'     : sys.float_info.min,
        },
    }

    for pc in preprocess_configs:
        if not modelcfg['lang']:
            modelcfg['lang'] = pc['preprocessing']['text']['language']
        else:
            if modelcfg['lang'] != pc['preprocessing']['text']['language']:
                raise Exception ('Multiple languages detected')

        if not modelcfg['audio']['sampling_rate']:
            modelcfg['audio']['sampling_rate'] = pc['preprocessing']['audio']['sampling_rate']
        else:
            if modelcfg['audio']['sampling_rate'] != pc['preprocessing']['audio']['sampling_rate']:
                raise Exception ('inconsistent rample rates detected')

        if not modelcfg['audio']['hop_length']:
            modelcfg['audio']['hop_length'] = pc['preprocessing']['stft']['hop_length']
        else:
            if modelcfg['audio']['hop_length'] != pc['preprocessing']['stft']['hop_length']:
                raise Exception ('inconsistent hop lengths detected')

        if not modelcfg['audio']['n_mel_channels']:
            modelcfg['audio']['n_mel_channels'] = pc['preprocessing']['mel']['n_mel_channels']
        else:
            if modelcfg['audio']['n_mel_channels'] != pc['preprocessing']['mel']['n_mel_channels']:
                raise Exception ('inconsistent number of mel channels detected')

        if not modelcfg['audio']['filter_length']:
            modelcfg['audio']['filter_length'] = pc['preprocessing']['stft']['filter_length']
        else:
            if modelcfg['audio']['filter_length'] != pc['preprocessing']['stft']['filter_length']:
                raise Exception ('inconsistent filter length detected')

        if not modelcfg['audio']['win_length']:
            modelcfg['audio']['win_length'] = pc['preprocessing']['stft']['win_length']
        else:
            if modelcfg['audio']['win_length'] != pc['preprocessing']['stft']['win_length']:
                raise Exception ('inconsistent win length detected')

        if not modelcfg['audio']['mel_fmin']:
            modelcfg['audio']['mel_fmin'] = pc['preprocessing']['mel']['mel_fmin']
        else:
            if modelcfg['audio']['mel_fmin'] != pc['preprocessing']['mel']['mel_fmin']:
                raise Exception ('inconsistent mel fmin detected')

        if not modelcfg['audio']['mel_fmax']:
            modelcfg['audio']['mel_fmax'] = pc['preprocessing']['mel']['mel_fmax']
        else:
            if modelcfg['audio']['mel_fmax'] != pc['preprocessing']['mel']['mel_fmax']:
                raise Exception ('inconsistent mel fmax detected')

        with open(os.path.join(pc['path']['preprocessed_path'], 'stats.json')) as statsf:
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

    lexicon       = Lexicon.load(modelcfg['lang'], load_dicts=False)
    symbols       = G2PSymbols (lexicon.graphemes, lexicon.phonemes)

    os.makedirs (args.out_folder, exist_ok=True)

    with open (os.path.join(args.out_folder, 'modelcfg.yaml'), 'w') as modelcfgf:
        yaml.dump(modelcfg, modelcfgf, default_flow_style=False)

    args.num_workers *= args.devices

    datamodule = LJSpeechDataModule(preprocess_configs=preprocess_configs,
                                    symbols=symbols,
                                    batch_size=cfg['training']['batch_size'],
                                    num_workers=args.num_workers)

    model = ZeroVox ( symbols=symbols,
                      stats=(modelcfg['stats']['pitch_min'],modelcfg['stats']['pitch_max'],modelcfg['stats']['energy_min'],modelcfg['stats']['energy_max']),
                      hifigan_model=args.hifigan_model,
                      sampling_rate=modelcfg['audio']['sampling_rate'],
                      hop_length=modelcfg['audio']['hop_length'],
                      n_mels=modelcfg['audio']['n_mel_channels'],
                      lr=cfg['training']['lr'],
                      weight_decay=cfg['training']['weight_decay'],
                      max_epochs=cfg['training']['max_epochs'],
                      warmup_epochs=cfg['training']['warmup_epochs'],
                      encoder_depth=cfg['model']['encoder']['depth'],
                      decoder_n_blocks=cfg['model']['decoder']['n_blocks'],
                      decoder_block_depth=cfg['model']['decoder']['block_depth'],
                      decoder_x2_fix=True,
                      reduction=cfg['model']['emb_reduction'],
                      encoder_n_heads=cfg['model']['encoder']['n_heads'],
                      embed_dim=cfg['model']['emb_dim'],
                      encoder_kernel_size=cfg['model']['encoder']['kernel_size'],
                      decoder_kernel_size=cfg['model']['decoder']['kernel_size'],
                      encoder_expansion=cfg['model']['encoder']['expansion'],
                      wav_path=os.path.join(args.out_folder, 'validation'),
                      infer_device=args.infer_device,
                      verbose=args.verbose,
                      punct_embed_dim=cfg['model']['punct_emb_dim'],
                      gst_n_style_tokens=cfg['model']['gst']['n_style_tokens'],
                      gst_n_heads=cfg['model']['gst']['n_heads'],
                      gst_ref_enc_filters=cfg['model']['gst']['ref_enc_filters'])

    checkpoint_callback = ZVModelCheckpointCheckpoint(
        monitor='loss',
        dirpath=os.path.join(args.out_folder, 'checkpoints'),
        # filename='epoch={epoch:02d}-loss={loss:.2f}',
        # filename='best',
        filename='{epoch:04d}',
        auto_insert_metric_name=False,
        save_top_k=5,
        # save_top_k=1,
        verbose=True,
        save_on_train_epoch_end=True,
    )

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      check_val_every_n_epoch=cfg['training']['val_epochs'],
                      max_epochs=cfg['training']['max_epochs'],
                      default_root_dir=args.out_folder,
                      callbacks=[checkpoint_callback],
                      gradient_clip_val=cfg['training']['grad_clip'],
                      num_sanity_val_steps=0)

    if args.compile:
        model = torch.compile(model)

    start_time = datetime.datetime.now()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    trainer.fit(model, datamodule=datamodule, ckpt_path = ckpt_path)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"Training time: {elapsed_time}")
