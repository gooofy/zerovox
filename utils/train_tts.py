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

from zerovox.tts.data import LJSpeechDataModule
from zerovox.tts.model import ZeroVox
from zerovox.g2p.data import G2PSymbols
from zerovox.lexicon import Lexicon
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

    parser.add_argument("configs",
                        type=str,
                        help="Path to config.yamls",
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

    parser.add_argument('--train-decoder-only',
                        action='store_true',
                        help='Train only the decoder part of the model (useful in incremental mode only, keeps all other module weights fixed)')

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

    preprocess_configs = []

    for fn in tqdm(cfgfns, desc="loading configs"):
        preprocess_configs.append(yaml.load(open(fn, "r"), Loader=yaml.FullLoader))

    cfg = yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader)

    modelcfg = {
        'lang'          : [],
        'audio': {
            'sampling_rate' : None,
            'hop_size'      : None,
            'num_mels'      : None,
            'filter_length' : None,
            'win_length'    : None,
            'mel_fmin'      : None,
            'mel_fmax'      : None,
            'fft_size'      : None,
            'eps'           : None,
            'window'        : None,
            'log_base'      : None
        },
        'model': {
            'emb_dim'       : cfg['model']['emb_dim'],
            'emb_reduction' : cfg['model']['emb_reduction'],
            'punct_emb_dim' : cfg['model']['punct_emb_dim'],
            'max_seq_len'   : cfg['model']['max_seq_len'],
            'encoder'       : {
                'fs2_layer'              : cfg['model']['encoder']['fs2_layer'],
                'fs2_head'               : cfg['model']['encoder']['fs2_head'],
                'fs2_dropout'            : cfg['model']['encoder']['fs2_dropout'],
                'vp_filter_size'         : cfg['model']['encoder']['vp_filter_size'],
                'vp_kernel_size'         : cfg['model']['encoder']['vp_kernel_size'],
                'vp_dropout'             : cfg['model']['encoder']['vp_dropout'],
                've_n_bins'              : cfg['model']['encoder']['ve_n_bins'],
            },
            'decoder'       : {
                'kind'                   : cfg['model']['decoder']['kind'],

                'n_layers'               : cfg['model']['decoder']['n_layers'],
                'n_head'                 : cfg['model']['decoder']['n_head'],
                'conv_filter_size'       : cfg['model']['decoder']['conv_filter_size'],
                'conv_kernel_size'       : cfg['model']['decoder']['conv_kernel_size'],
                'dropout'                : cfg['model']['decoder']['dropout'],
                'scln'                   : cfg['model']['decoder']['scln'],
            },
            'resnet' : {
                'layers'         : cfg['model']['resnet']['layers'],
                'num_filters'    : cfg['model']['resnet']['num_filters'],
                'encoder_type'   : cfg['model']['resnet']['encoder_type'],
            },
        },
        'stats': {
            'energy_min'    : sys.float_info.max,
            'energy_max'    : sys.float_info.min,
            'pitch_min'     : sys.float_info.max,
            'pitch_max'     : sys.float_info.min,
        },
    }

    for pc in tqdm(preprocess_configs, desc="check cfg consistency"):
        if pc['preprocessing']['text']['language'] not in modelcfg['lang']:
            modelcfg['lang'].append(pc['preprocessing']['text']['language'])

        if not modelcfg['audio']['sampling_rate']:
            modelcfg['audio']['sampling_rate'] = pc['preprocessing']['audio']['sampling_rate']
        else:
            if modelcfg['audio']['sampling_rate'] != pc['preprocessing']['audio']['sampling_rate']:
                raise Exception ('inconsistent rample rates detected')

        if not modelcfg['audio']['hop_size']:
            modelcfg['audio']['hop_size'] = pc['preprocessing']['mel']['hop_size']
        else:
            if modelcfg['audio']['hop_size'] != pc['preprocessing']['mel']['hop_size']:
                raise Exception ('inconsistent hop lengths detected')

        if not modelcfg['audio']['num_mels']:
            modelcfg['audio']['num_mels'] = pc['preprocessing']['mel']['num_mels']
        else:
            if modelcfg['audio']['num_mels'] != pc['preprocessing']['mel']['num_mels']:
                raise Exception ('inconsistent number of mel channels detected')

        if not modelcfg['audio']['filter_length']:
            modelcfg['audio']['filter_length'] = pc['preprocessing']['mel']['filter_length']
        else:
            if modelcfg['audio']['filter_length'] != pc['preprocessing']['mel']['filter_length']:
                raise Exception ('inconsistent filter length detected')

        if not modelcfg['audio']['win_length']:
            modelcfg['audio']['win_length'] = pc['preprocessing']['mel']['win_length']
        else:
            if modelcfg['audio']['win_length'] != pc['preprocessing']['mel']['win_length']:
                raise Exception ('inconsistent win length detected')

        if not modelcfg['audio']['mel_fmin']:
            modelcfg['audio']['mel_fmin'] = pc['preprocessing']['mel']['fmin']
        else:
            if modelcfg['audio']['mel_fmin'] != pc['preprocessing']['mel']['fmin']:
                raise Exception ('inconsistent mel fmin detected')

        if not modelcfg['audio']['mel_fmax']:
            modelcfg['audio']['mel_fmax'] = pc['preprocessing']['mel']['fmax']
        else:
            if modelcfg['audio']['mel_fmax'] != pc['preprocessing']['mel']['fmax']:
                raise Exception ('inconsistent mel fmax detected')

        if not modelcfg['audio']['fft_size']:
            modelcfg['audio']['fft_size'] = pc['preprocessing']['mel']['fft_size']
        else:
            if modelcfg['audio']['fft_size'] != pc['preprocessing']['mel']['fft_size']:
                raise Exception ('inconsistent fft size detected')

        if not modelcfg['audio']['eps']:
            modelcfg['audio']['eps'] = pc['preprocessing']['mel']['eps']
        else:
            if modelcfg['audio']['eps'] != pc['preprocessing']['mel']['eps']:
                raise Exception ('inconsistent eps detected')

        if not modelcfg['audio']['window']:
            modelcfg['audio']['window'] = pc['preprocessing']['mel']['window']
        else:
            if modelcfg['audio']['window'] != pc['preprocessing']['mel']['window']:
                raise Exception ('inconsistent window detected')

        if not modelcfg['audio']['log_base']:
            modelcfg['audio']['log_base'] = pc['preprocessing']['mel']['log_base']
        else:
            if modelcfg['audio']['log_base'] != pc['preprocessing']['mel']['log_base']:
                raise Exception ('inconsistent log_base detected')

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

    lexicon       = Lexicon.load(modelcfg['lang'][0], load_dicts=False)
    symbols       = G2PSymbols (lexicon.graphemes, lexicon.phonemes)

    os.makedirs (args.out_folder, exist_ok=True)

    if args.name:
        modelcfg_path = Path(args.out_folder) / f"modelcfg_{args.name}.yaml"
        wav_path = Path(args.out_folder) / 'validation' / args.name
        checkpoint_path = Path(args.out_folder) / 'checkpoints' / args.name
    else:
        modelcfg_path = Path(args.out_folder) / "modelcfg.yaml"
        wav_path = Path(args.out_folder) / 'validation'
        checkpoint_path = Path(args.out_folder) / 'checkpoints'

    with open (modelcfg_path, 'w') as modelcfgf:
        yaml.dump(modelcfg, modelcfgf, default_flow_style=False)

    args.num_workers *= args.devices

    datamodule = LJSpeechDataModule(preprocess_configs=preprocess_configs,
                                    symbols=symbols,
                                    stats=modelcfg['stats'],
                                    num_bins=cfg['model']['encoder']['ve_n_bins'],
                                    batch_size=cfg['training']['batch_size'],
                                    num_workers=args.num_workers)

    model = ZeroVox ( symbols=symbols,
                      # stats=(modelcfg['stats']['pitch_min'],modelcfg['stats']['pitch_max'],modelcfg['stats']['energy_min'],modelcfg['stats']['energy_max']),
                      meldec_model=args.meldec_model,
                      sampling_rate=modelcfg['audio']['sampling_rate'],
                      hop_length=modelcfg['audio']['hop_size'],
                      n_mels=modelcfg['audio']['num_mels'],
                      lr=cfg['training']['lr'],
                      weight_decay=cfg['training']['weight_decay'],
                      betas=cfg['training']['betas'],
                      eps=cfg['training']['eps'],
                      max_epochs=cfg['training']['max_epochs'],
                      warmup_epochs=cfg['training']['warmup_epochs'],

                      embed_dim=cfg['model']['emb_dim'],
                      punct_embed_dim=cfg['model']['punct_emb_dim'],
                      dpe_embed_dim=cfg['model']['dpe_emb_dim'],
                      emb_reduction=cfg['model']['emb_reduction'],
                      max_seq_len=cfg['model']['max_seq_len'],

                      fs2enc_layer=cfg['model']['encoder']['fs2_layer'],
                      fs2enc_head=cfg['model']['encoder']['fs2_head'],
                      fs2enc_dropout=cfg['model']['encoder']['fs2_dropout'],
                      vp_filter_size=cfg['model']['encoder']['vp_filter_size'],
                      vp_kernel_size=cfg['model']['encoder']['vp_kernel_size'],
                      vp_dropout=cfg['model']['encoder']['vp_dropout'],
                      ve_n_bins=cfg['model']['encoder']['ve_n_bins'],

                      resnet_layers=cfg['model']['resnet']['layers'],
                      resnet_num_filters=cfg['model']['resnet']['num_filters'],
                      resnet_encoder_type=cfg['model']['resnet']['encoder_type'],

                      decoder_kind=cfg['model']['decoder']['kind'],
                      decoder_n_layers=cfg['model']['decoder']['n_layers'],
                      decoder_n_head=cfg['model']['decoder']['n_head'],
                      decoder_conv_filter_size=cfg['model']['decoder']['conv_filter_size'],
                      decoder_conv_kernel_size=cfg['model']['decoder']['conv_kernel_size'],
                      decoder_dropout=cfg['model']['decoder']['dropout'],
                      decoder_scln=cfg['model']['decoder']['scln'],

                      wav_path=str(wav_path),
                      infer_device=args.infer_device,
                      verbose=args.verbose)

    # we restore the checkpoint manually to support partial training
    # https://github.com/Lightning-AI/pytorch-lightning/issues/2656

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        #torch.serialization.add_safe_globals([G2PSymbols])
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
        save_top_k=5,
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
                      check_val_every_n_epoch=cfg['training']['val_epochs'],
                      max_epochs=cfg['training']['max_epochs'],
                      default_root_dir=args.out_folder,
                      callbacks=[checkpoint_callback],
                      gradient_clip_val=cfg['training']['grad_clip'],
                      num_sanity_val_steps=0,
                      logger=logger,
                      log_every_n_steps=1)

    if args.compile:
        model = torch.compile(model)

    start_time = datetime.datetime.now()
    #FIXME: remove ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    #FIXME: remove trainer.fit(model, datamodule=datamodule, ckpt_path = ckpt_path)
    trainer.fit(model, datamodule=datamodule)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"Training time: {elapsed_time}")
