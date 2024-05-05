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
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--val_epochs", type=int, default=10)
 
    parser.add_argument("configs",
                        type=str,
                        help="Path to config.yamls",
                        nargs='+')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-5,
                        metavar='N',
                        help='Optimizer weight decay')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        metavar='N',
                        help='Learning rate for AdamW.')
    parser.add_argument('--grad-clip',
                        type=float,
                        default=1.0,
                        metavar='N',
                        help='Gradient clipping value, default: 1.0')

    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='Batch size')

    parser.add_argument('--depth',
                        type=int,
                        default=2,
                        help='Encoder depth. Default for tiny, small & base.')
    parser.add_argument('--block-depth',
                        type=int,
                        default=2,
                        help='Decoder block depth. Default for tiny & small. Base:  3')
    parser.add_argument('--n-blocks',
                        type=int,
                        default=2,
                        help='Decoder blocks. Default for tiny. Small & base: 3.')
    parser.add_argument('--reduction',
                        type=int,
                        default=4,
                        help='Embed dim reduction factor. Default for tiny. Small: 2. Base: 1.')
    parser.add_argument('--head',
                        type=int,
                        default=1,
                        help='Number of transformer encoder head. Default for tiny & small. Base: 2.')
    parser.add_argument('--embed-dim',
                        type=int,
                        default=128,
                        help='Embedding or feature dim. To be reduced by --reduction.')
    parser.add_argument('--punct-embed-dim',
                        type=int,
                        default=16,
                        help='Punctuation embedding dim. Default: 16')
    parser.add_argument('--speaker-embed-dim',
                        type=int,
                        default=192,
                        help='Speaker embedding dim. Default: 192')
    parser.add_argument('--kernel-size',
                        type=int,
                        default=3,
                        help='Conv1d kernel size (Encoder). Default for tiny & small. Base is 5.')
    parser.add_argument('--decoder-kernel-size',
                        type=int,
                        default=5,
                        help='Conv1d kernel size (Decoder). Default for tiny, small & base: 5.')
    parser.add_argument('--expansion',
                        type=int,
                        default=1,
                        help='MixFFN expansion. Default for tiny & small. Base: 2.')
    parser.add_argument('--out-folder',
                        default="mymodel1",
                        type=str,
                        help="Output folder for checkpoints, modelcfg and validation data",)


    parser.add_argument("--hifigan-checkpoint",
                        default="VCTK_V2",
                        type=str,
                        help="HiFiGAN checkpoint",)

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

    parser.add_argument('--jit',
                        type=str,
                        default=None,
                        help='Convert to jit model')
    # use jit modules 
    parser.add_argument('--to-torchscript',
                        action='store_true',
                        help='Convert model to torchscript')
 
    parser.add_argument('--compile',
                        action='store_true',
                        help='Train using the compiled model')


    args = parser.parse_args()

    args.num_workers *= args.devices

    return args


def print_args(args):
    opt_log =  '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log

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

    modelcfg = {
        'lang'          : None,
        'sampling_rate' : None,
        'hop_length'    : None,
        'energy_min'    : sys.float_info.max,
        'energy_max'    : sys.float_info.min,
        'pitch_min'     : sys.float_info.max,
        'pitch_max'     : sys.float_info.min,

        'depth'               : args.depth,
        'n_blocks'            : args.n_blocks,
        'block_depth'         : args.block_depth,
        'reduction'           : args.reduction,
        'head'                : args.head,
        'embed_dim'           : args.embed_dim,
        'punct_embed_dim'     : args.punct_embed_dim,
        'speaker_embed_dim'   : args.speaker_embed_dim,
        'kernel_size'         : args.kernel_size,
        'decoder_kernel_size' : args.decoder_kernel_size,
        'expansion'           : args.expansion,

    }

    for pc in preprocess_configs:
        if not modelcfg['lang']:
            modelcfg['lang'] = pc['preprocessing']['text']['language']
        else:
            if modelcfg['lang'] != pc['preprocessing']['text']['language']:
                raise Exception ('Multiple languages detected')

        if not modelcfg['sampling_rate']:
            modelcfg['sampling_rate'] = pc['preprocessing']['audio']['sampling_rate']
        else:
            if modelcfg['sampling_rate'] != pc['preprocessing']['audio']['sampling_rate']:
                raise Exception ('inconsistent rample rates detected')

        if not modelcfg['hop_length']:
            modelcfg['hop_length'] = pc['preprocessing']['stft']['hop_length']
        else:
            if modelcfg['hop_length'] != pc['preprocessing']['stft']['hop_length']:
                raise Exception ('inconsistent hop lengths detected')

        with open(os.path.join(pc['path']['preprocessed_path'], 'stats.json')) as statsf:
            stats = json.load(statsf)
            pitch_min, pitch_max   = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

            if energy_min < modelcfg['energy_min']:
                modelcfg['energy_min'] = energy_min
            if energy_max > modelcfg['energy_max']:
                modelcfg['energy_max'] = energy_max

            if pitch_min < modelcfg['pitch_min']:
                modelcfg['pitch_min'] = pitch_min
            if pitch_max > modelcfg['pitch_max']:
                modelcfg['pitch_max'] = pitch_max

    lexicon       = Lexicon.load(modelcfg['lang'])
    symbols       = G2PSymbols (lexicon.graphemes, lexicon.phonemes)

    os.makedirs (args.out_folder, exist_ok=True)

    with open (os.path.join(args.out_folder, 'modelcfg.json'), 'w') as modelcfgf:
        modelcfgf.write(json.dumps(modelcfg, indent=4))

    args.num_workers *= args.devices

    datamodule = LJSpeechDataModule(preprocess_configs=preprocess_configs,
                                    symbols=symbols,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    model = ZeroVox ( symbols=symbols,
                            stats=(modelcfg['pitch_min'],modelcfg['pitch_max'],modelcfg['energy_min'],modelcfg['energy_max']),
                            sampling_rate=modelcfg['sampling_rate'],
                            hop_length=modelcfg['hop_length'],
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            max_epochs=args.max_epochs,
                            warmup_epochs=args.warmup_epochs,
                            depth=args.depth,
                            n_blocks=args.n_blocks,
                            block_depth=args.block_depth,
                            reduction=args.reduction,
                            head=args.head,
                            embed_dim=args.embed_dim,
                            punct_embed_dim=args.punct_embed_dim,
                            speaker_embed_dim=args.speaker_embed_dim,
                            kernel_size=args.kernel_size,
                            decoder_kernel_size=args.decoder_kernel_size,
                            expansion=args.expansion,
                            wav_path=os.path.join(args.out_folder, 'validation'),
                            hifigan_checkpoint=args.hifigan_checkpoint,
                            infer_device=args.infer_device,
                            verbose=args.verbose)

    if args.verbose:
        print_args(args)

    checkpoint_callback = ZVModelCheckpointCheckpoint(
        monitor='loss',
        dirpath=os.path.join(args.out_folder, 'checkpoints'),
        # filename='epoch={epoch:02d}-loss={loss:.2f}',
        filename='best',
        auto_insert_metric_name=False,
        # save_top_k=2,
        save_top_k=1,
        verbose=True,
        save_on_train_epoch_end=True,
    )

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      check_val_every_n_epoch=args.val_epochs,
                      max_epochs=args.max_epochs,
                      default_root_dir=args.out_folder,
                      callbacks=[checkpoint_callback],
                      gradient_clip_val=args.grad_clip)

    if args.compile:
        model = torch.compile(model)

    start_time = datetime.datetime.now()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    trainer.fit(model, datamodule=datamodule, ckpt_path = ckpt_path)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"Training time: {elapsed_time}")
