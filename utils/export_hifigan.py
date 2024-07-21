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

import argparse
import glob
import shutil
import torch
import yaml
import os
import sys
import numpy

from tqdm import tqdm
from scipy.io import wavfile

from zerovox.g2p.data import G2PSymbols
from zerovox.g2p.g2p  import G2PTokenizer
from zerovox.lexicon  import Lexicon
from zerovox.tts.data import LJSpeechDataModule
from zerovox.tts.model import ZeroVox, write_to_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs='+', help="path to preprocess.yamls")
    parser.add_argument("--out-dir", type=str, help="exported corpus output path")
    parser.add_argument("--model",
                        default=None,
                        required=True,
                        help="Path to model directory",)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hifigan-checkpoint",
                        default="VCTK_V2",
                        type=str,
                        help="HiFiGAN model",)
    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default=choices[0],
                        choices=choices,
                        type=str,
                        help="Inference device",)

    args = parser.parse_args()

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

    with open (os.path.join(args.model, "modelcfg.yaml")) as modelcfgf:
        modelcfg = yaml.load(modelcfgf, Loader=yaml.FullLoader)

    lexicon       = Lexicon.load(modelcfg['lang'])
    symbols       = G2PSymbols (lexicon.graphemes, lexicon.phonemes)

    list_of_files = glob.glob(os.path.join(args.model, 'checkpoints/*.ckpt'))
    checkpoint = max(list_of_files, key=os.path.getctime)

    model = ZeroVox.load_from_checkpoint(lang=modelcfg['lang'],
                                         hifigan_checkpoint=args.hifigan_checkpoint,
                                         sampling_rate=modelcfg['audio']['sampling_rate'],
                                         hop_length=modelcfg['audio']['hop_length'],
                                         checkpoint_path=checkpoint,
                                         infer_device=args.infer_device,
                                         map_location=args.infer_device)
#                                         map_location=torch.device('cpu'))

    model = model.to(args.infer_device)
    model.eval()

    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, mode=0o755)

    datamodule = LJSpeechDataModule(preprocess_configs=preprocess_configs,
                                    symbols=symbols,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    datamodule.prepare_data()
    dl = datamodule.train_dataloader()

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(dl)):
            x, y = batch
            # print (x)
            # print (y)

            # x, y = batch
            wavs, mels, lengths, _ = model.forward(x)
            wavs = wavs.to(torch.float).cpu().numpy()
            # write_to_file(wavs, modelcfg['audio']['sampling_rate'], hop_length=modelcfg['audio']['hop_length'], lengths=lengths.cpu().numpy(), \
            #     wav_path=args.out_dir, filename=f"prediction-{batch_idx}")

            wavs = (wavs * 32760).astype("int16")
            wavs = [wav for wav in wavs]
            lengths *= modelcfg['audio']['hop_length']

            for i in range(len(wavs)):
                shutil.copyfile(os.path.join(x['preprocessed_paths'][i], os.path.join('wavs', x['basenames'][i]+'.wav')), 
                                os.path.join(args.out_dir, f"{batch_idx}-{i}-orig.wav"))

                wavs[i] = wavs[i][: lengths[i]]
                    
                path = os.path.join(args.out_dir, f"{batch_idx}-{i}.wav")
                wavfile.write(path, modelcfg['audio']['sampling_rate'], wavs[i])

                path = os.path.join(args.out_dir, f"{batch_idx}-{i}.npy")
                numpy.save(path, mels[i])

                path = os.path.join(args.out_dir, f"{batch_idx}-{i}.txt")
                with open(path, "w") as f:
                    f.write(x['text'][i])
