#!/bin/env python3

'''
zerovox

    Apache 2.0 License
    2024, 2025 by Guenter Bartsch

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
import h5py
import numpy

from tqdm import tqdm
from scipy.io import wavfile

from zerovox.tts.data import LJSpeechDataModule
from zerovox.tts.model import ZeroVox, DEFAULT_MELDEC_MODEL_NAME
from zerovox.tts.synthesize import ZeroVoxTTS
from zerovox.tts.symbols import Symbols

#DEBUG_LIMIT=2
DEBUG_LIMIT=0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("corpora",
                        type=str,
                        help="Path to corpus .yamls",
                        nargs='+')
    parser.add_argument("--out-dir", type=str, help="exported corpus output path")
    parser.add_argument("--model",
                        default=ZeroVoxTTS.get_default_model(),
                        help=f"TTS model to use: Path to model directory or model name, default: {ZeroVoxTTS.get_default_model()}")
    parser.add_argument("--meldec-model",
                        default=DEFAULT_MELDEC_MODEL_NAME,
                        type=str,
                        help=f"MELGAN model to use (meldec-libritts-multi-band-melgan-v2 or meldec-libritts-hifigan-v1, default: {DEFAULT_MELDEC_MODEL_NAME})",)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default=choices[0],
                        choices=choices,
                        type=str,
                        help="Inference device",)

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

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

    with open (os.path.join(args.model, "modelcfg.yaml")) as modelcfgf:
        modelcfg = yaml.load(modelcfgf, Loader=yaml.FullLoader)

    list_of_files = glob.glob(os.path.join(args.model, 'checkpoints/*.ckpt'))
    checkpoint = max(list_of_files, key=os.path.getctime)

    model = ZeroVox.load_from_checkpoint(lang=modelcfg['lang'],
                                         meldec_model=args.meldec_model,
                                         sampling_rate=modelcfg['audio']['sampling_rate'],
                                         hop_length=modelcfg['audio']['hop_size'],
                                         checkpoint_path=str(checkpoint),
                                         infer_device=args.infer_device,                                                              
                                         map_location=args.infer_device,
                                         strict=False,
                                         verbose=args.verbose,
                                         betas=[0.9,0.99], # FIXME : remove
                                         eps=1e-9) # FIXME: remove



    model = model.to(args.infer_device)
    model.eval()

    symbols       = Symbols (modelcfg['model']['phones'], modelcfg['model']['puncts'])

    # shutil.rmtree(args.out_dir, ignore_errors=True)

    os.makedirs(os.path.join(args.out_dir, "train"), mode=0o755)
    os.makedirs(os.path.join(args.out_dir, "dev"), mode=0o755)

    datamodule = LJSpeechDataModule(corpora=corpora,
                                    symbols=symbols,
                                    stats=modelcfg['stats'],
                                    num_bins=modelcfg['model']['encoder']['ve_n_bins'],
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)


    datamodule.prepare_data()
    dl = datamodule.train_dataloader()

    cnt = 0
    sr = modelcfg['audio']['sampling_rate']
    hop_length=modelcfg['audio']['hop_size']

    with torch.no_grad():

        debug_cnt = 0

        for batch_idx, batch in enumerate(tqdm(dl)):

            x, y = batch
            # print (x)
            # print (y)

            for k in x.keys():
                if torch.is_tensor(x[k]):
                    x[k] = x[k].to(args.infer_device)

            # x, y = batch
            wavs, mels, lengths, _ = model.forward(x, force_duration=True)
            wavs = wavs.to(torch.float).cpu().numpy()

            wavs = (wavs * 32760).astype("int16")
            wavs = [wav for wav in wavs]

            for i in range(len(wavs)):

                cnt += 1
                if cnt % 100 == 0:
                    out_dir = os.path.join(args.out_dir, "dev", os.path.basename(x['preprocessed_paths'][0]))
                else:
                    out_dir = os.path.join(args.out_dir, "train", os.path.basename(x['preprocessed_paths'][0]))

                os.makedirs(os.path.join(out_dir), mode=0o755, exist_ok=True)

                wav_len = int(lengths[i])
                dur_sum = int(numpy.sum(x['duration'][i].cpu().numpy()))

                assert wav_len == dur_sum

                orig_wav_path = os.path.join(x['preprocessed_paths'][i], os.path.join('wavs', x['basenames'][i]+'.wav'))
                _, orig_wav = wavfile.read(orig_wav_path, 'r')
                #offset = x['starts'][i] * hop_length
                orig_wav = orig_wav[x['starts'][i] * hop_length : (x['ends'][i]+1) * hop_length]

                padding_needed = wav_len*hop_length - len(orig_wav)
                if padding_needed > 0:
                    print (f"warning: padding of {padding_needed} samples needed for {orig_wav_path}")
                    orig_wav = numpy.pad(orig_wav, (0, padding_needed))

                #path = os.path.join(out_dir, f"{batch_idx}-{i}.wav")
                path = os.path.join(out_dir, f"{x['basenames'][i]}.wav")
                wavfile.write(path, sr, orig_wav[:wav_len*hop_length])

                synth_wav = wavs[i][: wav_len*hop_length]
                    
                path = os.path.join(out_dir, f"{x['basenames'][i]}-synth.wav")
                wavfile.write(path, sr, synth_wav)

                path = os.path.join(out_dir, f"{x['basenames'][i]}.h5")
                mel = mels[i].cpu().numpy()
                mel = mel[:, :wav_len]
                with h5py.File(path, 'w') as hdf:
                    hdf.create_dataset('feats', data=mel.T)
                    hdf.create_dataset('wave', data=orig_wav)
                #numpy.save(path, mel)

                path = os.path.join(out_dir, f"{x['basenames'][i]}.txt")
                with open(path, "w") as f:
                    f.write(x['text'][i])

                # line = f"{batch_idx}-{i}|{x['text'][i]}\n"
                # if cnt % 100 == 0:
                #     valf.write(line)
                # else:
                #     trainf.write(line)

            debug_cnt += 1
            if DEBUG_LIMIT and debug_cnt >= DEBUG_LIMIT:
                print (f"*** debug limit ({DEBUG_LIMIT} batches) reached ***")
                break
