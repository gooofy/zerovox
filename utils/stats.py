#!/bin/env python3



import sys
import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("-c", "--model-config",
    #                     type=str,
    #                     help="Path to model config.yaml",
    #                     required=True)

    parser.add_argument("configs",
                        type=str,
                        help="Path to config.yamls",
                        nargs='+')


    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print out debug information')

    args = parser.parse_args()

    return args



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

    # cfg = yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader)

    sampling_rate = None
    hop_length = None
    num_speakers = 0
    total_length = 0

    for pc in tqdm(preprocess_configs):

        num_speakers += 1

        if not sampling_rate:
            sampling_rate = pc['preprocessing']['audio']['sampling_rate']
        else:
            if sampling_rate != pc['preprocessing']['audio']['sampling_rate']:
                raise Exception ('inconsistent rample rates detected')

        if not hop_length:
            hop_length = pc['preprocessing']['mel']['hop_size']
        else:
            if hop_length != pc['preprocessing']['mel']['hop_size']:
                raise Exception ('inconsistent hop lengths detected')

        mel_dir = os.path.join(pc['path']['preprocessed_path'], 'mel')
        for melfn in os.listdir(mel_dir):

            #print (melfn)

            if melfn.endswith('.npy'):
                mel_spectrogram = np.load(os.path.join(mel_dir, melfn))
                #print (mel_spectrogram.shape)
                num_frames = float(mel_spectrogram.shape[0])
                audio_length = (num_frames * hop_length) / sampling_rate
                total_length += audio_length

                #print (f"mel: {mel_spectrogram.shape} -> audio_length={audio_length}")

            #break

    print (f"sampling rate: {sampling_rate}, hop_length: {hop_length}, # speakers: {num_speakers}, audio length: {total_length}s={total_length/3600.0}h")
