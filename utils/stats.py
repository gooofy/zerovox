#!/bin/env python3



import sys
import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("modelcfg", type=str, help="model config preprocessing was done for")
    parser.add_argument("corpora", type=str, nargs='+', help="path[s] to corpus .yaml config file[s] or directorie[s]")

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print out debug information')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    modelcfg = yaml.load(open(args.modelcfg, "r"), Loader=yaml.FullLoader)

    print (f"audio cfg:\n{modelcfg['audio']}")

    for corpusfn in args.corpora:

        corpus_configs = []

        if os.path.isdir(corpusfn):
            for cfn in os.listdir(corpusfn):

                _, ext = os.path.splitext(cfn)
                if ext != '.yaml':
                    continue

                cfpath = os.path.join(corpusfn, cfn)
                cfg = yaml.load(open(cfpath, "r"), Loader=yaml.FullLoader)
                corpus_configs.append(cfg)
        else:
            cfg = yaml.load(open(corpusfn, "r"), Loader=yaml.FullLoader)
            corpus_configs.append(cfg)

        lang = None
        for corpus in corpus_configs:
            if not lang:
                lang = corpus['language']
            else:
                if lang != corpus['language']:
                    raise Exception ('inconsistent languages detected')

        sampling_rate = modelcfg['audio']['sampling_rate']
        hop_length = modelcfg['audio']['hop_size']
        num_speakers = 0
        total_length = 0

        for pc in tqdm(corpus_configs):

            num_speakers += 1

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

        print (f"{corpusfn}: sampling rate: {sampling_rate}, hop_length: {hop_length}, # speakers: {num_speakers}, audio length: {total_length:.1f}s = {total_length/3600.0:.1f}h")
