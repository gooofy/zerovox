#!/bin/env python3

import argparse
import os
import sys
import atexit
import random
import readline
import multiprocessing
import sounddevice

from zerovox.g2p.g2p import G2P
from zerovox.tts.synthesize import ZeroVoxTTS
from zerovox.g2p.g2p import DEFAULT_G2P_MODEL_NAME
from zerovox.lexicon.lexedit import LexEdit
from zerovox.tts.model import DEFAULT_HIFIGAN_MODEL_NAME

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='lextool', description='lexicon analyzer and editor')

    parser.add_argument('-l', '--lang', type=str, default='de', help="language, default: de")

    parser.add_argument('-O', '--oovs', type=str, help="OOV file to work on (reads from file, will skip existing entries)")

    parser.add_argument('-e', '--edit', type=str, help="entries to work on (comma separated list of entries to add or review)")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count())
    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default=choices[0],
                        choices=choices,
                        type=str,
                        help="Inference device",)
    parser.add_argument('-c', '--compile',
                        action='store_true',
                        help='Infer using the compiled model')    
    parser.add_argument('-m', "--model",
                        default=None,
                        help="Path to TTS model dir",)
    parser.add_argument("--hifigan-model",
                        default=DEFAULT_HIFIGAN_MODEL_NAME,
                        type=str,
                        help="HiFiGAN model",)
    parser.add_argument('--refaudio', type=str, help="reference audio wav file for synthesis")
    parser.add_argument("--g2p-model",
                        default=DEFAULT_G2P_MODEL_NAME,
                        type=str,
                        help=f"G2P model, default={DEFAULT_G2P_MODEL_NAME}",)                     


    args = parser.parse_args()

    g2p = G2P(args.lang, model=args.g2p_model)
    lex = g2p.lex

    words_to_edit = set()
    oovs_lex = {}

    if args.oovs:
        for line in open (args.oovs):
            sline = line.strip().lower()
            parts = sline.split('\t')
            word = parts[0]
            if not word:
                continue
            if word in lex:
                continue
            words_to_edit.add(word)
            oovs_lex[word] = parts[1].split(' ')

    if args.edit:
        for word in args.edit.split(','):
            words_to_edit.add(word.lower())
            if word in lex:
                oovs_lex[word] = lex[word]

    words_to_edit = list(words_to_edit)
    random.shuffle(words_to_edit)

    print (f"{len(words_to_edit)} entries found to work on")

    if args.model:
        modelcfg, synth = ZeroVoxTTS.load_model(args.model, 
                                                hifigan_model=args.hifigan_model,
                                                g2p=g2p,
                                                infer_device=args.infer_device,
                                                num_threads=args.threads,
                                                do_compile=args.compile,)
        
        if not args.refaudio:
            print ("*** ERROR: TTS model but no reference audio given.")
            sys.exit(1)

        print ("computing speaker embedding...")

        spkemb = synth.speaker_embed(args.refaudio)

    else:
        modelcfg = None
        synth = None
        spkemb = None

    histfile = os.path.join(os.path.expanduser("~"), ".lextool_history")
    try:
        readline.read_history_file(histfile)
        # default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)

    if args.model:
        sounddevice.default.reset()
        sounddevice.default.samplerate = modelcfg['audio']['sampling_rate']
        sounddevice.default.channels = 1
        sounddevice.default.dtype = 'int16'
        #sounddevice.default.device = None
        #sounddevice.default.latency = 'low'

    editor = LexEdit (g2p, synth, spkemb)

    editor.edit(words_to_edit, oovs_lex)
