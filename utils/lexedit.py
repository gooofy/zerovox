#!/bin/env python3

import argparse
import os
import sys
import atexit
import random
import readline
import multiprocessing
import sounddevice as sd

from zerovox.g2p.g2p import G2P
from zerovox.tts.synthesize import ZeroVoxTTS, DEFAULT_REFAUDIO
from zerovox.tts.model import DEFAULT_MELDEC_MODEL_NAME
from zerovox.lexicon.lexedit import LexEdit
from zerovox.g2p.g2p import DEFAULT_G2P_MODEL_NAME_DE, DEFAULT_G2P_MODEL_NAME_EN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='lexedit', description='lexicon editor')

    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default=choices[0],
                        choices=choices,
                        type=str,
                        help="Inference device",)
    parser.add_argument("-l", "--lang",
                        default='en',
                        choices=['en', 'de'],
                        type=str,
                        help="language: en or de, default: en",)
    parser.add_argument('--compile',
                        action='store_true',
                        help='Infer using the compiled model')    
    parser.add_argument("--model",
                        default=ZeroVoxTTS.get_default_model(),
                        help=f"TTS model to use: Path to model directory or model name, default: {ZeroVoxTTS.get_default_model()}")
    parser.add_argument("--meldec-model",
                        default=DEFAULT_MELDEC_MODEL_NAME,
                        type=str,
                        help=f"MELGAN model to use (meldec-libritts-multi-band-melgan-v2 or meldec-libritts-hifigan-v1, default: {DEFAULT_MELDEC_MODEL_NAME})",)
    parser.add_argument("--g2p-model",
                        #default=DEFAULT_G2P_MODEL_NAME,
                        type=str,
                        help="G2P model, default=auto select according to language setting",)                     
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--refaudio", type=str, default=DEFAULT_REFAUDIO, help=f"reference audio wav file, default: {DEFAULT_REFAUDIO}")

    parser.add_argument('-O', '--oovs', type=str, help="OOV file to work on (reads from file, will skip existing entries)")

    parser.add_argument('-e', '--edit', type=str, help="entries to work on (comma separated list of entries to add or review)")

    args = parser.parse_args()

    g2p_model = DEFAULT_G2P_MODEL_NAME_DE if args.lang=='de' else DEFAULT_G2P_MODEL_NAME_EN

    g2p = G2P(args.lang, model=g2p_model)
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

    modelcfg, synth = ZeroVoxTTS.load_model(args.model,
                                            g2p=g2p,
                                            lang=args.lang,
                                            meldec_model=args.meldec_model,
                                            infer_device=args.infer_device,
                                            num_threads=args.threads,
                                            do_compile=args.compile,
                                            verbose=args.verbose)

    sd.default.reset()
    sd.default.samplerate = modelcfg['audio']['sampling_rate']
    sd.default.channels = 1
    sd.default.dtype = 'int16'
    #sd.default.device = None
    #sd.default.latency = 'low'

    if args.verbose:
        print ("computing speaker embedding...")

    spkemb = synth.speaker_embed(ZeroVoxTTS.get_speakerref(args.refaudio, modelcfg['audio']['sampling_rate']))
    histfile = os.path.join(os.path.expanduser("~"), ".lextool_history")
    try:
        readline.read_history_file(histfile)
        # default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)

    sd.default.reset()
    sd.default.samplerate = modelcfg['audio']['sampling_rate']
    sd.default.channels = 1
    sd.default.dtype = 'int16'
    #sd.default.device = None
    #sd.default.latency = 'low'

    editor = LexEdit (g2p, synth, spkemb)

    editor.edit(words_to_edit, oovs_lex)
