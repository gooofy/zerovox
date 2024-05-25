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

import os
import torch
import librosa
import argparse
import readline   # noqa: F401
import multiprocessing
from pathlib import Path

from zerovox.tts.synthesize import ZeroVoxTTS
from zerovox.g2p.g2p import DEFAULT_G2P_MODEL_NAME
from zerovox.lexicon.lexedit import LexEdit

# https://stackoverflow.com/questions/8505163/is-it-possible-to-prefill-a-input-in-python-3s-command-line-interface
def input_with_prefill(prompt, text):
    def hook():
        readline.insert_text(text)
        readline.redisplay()
    readline.set_pre_input_hook(hook)
    result = input(prompt)
    readline.set_pre_input_hook()
    return result

class Reviewer:

    def __init__(self, modelcfg, synth: ZeroVoxTTS, corpus_path, spkemb, verbose):

        self._modelcfg    = modelcfg
        self._synth       = synth
        self._corpus_path = corpus_path
        self._spkemb      = spkemb
        self._verbose     = verbose

        self._metadata = []
        self._good = set()
        self._bad  = set()
        with open (corpus_path / 'all.csv', 'r') as corpusf:
            for line in corpusf:
                parts = line.strip().split('|')
                self._metadata.append(parts)
                if not os.path.exists(corpus_path / parts[0]):
                    self._bad.add(parts[0])

        with open (corpus_path / 'metadata.csv', 'r') as corpusf:
            for line in corpusf:
                parts = line.strip().split('|')
                self._good.add(parts[0])
        with open (corpus_path / 'bad.csv', 'r') as corpusf:
            for line in corpusf:
                parts = line.strip().split('|')
                self._bad.add(parts[0])

        self._g2p = self._synth.g2p

        self._cur_md = 0
        self._lexedit = LexEdit(self._g2p, self._synth, spkemb)

    def _play(self):
        wav, _ = librosa.load(str(self._corpus_path / self._wavpath))
        sd.stop()
        sd.play(wav)

    def _save(self):
        with open (corpus_path / 'metadata.csv', 'w') as goodf:
            with open (corpus_path / 'bad.csv', 'w') as badf:
                for wavpath, label in self._metadata:
                    if wavpath in self._good:
                        goodf.write(f"{wavpath}|{label}\n")
                    elif wavpath in self._bad:
                        badf.write(f"{wavpath}|{label}\n")

    def _oovcheck(self):
        tokens = self._g2p.tokenize(self._label)
        self._oovs = []
        for token in tokens:
            if self._g2p.symbols.is_punct(token):
                continue
            if token not in self._g2p.lex:
                self._oovs.append(token)

    def _next(self):
        self._cur_md += 1
        while self._cur_md < len(self._metadata):
            self._wavpath = self._metadata[self._cur_md][0]
            self._label   = self._metadata[self._cur_md][1]
            if self._wavpath not in self._good and self._wavpath not in self._bad:
                self._oovcheck()
                return True
            self._cur_md += 1
        return False


    def edit(self):

        self._cur_md = -1
        if not self._next():
            print ("not further utterances to review.")
            return

        while True:

            print (f"{self._cur_md+1:3}/{len(self._metadata)} [good: {len(self._good)}] {self._wavpath}: {self._label}")
            tokens = self._g2p.tokenize(self._label)
            print (repr(tokens))

            if self._oovs:
                print (f"OOVS found: {self._oovs}")

            cmd = input("(h for help) >")

            if cmd == 'h':
                print (" h          help")
                print (" q          quit")
                print (" p          play wav")
                print (" s          synthesize label")
                print (" e          edit label")
                print (" l <word>   lexedit word")
                print (" o          lexedit oovs")
                print (" n          accept (good)")
                print (" b          reject (bad)")

            elif cmd == 'q':
                break

            elif cmd == 'p':

                self._play()

            elif cmd == 'e':

                self._label = input_with_prefill("label > ", self._label)
                self._metadata[self._cur_md][1] = self._label
                self._oovcheck()

            elif cmd == 's':

                if args.infer_device == "cuda":
                    torch.cuda.synchronize()

                wav, _, _ = synth.tts(self._label, self._spkemb, verbose=self._verbose)

                if args.infer_device == "cuda":
                    torch.cuda.synchronize()

                sd.stop()
                sd.play(wav)

            elif cmd == 'n' or cmd == 'b':
                if cmd == 'n':
                    if not self._oovs:
                        self._good.add(self._wavpath)
                    else:
                        print ("***ERROR: unresolved oovs!")
                        continue
                else:
                    self._bad.add(self._wavpath)
                self._save()
                if not self._next():
                    break
                self._play()

            elif cmd.startswith ('l '):

                parts = cmd.split(' ')
                if len(parts) != 2:
                    print ("*** ERROR: usage: l <word>")
                    continue

                graph = parts[1].lower()
                oovsdict = {}
                if graph in self._g2p.lex:
                    oovsdict[graph] = self._g2p.lex[graph]

                self._lexedit.edit([graph], oovsdict)
                self._oovcheck()

            elif cmd == 'o':
                self._lexedit.edit(self._oovs, {})
                self._oovcheck()


            else:
                print ("*** ERROR: unknown command")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='review_lj_corpus', description='interactive lj format corpus review utility')

    parser.add_argument("path", type=str, help="path to corpus dir")

    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default=choices[0],
                        choices=choices,
                        type=str,
                        help="Inference device",)
    parser.add_argument('--compile',
                        action='store_true',
                        help='Infer using the compiled model')    
    parser.add_argument("--model",
                        default=None,
                        required=True,
                        help="Path to model directory",)
    parser.add_argument("--hifigan-checkpoint",
                        default="VCTK_V2",
                        type=str,
                        help="HiFiGAN model",)
    parser.add_argument("--g2p-model",
                        default=DEFAULT_G2P_MODEL_NAME,
                        type=str,
                        help=f"G2P model, default={DEFAULT_G2P_MODEL_NAME}",)                     
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--refaudio", type=str, required=True, help="reference audio wav file")

    args = parser.parse_args()

    modelcfg, synth = ZeroVoxTTS.load_model(args.model,
                                            g2p=args.g2p_model,
                                            hifigan_checkpoint=args.hifigan_checkpoint,
                                            infer_device=args.infer_device,
                                            num_threads=args.threads,
                                            do_compile=args.compile)

    import sounddevice as sd
    sd.default.reset()
    sd.default.samplerate = modelcfg['audio']['sampling_rate']
    sd.default.channels = 1
    sd.default.dtype = 'int16'

    if args.verbose:
        print ("computing speaker embedding...")

    spkemb = synth.speaker_embed(args.refaudio)

    corpus_path = Path(args.path)

    if not os.path.exists(corpus_path / 'all.csv'):
        os.rename(corpus_path / 'metadata.csv', corpus_path / 'all.csv')
        open (corpus_path / 'metadata.csv', 'w').close()
        open (corpus_path / 'bad.csv', 'w').close()


    reviewer = Reviewer(modelcfg=modelcfg, synth=synth, corpus_path=corpus_path, spkemb=spkemb, verbose=args.verbose)
    reviewer.edit()
