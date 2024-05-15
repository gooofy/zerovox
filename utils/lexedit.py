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

phonehelp = {'a': 'a', 'aɪ': 'aI', 'aʊ': 'aU', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u', 'y': 'y', 'æ': '{', 'ø': '2', 'œ': '9', 'ɐ': '6', 'ɑ': 'A', 'ɔ': 'O', 'ɔɪ': 'OI', 'ɔʏ': 'OY', 'ə': '@', 'ɛ': 'E', 'ɜ': '3', 'ɪ': 'I', 'ʊ': 'U', 'ʌ': 'V', 'ʏ': 'Y', 'b': 'b', 'd': 'd', 'f': 'f', 'g': 'g', 'h': 'h', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'r': 'r', 's': 's', 't': 't', 'v': 'v', 'w': 'w', 'x': 'x', 'z': 'z', 'ç': 'C', 'ð': 'D', 'ɥ': 'H', 'ɳ': 'N', 'ʁ': 'R', 'ʃ': 'S', 'ʒ': 'Z', 'θ': 'T'}

ipa2xsampa = {
    "'a" : "'a",
    "'aɪ" : "'aI",
    "'aʊ" : "'aU",
    "'aː" : "'a:",
    "'e" : "'e",
    "'eː" : "'e:",
    "'i" : "'i",
    "'iː" : "'i:",
    "'o" : "'o",
    "'oː" : "'o:",
    "'u" : "'u",
    "'uː" : "'u:",
    "'y" : "'y",
    "'yː" : "'y:",
    "'æ" : "'{",
    "'ø" : "'2",
    "'øː" : "'2:",
    "'œ" : "'9",
    "'ɐ" : "'6",
    "'ɑ" : "'A",
    "'ɑː" : "'A:",
    "'ɔ" : "'O",
    "'ɔɪ" : "'OI",
    "'ɔʏ" : "'OY",
    "'ɔː" : "'O:",
    "'ə" : "'@",
    "'ɛ" : "'E",
    "'ɛː" : "'E:",
    "'ɜ" : "'3",
    "'ɜː" : "'3:",
    "'ɪ" : "'I",
    "'ʊ" : "'U",
    "'ʌ" : "'V",
    "'ʏ" : "'Y",
    "'ʔa" : "'?a",
    "'ʔaɪ" : "'?aI",
    "'ʔaʊ" : "'?aU",
    "'ʔaː" : "'?a:",
    "'ʔe" : "'?e",
    "'ʔeː" : "'?e:",
    "'ʔi" : "'?i",
    "'ʔiː" : "'?i:",
    "'ʔo" : "'?o",
    "'ʔoː" : "'?o:",
    "'ʔu" : "'?u",
    "'ʔuː" : "'?u:",
    "'ʔy" : "'?y",
    "'ʔyː" : "'?y:",
    "'ʔæ" : "'?{",
    "'ʔø" : "'?2",
    "'ʔøː" : "'?2:",
    "'ʔœ" : "'?9",
    "'ʔɑ" : "'?A",
    "'ʔɑː" : "'?A:",
    "'ʔɔ" : "'?O",
    "'ʔɔɪ" : "'?OI",
    "'ʔɔʏ" : "'?OY",
    "'ʔə" : "'?@",
    "'ʔɛ" : "'?E",
    "'ʔɛː" : "'?E:",
    "'ʔɪ" : "'?I",
    "'ʔʊ" : "'?U",
    "'ʔʌ" : "'?V",
    "'ʔʏ" : "'?Y",
    "a" : "a",
    "aɪ" : "aI",
    "aʊ" : "aU",
    "aː" : "a:",
    "b" : "b",
    "d" : "d",
    "e" : "e",
    "eː" : "e:",
    "f" : "f",
    "g" : "g",
    "h" : "h",
    "i" : "i",
    "iː" : "i:",
    "j" : "j",
    "k" : "k",
    "l" : "l",
    "m" : "m",
    "n" : "n",
    "o" : "o",
    "oː" : "o:",
    "p" : "p",
    "r" : "r",
    "s" : "s",
    "t" : "t",
    "u" : "u",
    "uː" : "u:",
    "v" : "v",
    "w" : "w",
    "x" : "x",
    "y" : "y",
    "yː" : "y:",
    "z" : "z",
    "æ" : "{",
    "ç" : "C",
    "ð" : "D",
    "ø" : "2",
    "øː" : "2:",
    "œ" : "9",
    "ɐ" : "6",
    "ɑ" : "A",
    "ɑː" : "A:",
    "ɔ" : "O",
    "ɔɪ" : "OI",
    "ɔʏ" : "OY",
    "ɔː" : "O:",
    "ə" : "@",
    "ɛ" : "E",
    "ɛː" : "E:",
    "ɜ" : "3",
    "ɜː" : "3:",
    "ɥ" : "H",
    "ɪ" : "I",
    "ɳ" : "N",
    "ʁ" : "R",
    "ʃ" : "S",
    "ʊ" : "U",
    "ʌ" : "V",
    "ʏ" : "Y",
    "ʒ" : "Z",
    "ʔa" : "?a",
    "ʔaɪ" : "?aI",
    "ʔaʊ" : "?aU",
    "ʔaː" : "?a:",
    "ʔe" : "?e",
    "ʔeː" : "?e:",
    "ʔi" : "?i",
    "ʔiː" : "?i:",
    "ʔo" : "?o",
    "ʔoː" : "?o:",
    "ʔu" : "?u",
    "ʔuː" : "?u:",
    "ʔy" : "?y",
    "ʔyː" : "?y:",
    "ʔæ" : "?{",
    "ʔøː" : "?2:",
    "ʔɐ" : "?6",
    "ʔɔ" : "?O",
    "ʔɔɪ" : "?OI",
    "ʔɔʏ" : "?OY",
    "ʔɔː" : "?O:",
    "ʔə" : "?@",
    "ʔɛ" : "?E",
    "ʔɛː" : "?E:",
    "ʔɪ" : "?I",
    "ʔʊ" : "?U",
    "ʔʏ" : "?Y",
    "θ" : "T",
}

xsampa2ipa = { }

for ipa, xs in ipa2xsampa.items():
    xsampa2ipa [xs] = ipa

# https://stackoverflow.com/questions/8505163/is-it-possible-to-prefill-a-input-in-python-3s-command-line-interface
def input_with_prefill(prompt, text):
    def hook():
        readline.insert_text(text)
        readline.redisplay()
    readline.set_pre_input_hook(hook)
    result = input(prompt)
    readline.set_pre_input_hook()
    return result

class LexEdit:

    def __init__ (self, g2p: G2P, synth: ZeroVoxTTS, spkemb):

        self._g2p      = g2p
        self._lex      = g2p.lex
        self._synth    = synth
        self._spkemb   = spkemb

    def _get_ipa (self, word, oovs_lex, gen=False):
        if word in self._lex:
            return self._lex[word]
        if not gen and word in oovs_lex:
            return oovs_lex[word]
        ipa, _ = self._g2p.predict(word)
        return ipa

    def say(self):

        if not self._synth:
            return
        
        #ipa = ['oː', '{sp}'] + self._ipa + ['{sp}', 'oː', '.']
        ipa = self._ipa + ['.']

        wav, _ = self._synth.ipa(ipa, self._spkemb)
 
        sounddevice.play(wav)
        #sounddevice.wait()

        #write_wav_to_file(wav, length=length, filename='foo.wav', sample_rate=self._sample_rate, hop_length=self._hop_length)

    def edit(self, words:list[str], oovs_lex:dict[str, str]) -> bool:
        if not words:
            print ("no words given to edit")
            return False
        
        self._wordidx = 0
        self._word = words[self._wordidx]
        self._ipa = self._get_ipa (self._word, oovs_lex)
        # self._ipa = ['a'] # FIXME: remove!

        self.say()

        while True:

            cmd = input(f"#{self._wordidx:3}/{len(words)}: {self._word} [ {' '.join(self._ipa)} ] > ")

            if cmd == 'h' or cmd == 'help' or cmd == '?':

                print ("available phones:")
                for ipa in sorted (phonehelp):
                    print (f"{phonehelp[ipa]}:{ipa} ", end='')

                print ("h: help")
                print ("s: say")
                print ("p: predict phonemes from given grapheme")
                print ("e: edit")
                print ("g: generate")
                print ("o: generate from oov lex")
                print ("n: next word")
                print ("p: previous word")
                print ("d: delete word")
                print ("q: quit")

            elif cmd == 'q':
                return True

            elif cmd == 'Q':
                return False

            elif cmd == 's':
                self.say()

            elif cmd == 'p':
                grapheme = input ("grapheme > ")
                self._ipa = self._get_ipa (grapheme, oovs_lex)
                self._lex[self._word] = self._ipa
                self._lex.save()
                self.say()

            elif cmd == 'e':

                xs = []
                for p in self._ipa:
                    xs.append(ipa2xsampa [p])

                xs = input_with_prefill("X-Sampa > ", ' '.join(xs))

                ipa = []
                ok=True
                for p in xs.split(' '):
                    if p in xsampa2ipa:
                        ipa.append(xsampa2ipa[p])
                    else:
                        print (f"*** ERROR: unknown XSAMPA symbol {p}")
                        ok=False
                if ok:
                    self._ipa = ipa
                    self._lex[self._word] = self._ipa
                    self._lex.save()

                self.say()

            elif cmd == 'g':

                self._ipa = self._get_ipa (self._word, oovs_lex, gen=True)
                self.say()

            elif cmd == 'o':

                self._ipa = self._get_ipa (self._word, oovs_lex, gen=False)
                self.say()

            elif cmd == 'n':

                self._lex[self._word] = self._ipa
                self._lex.save()

                if self._wordidx < len(words)-1:
                    self._wordidx += 1

                self._word = words[self._wordidx]
                self._ipa = self._get_ipa (self._word, oovs_lex)

                self.say()

            elif cmd == 'p':

                self._lex[self._word] = self._ipa
                if self._wordidx > 0:
                    self._wordidx -= 1

                self._word = words[self._wordidx]
                self._ipa = self._get_ipa (self._word, oovs_lex)

                self.say()

            elif cmd == 'd':

                if self._word in self._lex:
                    del self._lex[self._word]
                words.remove(self._word)
                if self._wordidx >= len(words):
                    self._wordidx = len(words)-1

                self._word = words[self._wordidx]
                self._ipa = self._get_ipa (self._word, oovs_lex)

                self.say()

            elif cmd == 'S':
                print (" ID | speaker")
                print ("----+-----------------------------------------")
                idx=0
                for speaker in self._modelcfg['speakers']:
                    print (f"{idx:3} | {speaker}")
                    idx += 1

                self._speaker = int(input("speaker id > "))

            else:
                print ("unknown command, enter h for help")

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
    parser.add_argument("--hifigan-checkpoint",
                        default="VCTK_V2",
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
                                                hifigan_checkpoint=args.hifigan_checkpoint,
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
