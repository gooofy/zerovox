#!/bin/env python3

import os
import sys
import atexit
import random
import readline
import multiprocessing
import sounddevice

from zerovox.g2p.g2p import G2P
from zerovox.tts.synthesize import ZeroVoxTTS
# from zerovox.g2p.g2p import DEFAULT_G2P_MODEL_NAME_DE

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
                else:
                    break

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

            else:
                print ("unknown command, enter h for help")

