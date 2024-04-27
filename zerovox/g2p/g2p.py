# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p

adaptation to german 2024 by G. Bartsch
'''
import nltk
from nltk.tokenize import TweetTokenizer
word_tokenize = TweetTokenizer().tokenize

import numpy as np
import re
import os

from num2words import num2words

dirname = os.path.dirname(__file__)

class G2p(object):
    def __init__(self):
        super().__init__()
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyzüöäß")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['ts', 'ə', 'iː', 'oː', 'pf', 'aj', 'd', 'tʃ', 'm', 'œ', 'z', 'ɛ', 'ɲ', 't', 'ɟ', 'n̩', 'b', 'ɪ', 'kʰ', 'h', 'eː', 'ɔ', 'f', 'v', 'l̩', 'n', 'x', 'yː', 'p', 'c', 'aː', 'ç', 'uː', 'ʃ', 'øː', 'a', 'l', 'j', 'ɔʏ', 'cʰ', 'aw', 'ŋ', 'ɐ', 'ʊ', 'pʰ', 'ʁ', 's', 'ʏ', 'ɡ', 'tʰ', 'k', 'm̩']

        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.load_lexicon()
        self.load_variables()

    def load_lexicon(self):

        self.lexicon = {} # word -> [ phonemes ]

        with open (os.path.join(dirname, 'german_mfa.dict'), 'r') as lexf:
            for line in lexf:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                graph = parts[0]
                phonemes = parts[1].split(' ')
                valid = True
                for c in graph:
                    if not c in self.graphemes:
                        valid = False
                        break

                if not valid:
                    continue

                self.lexicon[graph] = phonemes


    def load_variables(self):
        self.variables = np.load(os.path.join(dirname,'checkpoint_de.npz'))
        self.enc_emb = self.variables["enc_emb"]  # (33, 64). (len(graphemes), emb)
        self.enc_w_ih = self.variables["enc_w_ih"]  # (3*128, 64)
        self.enc_w_hh = self.variables["enc_w_hh"]  # (3*128, 128)
        self.enc_b_ih = self.variables["enc_b_ih"]  # (3*128,)
        self.enc_b_hh = self.variables["enc_b_hh"]  # (3*128,)

        self.dec_emb = self.variables["dec_emb"]  # (56, 64). (len(phonemes), emb)
        self.dec_w_ih = self.variables["dec_w_ih"]  # (3*128, 64)
        self.dec_w_hh = self.variables["dec_w_hh"]  # (3*128, 128)
        self.dec_b_ih = self.variables["dec_b_ih"]  # (3*128,)
        self.dec_b_hh = self.variables["dec_b_hh"]  # (3*128,)
        self.fc_w = self.variables["fc_w"]  # (56, 128)
        self.fc_b = self.variables["fc_b"]  # (56,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0  # initial hidden state
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)

        return x

    def predict(self, word):
        # encoder
        enc = self.encode(word)
        enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
                       self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3: break  # 3: </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def tokenize (self, text):

        # preprocessing

        text = text.lower().encode('latin1', errors='ignore').decode('latin1')
        text = text.replace("z.b.", "zum beispiel")
        text = text.replace("d.h.", "das heißt")

        # tokenization
        tokens = word_tokenize(text)

        res = []
        for token in tokens:

            if re.search("[0-9]", token):
                try:
                    token = num2words(token, lang='de')
                except:
                    pass

            res.append(token)

        return res

    def lookup (self, token):
        if token in self.lexicon:
            return self.lexicon[token]
        return None

    def __call__(self, text):
        # preprocessing

        text = text.lower().encode('latin1', errors='ignore').decode('latin1')
        text = text.replace("z.b.", "zum beispiel")
        text = text.replace("d.h.", "das heißt")

        # tokenization
        words = word_tokenize(text)
        #breakpoint()
        # tokens = pos_tag(words)  # tuples of (word, tag)

        prons = []
        for token in words:

            if re.search("[0-9]", token):
                token = num2words(token, lang='de')

            if re.search("[a-züöäß]", token) is None:
                pron = [token]
            elif token in self.lexicon:
                pron = self.lexicon[token]
            else: # predict for oov
                pron = self.predict(token)
                # print (f"predicted: {token} [{pron}]")

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

if __name__ == '__main__':
    texts = ["Ich habe 250 Euro in meiner Tasche.", # number -> spell-out
             "Verschiedene Haustiere, z.B. Hunde und Katzen", # z.B. -> zum Beispiel
             "KI ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen befasst.",
             "Dazu gehören nichtsteroidale Antirheumatika (z. B. Acetylsalicylsäure oder Ibuprofen), Lithium, Digoxin, Dofetilid oder Fluconazol"]
    g2p = G2p()
    for text in texts:
        out = g2p(text)
        print(out)

