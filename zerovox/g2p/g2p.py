# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p

adaptation to german / dp models 2024 by G. Bartsch
'''

import re
import yaml
from pathlib import Path

import torch
from nltk.tokenize import TweetTokenizer

from num2words import num2words

from zerovox import download_model_file
from zerovox.lexicon import Lexicon
from zerovox.g2p.data import G2PSymbols
from zerovox.g2p.model import ModelType, LightningTransformer


MODEL_NAME    = "zerovox-g2p-autoreg"
MODEL_VERSION = "1"

class G2P(object):

    def __init__(self, lang: str, infer_device: str='cpu'):

        super().__init__()

        self._cfg_path  = download_model_file(lang=lang, model=MODEL_NAME, version=MODEL_VERSION, relpath='config.yaml')
        self._ckpt_path = download_model_file(lang=lang, model=MODEL_NAME, version=MODEL_VERSION, relpath='best.ckpt')

        config = yaml.load( open(self._cfg_path, "r"), Loader=yaml.FullLoader)
        self._graphemes = list(config['preprocessing']['graphemes'])
        self._phonemes = config['preprocessing']['phonemes']

        model_type = ModelType(config['model']['type'])

        self._symbols = G2PSymbols (self._graphemes, self._phonemes)

        self._infer_device = infer_device
        self._model = LightningTransformer.load_from_checkpoint(self._ckpt_path,
                                                                model_type=model_type,
                                                                config=config,
                                                                symbols=self._symbols,
                                                                map_location=infer_device)
        self._model.eval()

        self._lang = lang

        self._lex = Lexicon.load(lang)

        self._word_tokenize = TweetTokenizer().tokenize


    def predict(self, word:str) -> tuple [list[str], float]:

        tokens = [self._symbols.start_token] + list(word.lower()) + [self._symbols.end_token]

        d = self._symbols.g2idx

        x = [d[t] for t in tokens]

        with torch.no_grad():
            inputs = torch.tensor([x], dtype=torch.long).to(device=self._infer_device)
            out_indices, out_probs = self._model.generate(inputs, self._symbols.start_token_pidx)

        phonemes = self._symbols.convert_ids_to_phonemes(out_indices[0].cpu().tolist()[1:])

        probs = out_probs[0].cpu().tolist()[1:]

        prob = sum(probs) / float (len(probs)) if len(probs)>0 else 0.0

        return phonemes, prob


    def tokenize (self, text:str) -> list[str]:

        # preprocessing

        text = text.lower().encode('latin1', errors='ignore').decode('latin1')
        text = text.replace("z.b.", "zum beispiel")
        text = text.replace("d.h.", "das heißt")

        # tokenization
        tokens = self._word_tokenize(text)

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

    def __call__(self, text:str) -> list[str]:

        tokens = self.tokenize(text)

        prons = []
        for token in tokens:

            if re.search("[a-züöäß]", token) is None:
                pron = [token]
            elif token in self._lex:
                pron = self._lex[token]
            else: # predict for oov
                pron, _ = self.predict(token)
                # print (f"predicted: {token} [{pron}]")

            prons.extend(pron)
            prons.append(" ")

        return prons[:-1]




if __name__ == '__main__':
    texts = ["Ich habe 250 Euro in meiner Tasche.", # number -> spell-out
             "Verschiedene Haustiere, z.B. Hunde und Katzen", # z.B. -> zum Beispiel
             "KI ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen befasst.",
             "Dazu gehören nichtsteroidale Antirheumatika (z. B. Acetylsalicylsäure oder Ibuprofen), Lithium, Digoxin, Dofetilid oder Fluconazol"]
    g2p = G2P('de')
    for text in texts:
        out = g2p(text)
        print(out)

