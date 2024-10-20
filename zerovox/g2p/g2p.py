# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p

adaptation to german / dp models 2024 by G. Bartsch
'''

import os
import re
import yaml
import argparse
from pathlib import Path

import torch

from zerovox.lexicon import Lexicon
from zerovox.g2p.data import G2PSymbols
from zerovox.g2p.model import ModelType, LightningTransformer
from zerovox.g2p.tokenizer import G2PTokenizer

DEFAULT_G2P_MODEL_NAME_DE = "zerovox-g2p-autoreg-zamia-de"
DEFAULT_G2P_MODEL_NAME_EN = "zerovox-g2p-autoreg-zamia-en"

def _download_model_file(model:str, relpath:str) -> Path:

    target_dir  = Path(os.getenv("CACHED_PATH_ZEROVOX", Path.home() / ".cache" / "zerovox")) / "model_repo" / model
    target_path = target_dir / relpath

    if target_path.exists():
        return target_path

    os.makedirs (target_dir, exist_ok=True)

    url = f"https://huggingface.co/goooofy/{model}/resolve/main/{relpath}?download=true"

    torch.hub.download_url_to_file(url, str(target_path))

    return target_path


class G2P(object):

    def __init__(self, lang: str, infer_device: str='cpu', model: str|os.PathLike=DEFAULT_G2P_MODEL_NAME_DE):

        super().__init__()

        if os.path.isdir(model):
            self._cfg_path  = Path(model) / 'config.yaml'
            self._ckpt_path = Path(model) / 'best.ckpt'
        else:
            self._cfg_path  = _download_model_file(model=model, relpath='config.yaml')
            self._ckpt_path = _download_model_file(model=model, relpath='best.ckpt')

        self._model_name = model

        config = yaml.load( open(self._cfg_path, "r"), Loader=yaml.FullLoader)
        self._graphemes = sorted(list(config['preprocessing']['graphemes']))
        self._phonemes = sorted(config['preprocessing']['phonemes'])

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

        self._tokenizer = G2PTokenizer(lang)

    @property
    def lex(self):
        return self._lex

    @property
    def symbols(self):
        return self._symbols

    @property
    def model_name(self):
        return self._model_name

    def predict(self, word:str) -> tuple [list[str], float]:

        tokens = [self._symbols.start_token] + list(word.lower()) + [self._symbols.end_token]
        #print(f"tokens: {tokens}")

        d = self._symbols.g2idx

        x = []
        for t in tokens:
            if t not in d:
                continue
            x.append(d[t])
        #print(f"-> x: {x}")

        with torch.no_grad():
            inputs = torch.tensor([x], dtype=torch.long).to(device=self._infer_device)
            out_indices, out_probs = self._model.generate(inputs)

        #print(f"-> out_indices[0]: {out_indices[0]}")
        phonemes = self._symbols.convert_ids_to_phonemes(out_indices[0].cpu().tolist()[1:])

        probs = out_probs[0].cpu().tolist()[1:]

        prob = sum(probs) / float (len(probs)) if len(probs)>0 else 0.0

        return phonemes, prob

    def tokenize (self, text:str) -> list[str]:
        return self._tokenizer.tokenize(text)

    def lookup (self, token):
        if token in self._lex:
            return self._lex[token]
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

    @property
    def phonemes(self) -> list[str]:
        return self._phonemes

    @property
    def graphemes(self) -> list[str]:
        return self._graphemes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    choices = ['cpu', 'gpu']
    parser.add_argument("--infer-device", type=str, default=choices[0], choices=choices)
    parser.add_argument("--lang", type=str, default="de")
    parser.add_argument("--model-path", type=str)

    args = parser.parse_args()

    g2p = G2P(lang=args.lang, infer_device=args.infer_device, model=args.model_path)

    texts = ["open-access-zeitschriften, neue-heimat-bestände und actionabenteuer-adaption.", # deal with hyphens
             "Ich habe 250 Euro in meiner Tasche.", # number -> spell-out
             "Verschiedene Haustiere, z.B. Hunde und Katzen", # z.B. -> zum Beispiel
             "KI ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen befasst.",
             "Dazu gehören nichtsteroidale Antirheumatika (z. B. Acetylsalicylsäure oder Ibuprofen), Lithium, Digoxin, Dofetilid oder Fluconazol"]

    for text in texts:
        out = g2p(text)
        print(out)

    for word in ['', 'Acetylsalicylsäure', 'Wissenschaftler', 'der', 'aber']:
        phonemes, p = g2p.predict(word)
        print (f'{word}: {" ".join(phonemes)}')
