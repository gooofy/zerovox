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

import numpy as np
import torch
import os
import json
import glob

from zerovox.tts.model import ZeroVox
from zerovox.g2p.g2p import G2P

class ZeroVoxTTS:

    def __init__(self,
                 language: str,
                 checkpoint: str | os.PathLike,
                 hifigan_checkpoint: str,
                 g2p: G2P,
                 hop_length: int,
                 sampling_rate : int,
                 infer_device: str = 'cpu',
                 num_threads: int = None,
                 do_compile: bool = False):

        self._hop_length = hop_length
        self._infer_device = infer_device

        self._g2p = g2p

        self._model = ZeroVox.load_from_checkpoint(lang=language,
                                                   hifigan_checkpoint=hifigan_checkpoint,
                                                   sampling_rate=sampling_rate,
                                                   hop_length=hop_length,
                                                   checkpoint_path=checkpoint,
                                                   infer_device=infer_device,                                                              
                                                   map_location=torch.device('cpu'))

        self._model = self._model.to(infer_device)
        self._model.eval()

        if num_threads is not None:
            torch.set_num_threads(num_threads)
        if do_compile:
            self._model = torch.compile(self._model, mode="reduce-overhead", backend="inductor")

        self._symbols = self._g2p.symbols

    def ipa2phonemids(self, ipa:list[str]) -> tuple[list[int], list[int]]:

        phones = []
        puncts = []

        punct = self._symbols.encode_punct(' ')
        pidx = 0

        while pidx < len(ipa):

            # collapse whitespace, handle punctuation

            phone = ipa[pidx]
            if phone == ' ' or self._symbols.is_punct(phone):

                punct = self._symbols.encode_punct(phone)

                pidx += 1
                while pidx < len(ipa):
                    phone = ipa[pidx]
                    if not self._symbols.is_punct(phone):
                        break

                    if phone != ' ':
                        punct = self._symbols.encode_punct(phone)

                    pidx += 1

                if puncts:
                    puncts[-1] = punct

                continue

            if not self._symbols.is_phone(phone):
                pidx += 1
                continue

            phones.append(phone)
            puncts.append(punct)
            punct = self._symbols.encode_punct('')
            pidx += 1

        return self._symbols.phones_to_ids(phones), self._symbols.puncts_to_ids(puncts)

    def text2phonemeids(self, text:str, verbose:bool=False) -> list[int]:

        ipa = self._g2p(text)

        phone_ids, punct_ids = self.ipa2phonemids(ipa)

        if verbose:
            print(f"Raw Text Sequence: {text}")
            print(f"Phoneme Sequence : {ipa}")
            print(f"Phoneme IDs      : {phone_ids}")
            print(f"Punct IDs        : {punct_ids}")

        return phone_ids, punct_ids

    def tts (self, text:str, spkemb, verbose:bool=False):
        text = text.strip()

        phone_ids, punct_ids = self.text2phonemeids(text, verbose=verbose)

        if not phone_ids:
            return np.array([[0.0]], dtype=np.float32), np.array([[0]], dtype=np.int32), 0

        phoneme = np.array([phone_ids], dtype=np.int32)
        puncts  = np.array([punct_ids], dtype=np.int32)
        #spkembs = np.array([[spkemb] * len(punct_ids)], dtype=np.int32)
        spkembs = np.array([spkemb], dtype=np.int32)
        with torch.no_grad():
            phoneme = torch.from_numpy(phoneme).int().to(self._infer_device)
            puncts = torch.from_numpy(puncts).int().to(self._infer_device)
            spkembs = torch.from_numpy(spkembs).int().to(self._infer_device)
            wavs, lengths, _ = self._model({"phoneme": phoneme, "puncts": puncts, "spkembs": spkembs})
            wavs = wavs.cpu().numpy()
            lengths = lengths.cpu().numpy()

        wav = np.reshape(wavs, (-1, 1))

        return wav, phoneme, lengths[0]

    def ipa (self, ipa:list[str], spkemb):

        phone_ids, punct_ids = self.ipa2phonemids(ipa)

        if not phone_ids:
            return np.array([[0.0]], dtype=np.float32), 0

        phoneme  = np.array([phone_ids], dtype=np.int32)
        puncts   = np.array([punct_ids], dtype=np.int32)
        spkembs = np.array([spkemb], dtype=np.int32)

        with torch.no_grad():
            phoneme = torch.from_numpy(phoneme).int().to(self._infer_device)
            puncts = torch.from_numpy(puncts).int().to(self._infer_device)
            spkembs = torch.from_numpy(spkembs).int().to(self._infer_device)
            wavs, lengths, _ = self._model({"phoneme": phoneme, "puncts": puncts, "spkembs": spkembs})
            wavs = wavs.cpu().numpy()
            lengths = lengths.cpu().numpy()

        wav = np.reshape(wavs, (-1, 1))

        return wav, lengths[0]

    @classmethod
    def load_model(cls, 
                   modelpath: str | os.PathLike,
                   hifigan_checkpoint: str | os.PathLike,
                   g2p: G2P,
                   infer_device: str = 'cpu',
                   num_threads: int = None,
                   do_compile: bool = False) -> tuple[dict[str, any], "ZeroVoxTTS"]:
        
        with open (os.path.join(modelpath, "modelcfg.json")) as modelcfgf:
            modelcfg = json.load(modelcfgf)

        list_of_files = glob.glob(os.path.join(modelpath, 'checkpoints/*.ckpt'))
        checkpoint = max(list_of_files, key=os.path.getctime)

        synth = ZeroVoxTTS ( language=modelcfg['lang'],
                             checkpoint=checkpoint,
                             hifigan_checkpoint=hifigan_checkpoint,
                             g2p=g2p,
                             hop_length=modelcfg['hop_length'],
                             infer_device=infer_device,
                             num_threads=num_threads,
                             do_compile=do_compile,
                             sampling_rate=modelcfg['sampling_rate'])
        
        return modelcfg, synth

