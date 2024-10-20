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
import yaml
import glob
import librosa
import time
import importlib.resources

from pathlib import Path

from zerovox.tts import refaudio, refaudio_local
from zerovox.tts.model import ZeroVox, download_model_file
from zerovox.g2p.g2p import G2P
from zerovox.tts.mels import get_mel_from_wav, TacotronSTFT

DEFAULT_TTS_MODEL_NAME='tts_en_de_zerovox_alpha1'
DEFAULT_REFAUDIO='en_speaker_00070.wav'

class ZeroVoxTTS:

    def __init__(self,
                 language: str,
                 checkpoint: str | os.PathLike,
                 meldec_model: str,
                 g2p: G2P,
                 hop_length: int,
                 sampling_rate : int,
                 n_mel_channels : int,
                 fft_size : int,
                 filter_length : int,
                 win_length : int,
                 mel_fmin: int,
                 mel_fmax: int,
                 eps: float,
                 window: str,
                 log_base: float,
                 infer_device: str = 'cpu',
                 num_threads: int = -1,
                 do_compile: bool = False,
                 verbose: bool = False):

        self._hop_length = hop_length
        self._infer_device = infer_device
        self._sampling_rate = sampling_rate

        self._language = language
        self._meldec_model = meldec_model

        self._fft_size = fft_size
        self._win_length = win_length
        self._num_mels = n_mel_channels
        self._mel_fmin = mel_fmin
        self._mel_fmax = mel_fmax
        self._window = window
        self._eps = eps
        self._log_base = log_base
        self._verbose = verbose

        self._g2p = g2p

        self._model = ZeroVox.load_from_checkpoint(lang=language,
                                                   meldec_model=meldec_model,
                                                   sampling_rate=sampling_rate,
                                                   hop_length=hop_length,
                                                   checkpoint_path=str(checkpoint),
                                                   infer_device=infer_device,                                                              
                                                   map_location=torch.device('cpu'),
                                                   strict=False,
                                                   verbose=verbose)

        self._model = self._model.to(infer_device)
        self._model.eval()

        if num_threads > 0:
            torch.set_num_threads(num_threads)
        if do_compile:
            self._model = torch.compile(self._model, mode="reduce-overhead", backend="inductor")

        self._symbols = self._g2p.symbols

        self._stft = TacotronSTFT(
                        filter_length=filter_length,
                        hop_length=hop_length,
                        win_length=win_length,
                        n_mel_channels=n_mel_channels,
                        sampling_rate=sampling_rate,
                        mel_fmin=mel_fmin,
                        mel_fmax=mel_fmax,
                        use_cuda=infer_device != 'cpu')

    @staticmethod
    def available_speakerrefs():
        for reffn in importlib.resources.files(refaudio).iterdir():
            speakerref = reffn.parts[-1]
            if speakerref.endswith('.wav'):
                yield speakerref
        for reffn in importlib.resources.files(refaudio_local).iterdir():
            speakerref = reffn.parts[-1]
            if speakerref.endswith('.wav'):
                yield speakerref

    @staticmethod
    def get_speakerref(speakerref, sampling_rate):
        if os.path.isfile(speakerref):
            wav, _ = librosa.load(speakerref, sr=sampling_rate)
        else:
            if importlib.resources.is_resource(refaudio_local, str(speakerref)):
                wav, _ = librosa.load(importlib.resources.open_binary(refaudio_local, str(speakerref)), sr=sampling_rate)
            else:
                wav, _ = librosa.load(importlib.resources.open_binary(refaudio, str(speakerref)), sr=sampling_rate)
        return wav

    def speaker_embed (self, wav: np.ndarray):

        # Trim the beginning and ending silence
        wav, _ = librosa.effects.trim(wav, top_db=40)

        mel_spectrogram, energy = get_mel_from_wav(wav, 
                                                   sampling_rate=self._sampling_rate,
                                                   fft_size=self._fft_size,
                                                   hop_size=self._hop_length,
                                                   win_length=self._win_length,
                                                   window=self._window,
                                                   num_mels=self._num_mels,
                                                   fmin=self._mel_fmin,
                                                   fmax=self._mel_fmax,
                                                   eps=self._eps,
                                                   log_base=self._log_base,
                                                   stft=self._stft)

        x = np.array([mel_spectrogram.T], dtype=np.float32)
        with torch.no_grad():
            x = torch.from_numpy(x).to(self._infer_device)
            style_embed = self._model._gst(x)

        return style_embed

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
                    if phone != ' ' and not self._symbols.is_punct(phone):
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

    def text2phonemeids(self, text:str) -> list[int]:

        ipa = self._g2p(text)

        phone_ids, punct_ids = self.ipa2phonemids(ipa)

        if self._verbose:
            print(f"Raw Text Sequence: {text}")
            print(f"Phoneme Sequence : {ipa}")
            print(f"Phoneme IDs      : {phone_ids}")
            print(f"Punct IDs        : {punct_ids}")

        return phone_ids, punct_ids

    def tts (self, text:str, spkemb):
        text = text.strip()

        tstart_g2p = time.time()
        phone_ids, punct_ids = self.text2phonemeids(text)

        if not phone_ids:
            return np.array([[0.0]], dtype=np.float32), np.array([[0]], dtype=np.int32), 0

        phoneme = np.array([phone_ids], dtype=np.int32)
        puncts  = np.array([punct_ids], dtype=np.int32)
        tend_g2p = time.time()


        tstart_synth = time.time()
        with torch.no_grad():
            phoneme = torch.from_numpy(phoneme).int().to(self._infer_device)
            puncts = torch.from_numpy(puncts).int().to(self._infer_device)
            wav, length, _ = self._model.inference({"phoneme": phoneme, "puncts": puncts}, style_embed=spkemb)
            wav = wav.cpu().numpy()
        tend_synth = time.time()

        if self._verbose:
            print (f"tts timing stats: g2p={tend_g2p-tstart_g2p}s, synth={tend_synth-tstart_synth}s")

        return wav, phoneme, length

    def ipa (self, ipa:list[str], spkemb):

        phone_ids, punct_ids = self.ipa2phonemids(ipa)

        if not phone_ids:
            return np.array([[0.0]], dtype=np.float32), 0

        phoneme  = np.array([phone_ids], dtype=np.int32)
        puncts   = np.array([punct_ids], dtype=np.int32)

        with torch.no_grad():
            phoneme = torch.from_numpy(phoneme).int().to(self._infer_device)
            puncts = torch.from_numpy(puncts).int().to(self._infer_device)
            wav, length, _ = self._model.inference({"phoneme": phoneme, "puncts": puncts}, style_embed=spkemb)
            wav = wav.cpu().numpy()

        return wav, length

    @property
    def g2p (self):
        return self._g2p

    @property
    def language (self):
        return self._g2p._lang

    @property
    def meldec_model (self):
        return self._meldec_model

    @classmethod
    def load_model(cls, 
                   modelpath: str | os.PathLike,
                   meldec_model: str | os.PathLike,
                   g2p: G2P | str,
                   lang: str,
                   infer_device: str = 'cpu',
                   num_threads: int = -1,
                   do_compile: bool = False,
                   verbose: bool = False) -> tuple[dict[str, any], "ZeroVoxTTS"]:

        # download model from huggingface if necessary

        if os.path.isdir(modelpath):

            config_path = Path(Path(modelpath) / 'modelcfg.yaml')
            list_of_files = glob.glob(os.path.join(modelpath, 'checkpoints/*.ckpt'))
            checkpoint = max(list_of_files, key=os.path.getctime)

        else:

            config_path = download_model_file(model=str(modelpath), relpath="modelcfg.yaml")
            checkpoint  = download_model_file(model=str(modelpath), relpath="checkpoint.pkl")

        if verbose:
            print("synthesize: using config    : ", config_path)
            print("synthesize: using checkpoint: ", checkpoint)

        with open (config_path) as modelcfgf:
            modelcfg = yaml.load(modelcfgf, Loader=yaml.FullLoader)

        if isinstance(g2p, str) :
            g2p = G2P(lang, model=g2p)

        synth = ZeroVoxTTS ( language=modelcfg['lang'],
                             checkpoint=checkpoint,
                             meldec_model=meldec_model,
                             g2p=g2p,
                             hop_length=modelcfg['audio']['hop_size'],
                             filter_length=modelcfg['audio']['filter_length'],
                             win_length=modelcfg['audio']['win_length'],
                             mel_fmin=modelcfg['audio']['mel_fmin'],
                             mel_fmax=modelcfg['audio']['mel_fmax'],
                             sampling_rate=modelcfg['audio']['sampling_rate'],
                             n_mel_channels=modelcfg['audio']['num_mels'],
                             fft_size=modelcfg['audio']['fft_size'],
                             eps=float(modelcfg['audio']['eps']),
                             window=modelcfg['audio']['window'],
                             log_base=modelcfg['audio']['log_base'],
                             infer_device=infer_device,
                             num_threads=num_threads,
                             do_compile=do_compile,
                             verbose=verbose)
        
        return modelcfg, synth

