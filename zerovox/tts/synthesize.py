'''
zerovox

    Apache 2.0 License
    2024, 2025 by Guenter Bartsch

originally based on:

    EfficientSpeech: An On-Device Text to Speech Model
    https://ieeexplore.ieee.org/abstract/document/10094639
    Rowel Atienza
    Apache 2.0 License
'''

import numpy as np
import torch
from torchinfo import summary
import os
import yaml
import glob
import librosa
import time
import importlib.resources

from pathlib import Path

from zerovox.tts import refaudio, refaudio_local
from zerovox.tts.model import ZeroVox, download_model_file
from zerovox.tts.mels import get_mel_from_wav
from zerovox.tts.symbols import Symbols
from zerovox.tts.normalize import ZeroVoxNormalizer
#from zerovox.parallel_wavegan.bin.preprocess import logmelfilterbank

DEFAULT_TTS_MODEL_NAME_EN='tts_en_zerovox2_medium_2_styledec'
DEFAULT_TTS_MODEL_NAME_DE='tts_de_zerovox2_medium_3_styledec'
DEFAULT_REFAUDIO='en_kevin.wav'

class ZeroVoxTTS:

    @staticmethod
    def get_default_model(lang:str):
        if lang=='en':
            model = os.getenv("ZEROVOX_TTS_MODEL_EN", DEFAULT_TTS_MODEL_NAME_EN)
        elif lang=='de':
            model = os.getenv("ZEROVOX_TTS_MODEL_DE", DEFAULT_TTS_MODEL_NAME_DE)
        return model

    def __init__(self,
                 language: str,
                 syms: Symbols,
                 checkpoint: str | os.PathLike,
                 meldec_model: str,
                 hop_length: int,
                 sampling_rate : int,
                 n_mel_channels : int,
                 fft_size : int,
                 win_length : int,
                 mel_fmin: int,
                 mel_fmax: int,
                 infer_device: str = 'cpu',
                 num_threads: int = -1,
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
        self._verbose = verbose

        self._model = ZeroVox.load_from_checkpoint(lang=language,
                                                   meldec_model=meldec_model,
                                                   sampling_rate=sampling_rate,
                                                   hop_length=hop_length,
                                                   checkpoint_path=str(checkpoint),
                                                   infer_device=infer_device,                                                              
                                                   map_location=torch.device('cpu'),
                                                   strict=False,
                                                   verbose=verbose,
                                                   betas=[0.9,0.99], # FIXME : remove
                                                   eps=1e-9) # FIXME: remove

        self._model = self._model.to(infer_device)
        self._model.eval()

        if num_threads > 0:
            torch.set_num_threads(num_threads)

        self._symbols = syms
        self._normalizer = ZeroVoxNormalizer(language)

    @staticmethod
    def available_speakerrefs():
        speakers = []
        for reffn in importlib.resources.files(refaudio_local).iterdir():
            speakerref = reffn.parts[-1]
            if speakerref.endswith('.wav'):
                speakers.append(speakerref)
        for reffn in importlib.resources.files(refaudio).iterdir():
            speakerref = reffn.parts[-1]
            if speakerref.endswith('.wav'):
                speakers.append(speakerref)
        return sorted(speakers, key=str.casefold)

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

        mel_spectrogram, _ = get_mel_from_wav(audio=wav,
                    sampling_rate=self._sampling_rate,
                    fft_size=self._fft_size, # =1024,
                    hop_size=self._hop_length, # =256,
                    win_length=self._win_length, # =1024,
                    num_mels=self._num_mels, #=80,
                    fmin=self._mel_fmin, #=0,
                    fmax=self._mel_fmax, #=8000,
                    )

        x = np.array([mel_spectrogram.T], dtype=np.float32)
        with torch.no_grad():
            x = torch.from_numpy(x).to(self._infer_device)
            style_embed = self._model._spkemb(x)

        return style_embed

    def transcript2phonemids(self, transcript:str) -> tuple[list[int], list[int]]:

        phones = []
        puncts = []

        punct = 0
        pidx = 0

        while pidx < len(transcript):

            # collapse whitespace, handle punctuation

            p = transcript[pidx]
            if p == ' ' or self._symbols.is_punct(p):

                pu = self._symbols.encode_punct(p)
                if pu>punct:
                    punct = pu

                pidx += 1
                while pidx < len(transcript):
                    p = transcript[pidx]
                    if p != ' ' and not self._symbols.is_punct(p):
                        break

                    pu = self._symbols.encode_punct(p)
                    if pu>punct:
                        punct = pu

                    pidx += 1

                if puncts:
                    puncts[-1] = punct

                continue

            if not self._symbols.is_phone(p):
                pidx += 1
                continue

            punct = 0
            phones.append(self._symbols.encode_phone(p))
            puncts.append(punct)
            pidx += 1

        return phones, puncts

    def text2phonemeids(self, text:str) -> tuple[list[int],list[int]]:

        phone_ids = []
        punct_ids = []

        transcript_uroman, _ = self._normalizer.normalize(text)

        phone_ids, punct_ids = self.transcript2phonemids(transcript_uroman)

        #  6,15,21,24,6,5,6,19,27,22,9,6,13,7,6,15,24,6,15,15
        #  0, 0, 0, 0,0,0,0, 1, 0, 1,0,0, 0,0,0, 2, 0,0, 0, 0
        #  E  n  t  w e d e  r  z  u h e  l f e  n, w e  n  n

        if self._verbose:
            print(f"Raw Text Sequence: {text}")
            print(f"Normalized       : {transcript_uroman}")
            print(f"Phoneme IDs      : {phone_ids}")
            print(f"Punct IDs        : {punct_ids}")

        return phone_ids, punct_ids

    def tts_ex (self, text:str, spkemb, duration=None):
        text = text.strip()

        tstart_g2p = time.time()
        phone_ids, punct_ids = self.text2phonemeids(text)

        if not phone_ids:
            return np.array([[0.0]], dtype=np.float32), np.array([[0]], dtype=np.int32), 0, np.array([[0.0]], dtype=np.float32)

        phoneme   = np.array([phone_ids], dtype=np.int32)
        puncts    = np.array([punct_ids], dtype=np.int32)
        duration  = np.array([duration], dtype=np.int32) if duration is not None else None
        tend_g2p = time.time()

        tstart_synth = time.time()
        with torch.no_grad():
            phoneme = torch.from_numpy(phoneme).int().to(self._infer_device)
            puncts = torch.from_numpy(puncts).int().to(self._infer_device)
            duration = torch.from_numpy(duration).int().to(self._infer_device) if duration is not None else None
            wav, length, _, mel = self._model.inference_ex({"phoneme": phoneme, "puncts": puncts, "duration": duration}, style_embed=spkemb, force_duration = duration is not None)
            wav = wav.cpu().numpy()
        tend_synth = time.time()

        if self._verbose:
            print (f"tts timing stats: g2p={tend_g2p-tstart_g2p}s, synth={tend_synth-tstart_synth}s")

        return wav, phoneme, length, mel.cpu().detach().numpy()

    def tts (self, text:str, spkemb):
        wav, phoneme, length, _ = self.tts_ex(text=text, spkemb=spkemb)
        return wav, phoneme, length

    def summary (self, depth, ref_mel):

        text = "This is a test."

        phone_ids, punct_ids = self.text2phonemeids(text)

        batch_size = 1

        phoneme   = np.array([phone_ids for b in range(batch_size)], dtype=np.int32)
        puncts    = np.array([punct_ids for b in range(batch_size)], dtype=np.int32)
        #ref_mels  = np.array([ref_mel.T for b in range(batch_size)])

        with torch.no_grad():
            # torch.Size([1, 1, 528])
            #spkemb = torch.rand(batch_size, 1, 528)
            phoneme = torch.from_numpy(phoneme).int().to(self._infer_device)
            puncts = torch.from_numpy(puncts).int().to(self._infer_device)
            #ref_mels = torch.from_numpy(ref_mels).to(self._infer_device)
            ref_mels = torch.rand(batch_size, 602, 80).to(self._infer_device)
            x = {"phoneme": phoneme, "puncts": puncts, "ref_mel": ref_mels}

            summary(self._model, input_data = {'x':x, 'force_duration':False, 'normalize_before':True}, depth=depth)

    @property
    def normalizer (self):
        return self._normalizer

    @property
    def language (self):
        return self._normalizer.language

    @language.setter
    def language(self, value):
        if value != self._normalizer.language:
            self._normalizer = ZeroVoxNormalizer(lang=value)

    @property
    def meldec_model (self):
        return self._meldec_model

    @classmethod
    def load_model(cls, 
                   modelpath: str | os.PathLike,
                   meldec_model: str | os.PathLike,
                   infer_device: str = 'cpu',
                   num_threads: int = -1,
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

        synth = ZeroVoxTTS ( language=modelcfg['lang'][0],
                             syms=Symbols(phones=modelcfg['model']['phones'], puncts=modelcfg['model']['puncts']),
                             checkpoint=checkpoint,
                             meldec_model=str(meldec_model),
                             hop_length=modelcfg['audio']['hop_size'],
                             win_length=modelcfg['audio']['win_length'],
                             mel_fmin=modelcfg['audio']['fmin'],
                             mel_fmax=modelcfg['audio']['fmax'],
                             sampling_rate=modelcfg['audio']['sampling_rate'],
                             n_mel_channels=modelcfg['audio']['num_mels'],
                             fft_size=modelcfg['audio']['fft_size'],
                             infer_device=infer_device,
                             num_threads=num_threads,
                             verbose=verbose)
        
        return modelcfg, synth

