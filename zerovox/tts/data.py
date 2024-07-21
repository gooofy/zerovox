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

import json
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from lightning import LightningDataModule

from zerovox.lexicon  import Lexicon
from zerovox.g2p.data import G2PSymbols

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1) #.to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

class LJSpeechDataModule(LightningDataModule):

    def __init__(self, preprocess_configs, symbols: G2PSymbols, batch_size=64, num_workers=4):
        super(LJSpeechDataModule, self).__init__()
        self.preprocess_configs = preprocess_configs
        self._symbols = symbols
        self.batch_size = batch_size
        self.num_workers = num_workers

    def collate_fn(self, batch):
        x, y = zip(*batch)
        len_arr = np.array([d["phoneme"].shape[0] for d in x])
        idxs = np.argsort(-len_arr).tolist()

        phonemes = [x[idx]["phoneme"] for idx in idxs]
        puncts = [x[idx]["puncts"] for idx in idxs]
        texts = [x[idx]["text"] for idx in idxs]
        mels = [y[idx]["mel"] for idx in idxs]
        pitches = [x[idx]["pitch"] for idx in idxs]
        energies = [x[idx]["energy"] for idx in idxs]
        durations = [x[idx]["duration"] for idx in idxs]
        basenames = [x[idx]["basename"] for idx in idxs]
        preprocessed_paths = [x[idx]["preprocessed_path"] for idx in idxs]

        phoneme_lens = np.array([phoneme.shape[0] for phoneme in phonemes])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        phonemes = pad_1D(phonemes)
        puncts  = pad_1D(puncts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        phonemes = torch.from_numpy(phonemes).int()
        puncts = torch.from_numpy(puncts).int()
        phoneme_lens = torch.from_numpy(phoneme_lens).int()
        max_phoneme_len = torch.max(phoneme_lens).item()
        phoneme_mask = get_mask_from_lengths(phoneme_lens, max_phoneme_len) 

        pitches = torch.from_numpy(pitches).float()
        energies = torch.from_numpy(energies).float()
        durations = torch.from_numpy(durations).int()

        mels = torch.from_numpy(mels).float()
        mel_lens = torch.from_numpy(mel_lens).int()
        max_mel_len = torch.max(mel_lens).item()
        mel_mask = get_mask_from_lengths(mel_lens, max_mel_len)

        x = {"phoneme": phonemes,
             "puncts": puncts,
             "phoneme_len": phoneme_lens,
             "phoneme_mask": phoneme_mask,
             "text": texts,
             "mel_len": mel_lens,
             "mel_mask": mel_mask,
             "pitch": pitches,
             "energy": energies,
             "duration": durations,
             "ref_mel": mels,
             "basenames": basenames,
             "preprocessed_paths": preprocessed_paths}

        y = {"mel": mels,}

        return x, y

    def prepare_data(self):
        self.train_dataset = LJSpeechDataset("train.txt",
                                             self.preprocess_configs,
                                             self._symbols)

        #print("Train dataset size: {}".format(len(self.train_dataset)))

        self.test_dataset = LJSpeechDataset("val.txt",
                                            self.preprocess_configs,
                                            self._symbols)

        #print("Test dataset size: {}".format(len(self.test_dataset)))

    #def setup(self, stage=None):
    #    self.prepare_data()

    def train_dataloader(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.num_workers)
        return self.train_dataloader

    def test_dataloader(self):
        self.test_dataloader = DataLoader(self.test_dataset,
                                          shuffle=False,
                                          batch_size=self.batch_size,
                                          collate_fn=self.collate_fn,
                                          num_workers=self.num_workers)
        return self.test_dataloader

    def val_dataloader(self):
        return self.test_dataloader()


class LJSpeechDataset(Dataset):

    def __init__(self, filename, preprocess_configs, symbols: G2PSymbols):

        self._symbols        = symbols
        self.max_text_length = 0

        self.preprocessed_paths = []
        self.basenames          = []
        self.texts              = []
        self.raw_texts          = []
        self.puncts             = []

        for pc in preprocess_configs:

            if pc["preprocessing"]["text"]["max_length"] > self.max_text_length:
                self.max_text_length = pc["preprocessing"]["text"]["max_length"]

            preprocessed_paths, basename, text, raw_text, punct = self.process_meta(filename, pc["path"]["preprocessed_path"],
                                                                                    )
            self.preprocessed_paths.extend(preprocessed_paths)
            self.basenames.extend(basename)
            self.texts.extend(text)
            self.raw_texts.extend(raw_text)
            self.puncts.extend(punct)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        basename = self.basenames[idx]
        raw_text = self.raw_texts[idx]
        phonemes = np.array(self._symbols.phones_to_ids(self.texts[idx].split(' ')))
        puncts   = np.array(self._symbols.puncts_to_ids(self.puncts[idx].split(' ')))
        preprocessed_path = self.preprocessed_paths[idx]
        mel_path = os.path.join(
            preprocessed_path,
            "mel",
            f"mel-{basename}.npy",
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            preprocessed_path,
            "pitch",
            f"pitch-{basename}.npy",
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            preprocessed_path,
            "energy",
            f"energy-{basename}.npy",
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            preprocessed_path,
            "duration",
            f"duration-{basename}.npy",
        )
        duration = np.load(duration_path)

        x = {"phoneme": phonemes,
             "puncts": puncts,
             "text": raw_text,
             "pitch": pitch,
             "energy": energy,
             "duration": duration,
             "basename": basename,
             "preprocessed_path": preprocessed_path}

        y = {"mel": mel,}

        return x, y

    def process_meta(self, filename, preprocessed_path):
        with open(
            os.path.join(preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            preprocessed_paths = []
            name = []
            phonemes = []
            raw_text = []
            puncts = []
            for line in f.readlines():
                n, p, r, punct = line.strip("\n").split("|")
                if len(r) > self.max_text_length:
                    continue
                preprocessed_paths.append(preprocessed_path)
                name.append(n)
                phonemes.append(p)
                raw_text.append(r)
                puncts.append(punct)
            return preprocessed_paths, name, phonemes, raw_text, puncts
