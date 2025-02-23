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
import sys
import numpy as np
from random import randrange
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from lightning import LightningDataModule

from zerovox.tts.symbols  import Symbols

MAX_REF_LEN = 500 # approx 5.5 seconds (24000/256*6)

def get_mask_from_lengths(lengths, max_len):
    batch_size = lengths.shape[0]

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1) #.to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

class LJSpeechDataModule(LightningDataModule):

    def __init__(self, corpora, symbols: Symbols, stats, num_bins, batch_size=64, num_workers=4):
        super(LJSpeechDataModule, self).__init__()
        self.corpora = corpora
        self._symbols = symbols
        self._stats = stats
        self._num_bins = num_bins
        self.batch_size = batch_size
        self.num_workers = num_workers

    def collate_fn(self, batch):

        phoneme_lens = torch.tensor([sample[0]['phoneme'].shape[0] for sample in batch], dtype=torch.int32)
        mel_lens     = torch.tensor([sample[1]['mel'].shape[0]     for sample in batch], dtype=torch.int32)

        phonemes   = pad_sequence([sample[0]['phoneme' ] for sample in batch], batch_first=True)
        puncts     = pad_sequence([sample[0]['puncts'  ] for sample in batch], batch_first=True)
        texts      = [sample[0]['text'] for sample in batch]
        mels       = pad_sequence([sample[1]['mel'     ] for sample in batch], batch_first=True)
        pitches    = pad_sequence([sample[0]['pitch'   ] for sample in batch], batch_first=True)
        energies   = pad_sequence([sample[0]['energy'  ] for sample in batch], batch_first=True)
        durations  = pad_sequence([sample[0]['duration'] for sample in batch], batch_first=True)
        # phonemposs = pad_sequence([torch.tensor(sample[0][''], dtype=torch.) for sample in batch], batch_first=True)
        basenames = [sample[0]['basename'] for sample in batch]
        preprocessed_paths = [sample[0]['preprocessed_path'] for sample in batch]
        starts = [sample[0]['start_hop'] for sample in batch]
        ends = [sample[0]['end_hop'] for sample in batch]

        ref_mel_len = min(mel_lens)
        if ref_mel_len > MAX_REF_LEN:
            ref_mel_len = MAX_REF_LEN
        ref_mels = []
        for i, mel in enumerate(mels):
            off = randrange(0, mel_lens[i]-ref_mel_len+1)
            ref_mels.append(mel[off:off+ref_mel_len])
        ref_mels = torch.stack(ref_mels, dim=0)

        max_phoneme_len = torch.max(phoneme_lens).item()
        phoneme_mask = get_mask_from_lengths(phoneme_lens, max_phoneme_len)

        max_mel_len = torch.max(mel_lens).item()
        mel_mask = get_mask_from_lengths(mel_lens, max_mel_len)

        x = {'phoneme': phonemes,
             'puncts': puncts,
             'phoneme_len': phoneme_lens,
             'phoneme_mask': phoneme_mask,
             'text': texts,
             'mel_len': mel_lens,
             'mel_mask': mel_mask,
             'pitch': pitches,
             'energy': energies,
             'duration': durations,
             #'phonemepos': phonemposs,
             'ref_mel': ref_mels,
             'basenames': basenames,
             'preprocessed_paths': preprocessed_paths,
             'starts': starts,
             'ends': ends
             }

        y = {'mel': mels,}

        return x, y

    def prepare_data(self):
        self.train_dataset = LJSpeechDataset("train.txt",
                                             self.corpora,
                                             self._symbols,
                                             self._stats,
                                             self._num_bins)

    #def setup(self, stage=None):
    #    self.prepare_data()

    def train_dataloader(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.num_workers)
        return self.train_dataloader


class LJSpeechDataset(Dataset):

    def __init__(self, filename, corpora, symbols: Symbols, stats, num_bins):

        self._symbols        = symbols

        self._preprocessed_paths = []
        self._filenames          = []
        self._phonemes           = []
        self._puncts             = []
        self._transcripts        = []
        self._stats              = stats
        self._num_bins           = num_bins

        for corpus in corpora:
            if os.path.exists(os.path.join(corpus["path"]["preprocessed_path"], filename)):
                preprocessed_paths, filenames, phonemes, puncts, transcripts = self.process_meta(filename, corpus["path"]["preprocessed_path"])
                self._preprocessed_paths.extend(preprocessed_paths)
                self._filenames.extend(filenames)
                self._phonemes.extend(phonemes)
                self._puncts.extend(puncts)
                self._transcripts.extend(transcripts)

    def __len__(self):
        return len(self._transcripts)

    def __getitem__(self, idx):
        
        filename = self._filenames[idx]

        basename = os.path.splitext(filename)[0]

        transcript = self._transcripts[idx]
        phonemes = torch.tensor(self._phonemes[idx], dtype=torch.int32)
        puncts   = torch.tensor(self._puncts[idx], dtype=torch.int32)
        preprocessed_path = self._preprocessed_paths[idx]

        mel_path = os.path.join(preprocessed_path,"mel",f"mel-{basename}.npy")
        mel = torch.from_numpy(np.load(mel_path)).to(torch.float32)

        startstop_path = os.path.join(preprocessed_path,"mel",f"startstop-{basename}.json")
        with open(startstop_path, "r") as f:
            d = json.load(f)
            start_hop = d['start_hop']
            end_hop = d['end_hop']

        pitch_path = os.path.join(preprocessed_path,"pitch",f"pitch-{basename}.npy")
        pitch = torch.from_numpy(np.load(pitch_path)).to(torch.float32)

        pitch = torch.log(pitch - torch.tensor(self._stats['pitch_min']-1.0).expand_as(pitch))
        pitch /= torch.log(torch.tensor(self._stats['pitch_max']-self._stats['pitch_min']+1.0).expand_as(pitch))

        energy_path = os.path.join(preprocessed_path,"energy",f"energy-{basename}.npy")
        energy = torch.from_numpy(np.load(energy_path)).to(torch.float32)
        energy = torch.log(energy - torch.tensor(self._stats['energy_min']-1.0).expand_as(energy))
        energy /= torch.log(torch.tensor(self._stats['energy_max']-self._stats['energy_min']+1.0).expand_as(energy))

        duration_path = os.path.join(preprocessed_path,"duration",f"duration-{basename}.npy")
        duration = torch.from_numpy(np.load(duration_path)).to(torch.int32)

        x = {'phoneme': phonemes,
             'puncts': puncts,
             'text': transcript,
             'pitch': pitch,
             'energy': energy,
             'duration': duration,
             'basename': basename,
             'preprocessed_path': preprocessed_path,
             'start_hop' : start_hop,
             'end_hop' : end_hop}

        y = {'mel': mel,}

        return x, y

    def process_meta(self, filename, preprocessed_path):
        with open(
            os.path.join(preprocessed_path, filename), 'r', encoding='utf-8'
        ) as f:
            preprocessed_paths = []
            filenames = []
            phonemes = []
            transcripts = []
            puncts = []
            for line in f.readlines():
                filename,phoneme,punct,transcript = line.strip("\n").split("|")

                basename = os.path.splitext(filename)[0]
                duration_path = os.path.join(preprocessed_path,"duration",f"duration-{basename}.npy")
                if os.path.exists(duration_path):
                    preprocessed_paths.append(preprocessed_path)
                    filenames.append(filename)
                    phonemes.append([int(p) for p in phoneme.split(',')])
                    puncts.append([int(p) for p in punct.split(',')])
                    transcripts.append(transcript)
                else:
                    print(f"{duration_path} missing -> skipping sample")
            return preprocessed_paths, filenames, phonemes, puncts, transcripts

# if __name__ == "__main__":

#     import tracemalloc
#     import yaml
#     from tqdm import tqdm
#     from setproctitle import setproctitle

#     setproctitle("ZeroTTS_data_memprof")

#     # memory profiling test code

#     CFG='configs/corpora/de_hui/de_hui_Karlsson.yaml'

#     corpora = [yaml.load(open(CFG, "r"), Loader=yaml.FullLoader)]
#     lexicon       = Lexicon.load('de', load_dicts=False)
#     symbols       = G2PSymbols (lexicon.graphemes, lexicon.phonemes)

#     data_module = LJSpeechDataModule(corpora=corpora, symbols=symbols, batch_size=16)
#     data_module.prepare_data()

#     # Start tracing memory allocations
#     tracemalloc.start()

#     dl_train = data_module.train_dataloader()
#     for i in range(100):
#         # Iterate through the train dataloader
#         for batch in tqdm(dl_train, desc=f"i={i}"):
#             #print ('.', end='')
#             #sys.stdout.flush()
#             pass

#     # Stop tracing memory allocations
#     snapshot = tracemalloc.take_snapshot()
#     top_stats = snapshot.statistics('lineno')

#     print("[ Top 10 memory-consuming lines ]")
#     for stat in top_stats[:10]:
#         print(stat)
