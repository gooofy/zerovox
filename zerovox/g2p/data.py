import os
from typing import Tuple, Dict, Any

from tqdm import tqdm

import torch
from torch.utils import data
from lightning import LightningDataModule

from zerovox.lexicon import Lexicon

#DEBUG_LIMIT = 8192
#DEBUG_LIMIT =128
DEBUG_LIMIT = 0


class G2PSymbols:

    _symcfg = {}
    _symcfg['en'] = { 
                    }

    _symcfg['de'] = { 
                      'punctuation' : [",", ".", "?", "!"],
                      'punctmap'    : {" ":"_", "":"#", ",":",", ".":".", "?":"?", "!":"!"},
                      'silences'    : ["sil", "sp", "spn", ""],
                    }

    def __init__(self, graphemes: list[str], phonemes: list[str]):

        self._start_token = '<s>'
        self._end_token = '</s>'
        self._pad_token = '<pad>'

        self._graphemes = graphemes.copy()
        self._graphemes.append(self._start_token)
        self._graphemes.append(self._end_token)
        self._graphemes.append(self._pad_token)
        self._graphset = set(graphemes)

        self._g2idx = {g: idx for idx, g in enumerate(self._graphemes)}
        self._idx2g = {idx: g for idx, g in enumerate(self._graphemes)}

        self._phonemes  = phonemes.copy()
        self._phonemes.append(self._start_token)
        self._phonemes.append(self._end_token)
        self._phonemes.append(self._pad_token)
        self._phoneset = set(self._phonemes)

        self._p2idx = {p: idx for idx, p in enumerate(self._phonemes)}
        self._idx2p = {idx: p for idx, p in enumerate(self._phonemes)}

        self._puncts   = set([",", ".", "?", "!"])
        self._punctmap = {" ":"_", "":"#", ",":",", ".":".", "?":"?", "!":"!"}
        self._silences = set(["sil", "sp", "spn", ""])

        # Mappings from punctuation to numeric ID and vice versa:
        self._punct_to_id = {}
        self._id_to_punct = {}
        idx = 0
        for _, punct in self._punctmap.items():
            self._punct_to_id[punct] = idx
            self._id_to_punct[idx]   = punct
            idx += 1

    def is_punct(self, phone):
        return phone in self._puncts

    def encode_punct(self, punct):
        return self._punctmap[punct]

    @property
    def num_puncts(self):
        return len(self._punctmap)
    
    def puncts_to_ids(self, puncts):
        sequence = []
        for p in puncts:
            if p not in self._punct_to_id:
                continue
            sequence.append(self._punct_to_id[p])
        return sequence
    
    def is_silence(self, phone):
        return phone in self._silences


    @property
    def num_graphemes(self):
        return len(self._graphemes)
    
    def is_grapheme(self, graph):
        return graph in self._graphset

    @property
    def num_phonemes(self):
        return len(self._phonemes)

    def is_phone(self, phone):
        return phone in self._phoneset

    @property
    def g2idx(self):
        return self._g2idx

    @property
    def p2idx(self):
        return self._p2idx

    @property
    def start_token(self):
        return self._start_token

    @property
    def start_token_pidx(self):
        return self._p2idx[self._start_token]

    @property
    def start_token_gidx(self):
        return self._g2idx[self._start_token]

    @property
    def end_token(self):
        return self._end_token

    @property
    def end_token_pidx(self):
        return self._p2idx[self._end_token]

    @property
    def end_token_gidx(self):
        return self._g2idx[self._end_token]

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def pad_token_pidx(self):
        return self._p2idx[self._pad_token]

    @property
    def pad_token_gidx(self):
        return self._g2idx[self._pad_token]

    def convert_ids_to_graphemes(self, ids):
        graphemes = []
        for idx in ids:
            if idx == self.end_token_gidx:
                break
            p = self._idx2g[idx]
            graphemes.append(p)
        return graphemes

    def convert_ids_to_phonemes(self, ids):
        phonemes = []
        for idx in ids:
            if idx == self.end_token_pidx:
                break
            p = self._idx2p[idx]
            phonemes.append(p)
        return phonemes
    
    def phones_to_ids(self, phones):
        sequence = []
        for p in phones:
            if p not in self._p2idx:
                continue
            sequence.append(self._p2idx[p])

        return sequence
    
class G2PDataset(data.Dataset):

    def __init__(self, words: list[str], prons : list[str], symbols:G2PSymbols):
        """
        words: list of words. e.g., ["w o r d", ]
        prons: list of prons. e.g., ['W ER1 D',]
        """
        self.words = words
        self.prons = prons
        self._symbols = symbols

    def _encode_x(self, inp):
        tokens = [self._symbols.start_token] + inp.lower().split() + [self._symbols.end_token]

        d = self._symbols.g2idx

        x = [d[t] for t in tokens]
        return x

    def _encode_y(self, inp):

        tokens = [self._symbols.start_token] + inp.split() + [self._symbols.end_token]

        d = self._symbols.p2idx

        y = [d[t] for t in tokens]

        return y

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word, pron = self.words[idx], self.prons[idx]
        x = self._encode_x(word)
        y = self._encode_y(pron)
        #decoder_input, y = y[:-1], y[1:]

        #return {'graph_idxs': x, 'graph': word, decoder_input, y, y_seqlen, pron}
        return {'graph_idxs': x, 'graphemes': word, 'phone_idxs': y, 'phonemes': pron}

class G2PDataModule(LightningDataModule):

    def __init__ (self, config: Dict[str, Any], lang: str, num_workers:int=12, batch_size:int=32):
        super(G2PDataModule, self).__init__()

        self._batch_size = batch_size
        self._num_workers = num_workers

        self._graphemes = sorted(list(config['preprocessing']['graphemes']))
        self._phonemes = sorted(config['preprocessing']['phonemes'])

        self._lexicon = Lexicon.load(lang)
        self._lexicon.verify_symbols(self._graphemes, self._phonemes)

        self._symbols = G2PSymbols (self._graphemes, self._phonemes)

    @property
    def symbols(self):
        return self._symbols

    def prepare_data(self) -> None:

        self._words = []
        self._prons = []

        for w, p in self._lexicon.items():
            self._words.append(" ".join(list(w)))
            self._prons.append(" ".join(p))

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            self._train_words = []
            self._train_prons = []
            self._val_words   = []
            self._val_prons   = []

            for i in range(len(self._words)):

                if i % 100 == 0:
                    self._val_words.append(self._words[i])
                    self._val_prons.append(self._prons[i])
                else:
                    self._train_words.append(self._words[i])
                    self._train_prons.append(self._prons[i])

            # num_train, num_val = int(len(self._words)*.9), int(len(self._words)*.1)
            # self._train_words, self._val_words = self._words[:num_train], self._words[-num_val:]
            # self._train_prons, self._val_prons = self._prons[:num_train], self._prons[-num_val:]

            print (f"# entries: train: {len(self._train_words)}, val: {len(self._val_words)}")

            self._train_dataset = G2PDataset(self._train_words, self._train_prons, self._symbols)
            self._val_dataset   = G2PDataset(self._val_words  , self._val_prons  , self._symbols)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            assert False # FIXME

        if stage == "predict":
            assert False # FIXME

    # def _drop_lengthy_samples(words, prons, enc_maxlen, dec_maxlen):
    #     """We only include such samples less than maxlen."""
    #     _words, _prons = [], []
    #     for w, p in zip(words, prons):
    #         if len(w.split()) + 1 > enc_maxlen: continue
    #         if len(p.split()) + 1 > dec_maxlen: continue # 1: <EOS>
    #         _words.append(w)
    #         _prons.append(p)
    #     return _words, _prons

    def _pad(self, batch):
        '''Pads pad_tokens such that the length of all samples in a batch is the same.'''

        # {'graph_idxs': x, 'graphemes': word, 'phone_idxs': y, 'phonemes': pron}

        x_maxlen = max([len(s['graph_idxs']) for s in batch])
        y_maxlen = max([len(s['phone_idxs']) for s in batch])

        padf = lambda key, maxlen, pad_token: [sample[key]+[pad_token]*(maxlen-len(sample[key])) for sample in batch]

        batch_padded = {
                'graph_idxs' : torch.tensor(padf ('graph_idxs', x_maxlen, self._symbols.pad_token_gidx), dtype=torch.long),
                'graphemes'  : [sample['graphemes'] for sample in batch],
                'phone_idxs' : torch.tensor(padf ('phone_idxs', y_maxlen, self._symbols.pad_token_pidx), dtype=torch.long),
                'phonemes'   : [sample['phonemes'] for sample in batch],
            }

        return batch_padded

    def train_dataloader(self):
        return data.DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, collate_fn=self._pad)

    def val_dataloader(self):
        return data.DataLoader(self._val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, collate_fn=self._pad)

    def test_dataloader(self):
        assert False # FIXME

    def predict_dataloader(self):
        assert False # FIXME
