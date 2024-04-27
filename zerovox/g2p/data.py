import os
from typing import Tuple, Dict, Any

from tqdm import tqdm

import torch
from torch.utils import data
from lightning import LightningDataModule

#DEBUG_LIMIT = 8192
#DEBUG_LIMIT =128
DEBUG_LIMIT = 0

def load_lex(lexicon_path:os.PathLike, graphemes: set[str], phoneset: set[str]) -> dict [str, str]:

    cnt = 0

    with open (lexicon_path, 'r') as lexf:
        for _ in lexf:
            cnt += 1

    lexicon = {} # word -> [ phonemes ]

    extra_graphemes = set()

    with open (lexicon_path, 'r') as lexf:
        for line in tqdm(lexf, total=cnt, desc='load lexicon'):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            graph = parts[0]

            if graph in lexicon:
                continue

            phonemes = parts[1].split(' ')
            valid = True
            for c in graph:
                if not c in graphemes:
                    valid = False
                    extra_graphemes.add(c)
                    print (f"warning: skipping invalid entry {graph} : {phonemes} extra grapheme: {c}")
                    #break

            if not valid:
                continue

            for phoneme in phonemes:
                if phoneme not in phoneset:
                    raise Exception (f"illegal phone {phoneme} detect in entry {graph}")
                
            # print (f"{graph} : {phonemes}")

            lexicon[graph] = phonemes

            if DEBUG_LIMIT and len(lexicon) >= DEBUG_LIMIT:
                break

    print (f"extra graphemes detected: {sorted(extra_graphemes)}")

    return lexicon


class G2PTokenizer:

    def __init__(self, graphemes: list[str], phonemes: list[str]):

        self._start_token = '<s>'
        self._end_token = '</s>'
        self._pad_token = '<pad>'

        self._graphemes = graphemes
        self._graphemes.append(self._start_token)
        self._graphemes.append(self._end_token)
        self._graphemes.append(self._pad_token)

        self._g2idx = {g: idx for idx, g in enumerate(graphemes)}
        self._idx2g = {idx: g for idx, g in enumerate(graphemes)}

        self._phonemes  = phonemes
        self._phonemes.append(self._start_token)
        self._phonemes.append(self._end_token)
        self._phonemes.append(self._pad_token)

        self._p2idx = {p: idx for idx, p in enumerate(phonemes)}
        self._idx2p = {idx: p for idx, p in enumerate(phonemes)}

    @property
    def num_graphemes(self):
        return len(self._graphemes)

    @property
    def num_phonemes(self):
        return len(self._phonemes)

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

class G2PDataset(data.Dataset):

    def __init__(self, words, prons, tokenizer):
        """
        words: list of words. e.g., ["w o r d", ]
        prons: list of prons. e.g., ['W ER1 D',]
        """
        self.words = words
        self.prons = prons
        self._tokenizer = tokenizer

    def _encode_x(self, inp):
        tokens = [self._tokenizer.start_token] + inp.lower().split() + [self._tokenizer.end_token]

        d = self._tokenizer.g2idx

        x = [d[t] for t in tokens]
        return x

    def _encode_y(self, inp):

        tokens = [self._tokenizer.start_token] + inp.split() + [self._tokenizer.end_token]

        d = self._tokenizer.p2idx

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

    def __init__ (self, config: Dict[str, Any], lex_path: os.PathLike, num_workers:int=12, batch_size:int=32):
        super(G2PDataModule, self).__init__()

        self._batch_size = batch_size
        self._num_workers = num_workers

        self._graphemes = set(list(config['preprocessing']['graphemes']))

        self._phonemes = set(config['preprocessing']['phonemes'])

        self._lexicon = load_lex(lex_path, self._graphemes, self._phonemes)

        self._graphemes = sorted(list(self._graphemes))
        self._phonemes = sorted(list(self._phonemes))

        self._tokenizer = G2PTokenizer (self._graphemes, self._phonemes)

    @property
    def tokenizer(self):
        return self._tokenizer

    def prepare_data(self) -> None:

        self._words = []
        self._prons = []

        for w, p in self._lexicon.items():
            self._words.append(" ".join(list(w)))
            self._prons.append(" ".join(p))

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            num_train, num_val = int(len(self._words)*.9), int(len(self._words)*.1)
            self._train_words, self._val_words = self._words[:num_train], self._words[-num_val:]
            self._train_prons, self._val_prons = self._prons[:num_train], self._prons[-num_val:]

            print (f"# entries: train: {len(self._train_words)}, val: {len(self._val_words)}")

            self._train_dataset = G2PDataset(self._train_words, self._train_prons, self._tokenizer)
            self._val_dataset   = G2PDataset(self._val_words  , self._val_prons  , self._tokenizer)

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
                'graph_idxs' : torch.tensor(padf ('graph_idxs', x_maxlen, self._tokenizer.pad_token_gidx), dtype=torch.long),
                'graphemes'  : [sample['graphemes'] for sample in batch],
                'phone_idxs' : torch.tensor(padf ('phone_idxs', y_maxlen, self._tokenizer.pad_token_pidx), dtype=torch.long),
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
