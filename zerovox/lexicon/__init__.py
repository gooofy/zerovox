import os
import yaml
from tqdm import tqdm

class Lexicon:

    def __init__(self, path: os.PathLike, verbose: bool = True, load_dicts: bool = True):

        self._verbose = verbose

        self._cfg = yaml.load(open(os.path.join(path, 'cfg.yaml'), "r"), Loader=yaml.FullLoader)

        # symbols:

        #   graphemes: 'abcdefghijklmnopqrstuvwxyzäöüß0123456789-''áãåçèéêëíñóôøúûāćđğıőśşżžșțọứàʿ'
        #   phonemes: ['ts', 'ə', 'iː', 'oː', 'pf', 'aj', 'd', 'tʃ', 'm', 'œ', 'z', 'ɛ', 'ɲ', 't', 'ɟ', 'n̩', 'b', 'ɪ', 'kʰ', 'h', 'eː', 'ɔ', 'f', 'v', 'l̩', 'n', 'x', 'yː', 'p', 'c', 'aː', 'ç', 'uː', 'ʃ', 'øː', 'a', 'l', 'j', 'ɔʏ', 'cʰ', 'aw', 'ŋ', 'ɐ', 'ʊ', 'pʰ', 'ʁ', 's', 'ʏ', 'ɡ', 'tʰ', 'k', 'm̩']

        # paths:

        #   base: 'german_mfa.dict'
        #   user: 'german_mfa_add.dict'

        self._graphemes = [*self._cfg['symbols']['graphemes']]
        self._phonemes  = self._cfg['symbols']['phonemes']

        self._base_path = os.path.join(path, self._cfg['paths']['base'])
        self._user_path = os.path.join(path, self._cfg['paths']['user'])

        if load_dicts:
            self._base_lex = self._load_dict(self._base_path)
            self._user_lex = self._load_dict(self._user_path)

    @property
    def graphemes(self) -> list[str]:
        return self._graphemes

    @property
    def phonemes(self) -> list[str]:
        return self._phonemes

    def __contains__(self, graph):
        return (graph in self._base_lex) or (graph in self._user_lex)
    
    def __getitem__(self, graph):
        if graph in self._user_lex:
            return self._user_lex[graph]
        return self._base_lex[graph]

    def __setitem__(self, graph, phonemes):
        self._user_lex[graph] = phonemes

    def __merge(self):
        lex = dict(self._base_lex)
        for graph, phones in self._user_lex.items():
            lex[graph] = phones
        return lex

    def __iter__(self):
        return self.__merge().__iter__()

    def items(self):
        return self.__merge().items()

    def _load_dict(self, path: os.PathLike) -> dict[str, str]:

        lex = {}
        cnt = 0
        if self._verbose:
            with open (path, 'r') as lexf:
                for _ in lexf:
                    cnt += 1

        graphemes = set(self._graphemes)
        phonemes = set(self._phonemes)
        extra_graphemes = set()

        with open (path, 'r') as lexf:

            if self._verbose:
                lexiter = tqdm(lexf, total=cnt, desc='load lexicon')
            else:
                lexiter = lexf

            for line in lexiter:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                graph = parts[0]
                if graph in lex:
                    continue

                phones = parts[1].split(' ')
                valid = True
                for c in graph:
                    if not c in graphemes:
                        if self._verbose:
                            print (f"Warning: invalid grapheme {c} detected in {graph} : {''.join(phones)}")
                        valid = False
                        extra_graphemes.add(c)

                if not valid:
                    continue

                for p in phones:
                    if not p in phonemes:
                        raise Exception (f"Invalid phoneme {p} detected in {graph} : {''.join(phones)}")

                lex[graph] = phones

        if self._verbose and extra_graphemes:
            print (f"extra graphemes detected: {sorted(extra_graphemes)}")

        return lex

    def save(self):
        with open (self._user_path, 'w') as lexf:
            for graph in sorted(self._user_lex):
                lexf.write(f"{graph}\t{' '.join(self._user_lex[graph])}\n")

    def verify_symbols(self, graphemes: list[str], phonemes: list[str]):

        g1 = set(self._graphemes)
        p1 = set(self._phonemes)

        g2 = set(graphemes)
        p2 = set(phonemes)

        assert (g1 == g2) and (p1 == p2)


    @classmethod
    def load(cls, lang: str, load_dicts: bool = True):
        return Lexicon(os.path.join(os.path.dirname(__file__), lang), load_dicts=load_dicts)
    

