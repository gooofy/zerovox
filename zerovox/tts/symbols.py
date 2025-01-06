
class Symbols:

    NO_PUNCT = '_NP_'

    def __init__(self, phones, puncts):

        self._phonemap  = {}
        self._phonemapr = {}
        idx = 0
        for p in phones:
            self._phonemap[p] = idx
            self._phonemapr[idx] = p
            idx += 1

        self._punctmap  = { Symbols.NO_PUNCT : 0 }
        self._punctmapr = { 0 : Symbols.NO_PUNCT }
        idx = 1
        for p in puncts:
            self._punctmap[p] = idx
            self._punctmapr[idx] = p
            idx += 1

    def is_phone(self, p):
        return p in self._phonemap

    def encode_phone(self, phone):
        return self._phonemap[phone]

    def decode_phone(self, phone):
        return self._phonemapr[phone]

    @property
    def num_phones(self):
        return len(self._phonemap)
    
    def is_punct(self, p):
        return p in self._punctmap

    def encode_punct(self, punct):
        return self._punctmap[punct]

    def decode_punct(self, punct):
        return self._punctmapr[punct]

    @property
    def num_puncts(self):
        return len(self._punctmap)
    
