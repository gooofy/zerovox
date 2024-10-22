import re

from nltk.tokenize import TweetTokenizer

from num2words import num2words

class G2PTokenizer:

    def __init__(self, lang: str):

        self._lang = lang
        self._word_tokenize = TweetTokenizer().tokenize

    def tokenize (self, text:str) -> list[str]:

        # preprocessing

        text = text.lower().encode('latin1', errors='ignore').decode('latin1')

        if self._lang == 'en':
            text = text.replace ('e.g.', 'for example')
            text = text.replace ('i.e.', 'that is')
        elif self._lang == 'de':
            text = text.replace ('z.b.', 'zum beispiel')
            text = text.replace ('d.h.', 'das heißt')
        else:
            raise Exception (f'unsupported language: {self._lang}')

        # tokenization
        tokens = self._word_tokenize(text)

        res = []
        for token in tokens:

            if token != '.':
                token = token.replace('.','') # dlf.de, e.on, etc.

            if token != ':':
                token = token.replace(':','')

            if token != ';':
                token = token.replace(';','')

            # deal with hypens
            if '-' in token:
                subtokens = token.split('-')
                minl = min([len(st) for st in subtokens])
                if minl<3:
                    subtokens = [token]
            else:
                subtokens = [token]

            for st in subtokens:
                if re.search("[0-9]", st):
                    try:
                        st = num2words(st, lang=self._lang)
                    except:
                        pass

                res.extend(st.split(' '))

        return res


