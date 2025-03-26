import re
import os
from pathlib import Path

import uroman
from nemo_text_processing.text_normalization.normalize import Normalizer

from zerovox.tts.symbols import Symbols

# extend as needed, only these have been tested so far:
SUPPORTED_LANGS = set(['en', 'de'])

class ZeroVoxNormalizer:

    def __init__(self, lang):
        assert lang in SUPPORTED_LANGS
        self._lang = lang
        self._uromanizer = uroman.Uroman()

        cache_dir = Path(os.getenv("CACHED_PATH_ZEROVOX", Path.home() / ".cache" / "zerovox" / "nemo"))
        self._nemo_normalizer = Normalizer(
            input_case='cased',
            lang=lang,
            cache_dir=str(cache_dir / lang)
        )

    @property
    def language (self):
        return self._lang

    def normalize(self, transcript):
        transcript_normalized = self._nemo_normalizer.normalize(transcript)

        transcript_uroman = str(self._uromanizer.romanize_string(transcript_normalized)).lower().strip()

        # Apply existing normalization steps
        transcript_uroman_normalized = self.normalize_uroman(transcript_uroman)

        #print (f"transcript                  : {transcript}")
        #print (f"transcript_normalized       : {transcript_normalized}")
        #print (f"transcript_uroman           : {transcript_uroman}")
        #print (f"transcript_uroman_normalized: {transcript_uroman_normalized}")

        return transcript_uroman, transcript_uroman_normalized

    def normalize_uroman(self, text_uroman):
        text = re.sub("([^a-z' ])", " ", text_uroman)
        text = re.sub(' +', ' ', text)
        return text.strip()


