import re
import os
from pathlib import Path

import uroman
from nemo_text_processing.text_normalization.normalize import Normalizer

from zerovox.tts.symbols import Symbols

_normalizer_cache = {}

def _get_normalizer(lang:str):

    if lang in _normalizer_cache:
        return _normalizer_cache[lang]

    cache_dir = Path(os.getenv("CACHED_PATH_ZEROVOX", Path.home() / ".cache" / "zerovox" / "nemo"))
    nemo_normalizer = Normalizer(
        input_case='cased',
        lang=lang,
        cache_dir=str(cache_dir / lang)
    )

    _normalizer_cache[lang] = uroman.Uroman(), nemo_normalizer

    return _normalizer_cache[lang]

def zerovox_normalize(transcript:str, lang:str):

    uromanizer, nemo_normalizer = _get_normalizer(lang) 

    transcript_normalized = nemo_normalizer.normalize(transcript)

    transcript_uroman = str(uromanizer.romanize_string(transcript_normalized)).lower().strip()

    # apply additional normalization steps

    transcript_uroman_normalized = re.sub("([^a-z' ])", " ", transcript_uroman)
    transcript_uroman_normalized = re.sub(' +', ' ', transcript_uroman_normalized)
    transcript_uroman_normalized = transcript_uroman_normalized.strip()

    #print (f"transcript                  : {transcript}")
    #print (f"transcript_normalized       : {transcript_normalized}")
    #print (f"transcript_uroman           : {transcript_uroman}")
    #print (f"transcript_uroman_normalized: {transcript_uroman_normalized}")

    return transcript_uroman, transcript_uroman_normalized

class ZeroVoxNormalizer:

    def __init__(self, lang):
        self._lang = lang

    @property
    def language (self):
        return self._lang

    def normalize(self, transcript):

        return zerovox_normalize(transcript=transcript, lang=self._lang)

