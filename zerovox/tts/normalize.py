import re

from num2words import num2words
import uroman
from zerovox.tts.symbols import Symbols

# extend as needed, only these have been tested so far:
SUPPORTED_LANGS = set(['en', 'de'])

class Normalizer:

    def __init__(self, lang):
        assert lang in SUPPORTED_LANGS
        self._lang = lang
        self._uromanizer = uroman.Uroman()

    @property
    def language (self):
        return self._lang

    def spell_out_numbers(self, text):
        """
        Spells out numbers within a string using num2words.

        Args:
            text: The input string.

        Returns:
            The string with numbers spelled out, or the original string if no numbers are found.
            Returns an error message if num2words raises an exception (like for very large numbers).
        """
        try:
            def replace_number(match):
                num_str = match.group(0)
                try:
                    num = int(num_str)
                    return num2words(num, lang=self._lang)
                except ValueError:  # Handle floats
                    try:
                        num = float(num_str)
                        if num.is_integer(): #check if it is an integer represented as float
                            return num2words(int(num), lang=self._lang)
                        else:
                            parts = str(num).split('.')
                            integer_part = num2words(int(parts[0]), lang=self._lang)
                            decimal_part = parts[1]
                            decimal_as_int = int(decimal_part)
                            decimal_spelled = num2words(decimal_as_int, lang=self._lang)
                            return f"{integer_part} point {decimal_spelled}"
                    except ValueError:
                        return num_str  # Return original if not a valid number
                except OverflowError: #handle numbers too big for int
                    return f"Number too large to spell out"
                except Exception as e:
                    return f"Error spelling out number: {e}"
                    
            pattern = r"\b\d+(\.\d+)?\b"  # Matches whole numbers or decimals
            new_text = re.sub(pattern, replace_number, text)

            pattern = r"\d+"  # Matches remaining digits or numbers
            new_text = re.sub(pattern, replace_number, new_text)

            return new_text
        except Exception as e:
            return f"An unexpected error occurred: {e}"


    def normalize_uroman(self, text_uroman):
        text = re.sub("([^a-z' ])", " ", text_uroman)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def normalize (self, transcript):

        transcript_uroman = transcript.replace("’", "'")
        transcript_uroman = self.spell_out_numbers(transcript_uroman)
        transcript_uroman = str(self._uromanizer.romanize_string(transcript_uroman)).lower().strip()

        transcript_normalized = self.normalize_uroman(transcript_uroman)

        return transcript_uroman, transcript_normalized
    
