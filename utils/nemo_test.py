#!/bin/env python3

# https://github.com/NVIDIA/NeMo-text-processing/blob/main/tutorials/Text_(Inverse)_Normalization.ipynb

import os
from pathlib import Path
import nemo_text_processing
from nemo_text_processing.text_normalization.normalize import Normalizer

cache_path = Path(os.getenv("CACHED_PATH_ZEROVOX", Path.home() / ".cache" / "zerovox" / "nemo"))

normalizer = Normalizer(input_case='lower_cased', lang='en', cache_dir=str(cache_path / 'en'))

# Examples
text_examples = [
    "123",  # Numbers
    "1,234.56",  # Numbers with commas and decimals
    "$100",  # Currency
    "USD 100", # Currency with code
    "€100",  # Other Currency Symbol
    "January 1st, 2024",  # Dates
    "Jan 1, 2024",  # Abbreviated dates
    "1/1/2024",  # Another date format
    "10:30 AM",  # Time
    "10:30:00", # Time with seconds
    "Mr. Smith",  # Abbreviations (handled somewhat, but often need customization)
    "10%",  # Percentages
    "10 kg", # Units
    "10 km/h", # Units with slashes
    "First", # Ordinals
    "second place", # Ordinals with words
    "10 Downing Street", # Address
    "The quick brown fox jumps over the lazy dog.", # Normal text
]

for text in text_examples:
    normalized_text = normalizer.normalize(text)
    print(f"Original: {text}")
    print(f"Normalized: {normalized_text}")
    print("-" * 20)

normalizer = Normalizer(input_case='lower_cased', lang='de', cache_dir=str(cache_path / 'de'))

text_examples = [
    "123",  # Zahlen
    "1.234,56",  # Zahlen mit Tausendertrennzeichen und Dezimalpunkt
    "€100",  # Währungssymbole
    "Euro 100",  # Währungen mit Namen
    "1. Januar 2024",  # Datumsangaben
    "1.1.2024",  # Andere Datumsformate
    "10:30 Uhr",  # Uhrzeit
    "10:30:00",  # Uhrzeit mit Sekunden
    "Herr Dr. Müller",  # Titel und Abkürzungen
    "10%",  # Prozentangaben
    "10 kg",  # Einheiten
    "10 km/h",  # Einheiten mit Slash
    "1. Übersicht",  # Ordinalzahlen
    "2. Platz",  # Ordinalzahlen mit Nomen
    "10 Downing Street",  # Adressen
    "Der schnelle braune Fuchs springt über den faulen Hund.",  # Normaler Text
    "d.h.",  # Abkürzung
    "z.B.",  # Abkürzung
    "ca.",  # Abkürzung
    "500 €",  # Währung am Ende
    "500 Euro",  # Währung mit Namen am Ende
    "2024-12-25",  # ISO-Datumsformat
    "10.05.2024",  # DD.MM.YYYY
]

for text in text_examples:
    normalized_text = normalizer.normalize(text)
    print(f"Original: {text}")
    print(f"Normalized: {normalized_text}")
    print("-" * 20)

# Customizing for specific cases (more advanced)
# You can customize the behavior by creating your own rules or lexicon files.
# This usually involves working with the underlying WFST grammars, which is more complex.
# For simple cases, you might be able to use regular expressions or string manipulation
# before or after the normalization process.

# Example of Preprocessing to help normalization
# def preprocess_text(text):
#     # Example: Replace some characters before normalization
#     text = text.replace("½", "one half")
#     return text

# custom_text = "I have ½ a pizza."
# preprocessed_custom_text = preprocess_text(custom_text)
# normalized_custom_text = normalizer.normalize(preprocessed_custom_text)
# print(f"Original: {custom_text}")
# print(f"Preprocessed: {preprocessed_custom_text}")
# print(f"Normalized: {normalized_custom_text}")


# For more complex customizations, consult the NeMo documentation regarding
# custom grammars and the WFST builder.  This will require a deeper dive
# into how NeMo's text normalization is implemented.