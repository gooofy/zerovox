#!/bin/env python3

# https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html

#python normalize.py --language de --text "CDU und CSU fallen im Vergleich zum Befragungszeitraum 11. bis 17. Februar um einen Prozentpunkt auf 29 Prozent Zustimmung. Auch die SPD steht mit einem Punkt weniger bei nun 15 Prozent. Zulegen kann die AfD um einen Punkt auf 21 Prozent. Sie würde damit mit Abstand zweitstärkste Kraft im nächsten Bundestag. Die Linke, die lange Zeit weit weg von der Fünfprozenthürde war, steht mit einem Punkt mehr bei 8 Prozent. Die FDP liegt wie im vorherigen Befragungszeitraum bei 5 Prozent. Sie wird demnach den ganzen Wahlabend bangen müssen, ob es wirklich für den Wiedereinzug reicht."



import nemo.text_processing.text_normalization as tn

# Initialize the normalizer
normalizer = tn.TextNormalizer(lang='en')  # You can specify other languages like 'es', 'fr', etc. if needed.

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


# Customizing for specific cases (more advanced)
# You can customize the behavior by creating your own rules or lexicon files.
# This usually involves working with the underlying WFST grammars, which is more complex.
# For simple cases, you might be able to use regular expressions or string manipulation
# before or after the normalization process.

# Example of Preprocessing to help normalization
def preprocess_text(text):
    # Example: Replace some characters before normalization
    text = text.replace("½", "one half")
    return text

custom_text = "I have ½ a pizza."
preprocessed_custom_text = preprocess_text(custom_text)
normalized_custom_text = normalizer.normalize(preprocessed_custom_text)
print(f"Original: {custom_text}")
print(f"Preprocessed: {preprocessed_custom_text}")
print(f"Normalized: {normalized_custom_text}")


# For more complex customizations, consult the NeMo documentation regarding
# custom grammars and the WFST builder.  This will require a deeper dive
# into how NeMo's text normalization is implemented.