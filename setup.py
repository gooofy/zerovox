from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'zerovox',
  packages=find_packages(
        where='.',
        #include=['zerovox', 'zerovox/g2p'],  # ['*'] by default
        #exclude=['mypackage.tests'],  # empty by default
  ),
  version = '0.2.0',
  description = 'zero-shot realtime TTS system, fully offline, free and open source',
  long_description=long_description,
  author = 'Günter Bartsch',
  author_email = 'guenter@zamia.org',
  url = 'https://github.com/gooofy/zerovox',
  download_url = 'https://github.com/gooofy/zerovox/archive/0.1.1.tar.gz',
  keywords = [
    "text-to-speech",
    "deep-learning",
    "speech",
    "pytorch",
    "tts",
    "speech-synthesis",
    "voice-synthesis",
    "voice-cloning",
    "speaker-encodings",
    "melgan",
    "speaker-encoder",
    "multi-speaker-tts",
    "hifigan",
    "tts-model"
  ],
  classifiers = [],
  install_requires = [
    'torch>=2.1.0',
    'torchinfo>=1.8.0',
    'torchaudio>=2.5.1',
    'nltk>=3.8.1',
    'num2words>=0.5.13',
    'readline>=6.2.4.1',
    'librosa>=0.10.2',
    'lightning>=2.4.0',
    'einops>=0.8.0',
    'sounddevice>=0.4.7',
    'setproctitle>=1.3.3',
    "psutil>=6.0.0",
    "h5py>=3.11.0"
  ],
  license='Apache Software License',
  include_package_data=True,
  entry_points={
    'console_scripts': [
      'zerovox-demo=zerovox.demo:main',
    ],
  },
)
