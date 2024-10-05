from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'zerovox',
  packages = ['zerovox'], # this must be the same as the name above
  version = '0.0.1',
  description = 'zero-shot TTS system for realtime/embedded use',
  long_description=long_description,
  author = 'Günter Bartsch',
  author_email = 'guenter@zamia.org',
  url = 'https://github.com/gooofy/zerovox',
  download_url = 'https://github.com/gooofy/zerovox/archive/0.0.1.tar.gz',
  keywords = ['g2p','tts','artificial-intelligence','deeplearning'],
  classifiers = [],
  install_requires = [
    'torch>=2.1.0',
    'torchinfo>=1.8.0',
    'nltk>=3.8.1',
    'num2words>=0.5.13',
    'readline>=6.2.4.1',
    'librosa>=0.10.2',
    'lightning>=2.4.0',
    'einops>=0.8.0',
    'sounddevice>=0.4.7',
  ],
  license='Apache Software License',
  include_package_data=True
)

