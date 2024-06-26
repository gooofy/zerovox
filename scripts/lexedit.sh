#!/bin/bash

if [ $# -ne 1 ] ; then
    echo "usage: $0 <word1,word2,...>"
    exit 1
fi

utils/lexedit.py -m models/tts_de_zerovox_base_2 \
    --refaudio=preprocessed_data/de_hui_Karlsson/wavs/dschinnistan_dschinnistan_01_f000093.wav \
    -e $1
