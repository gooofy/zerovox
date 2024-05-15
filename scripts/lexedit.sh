#!/bin/bash

if [ $# -ne 1 ] ; then
    echo "usage: $0 <word1,word2,...>"
    exit 1
fi

utils/lexedit.py -m models/tts_de_zerovox_medium_1 \
    --refaudio=preprocessed_data/de_hui_Karlsson/wavs/dschinnistan_dschinnistan_02_f000023.wav \
    -e $1