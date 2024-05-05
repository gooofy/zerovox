Zerovox Model Training
======================

Data Preparation
----------------

(1/5) prepare corpus yamls:

    pushd configs/corpora/cv_de_100
    ./gen_cv.sh
    popd

(2/5) prepare alignment:

    utils/prepare_align.py configs/corpora/cv_de_100

(3/5) OOVs:

    utils/oovtool.py configs/corpora/cv_de_100

(4/5) align:

    utils/align.py --kaldi-model=tts_de_kaldi_zamia_1 configs/corpora/cv_de_100

(5/5) preprocess:

    tils/preprocess.py configs/corpora/cv_de_100

TTS Model Training
------------------

    utils/train_tts.py \
        --head=2 --reduction=1 --expansion=2 --kernel-size=5 --n-blocks=3 --block-depth=3 \
        --accelerator=gpu --threads=24 --batch-size=64 --val_epochs=1 \
        --infer-device=cpu \
        --lr=0.0001 --warmup_epochs=25 \
        --hifigan-checkpoint=VCTK_V2 \
        --out-folder=models/tts_de_zerovox_base_1 \
        configs/corpora/cv_de_100 \
        configs/corpora/de_hui/de_hui_*.yaml \
        configs/corpora/de_thorsten.yaml

Kaldi Accoustic Model Training
==============================

    utils/train_kaldi.py --model-name=tts_de_kaldi_zamia_1 --num-jobs=12 configs/corpora/cv_de_100

G2P Model Training
==================

run training:

    scripts/train_de_autoreg.sh

Credits
=======

Based on Efficientspeech by Rowel Atienza

https://github.com/roatienza/efficientspeech

    @inproceedings{atienza2023efficientspeech,
      title={EfficientSpeech: An On-Device Text to Speech Model},
      author={Atienza, Rowel},
      booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={1--5},
      year={2023},
      organization={IEEE}
    }

The G2P transformer models are based on DeepPhonemizer by Axel Springer News Media & Tech GmbH & Co. KG - Ideas Engineering

https://github.com/as-ideas/DeepPhonemizer
