Zerovox Model Training
======================

Data Preparation
----------------

(1/n) prepare corpus yamls:

    pushd configs/corpora/cv_de_100
    ./gen_cv.sh
    popd

(2/n) prepare alignment:

    utils/prepare_align.py configs/corpora/cv_de_100

(3/n) OOVs:

    utils/oovtool.py configs/corpora/cv_de_100

(4/n) align:

check for oovs:

    for cfg in configs/corpora/cv_de_100/*.yaml ; do utils/align.py --oovs oovs_`basename $cfg .yaml`.txt $cfg ; done


Kaldi Accoustic Model Training
==============================



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
