ZeroVOX: A zero-shot realtime TTS system, fully offline, free and open source
=============================================================================

ZeroVOX is a text-to-speech (TTS) system built for real-time and embedded use.

ZeroVox runs entirely offline, ensuring privacy and independence from cloud services. It's completely free and open source, inviting community contributions and suggestions.

Modeled after FastSpeech2, ZeroVOX goes a step further with zero-shot speaker cloning, utilizing effective speaker embedding. The system supports both English and German speech generation from a single model, trained on an extensive dataset. ZeroVOX is phoneme-based, leveraging pronunciation dictionaries to ensure accurate word articulation, utilizing the CMU dictionary for English and a custom dictionary for German from the ZamiaSpeech project where also the phoneme set used originates from.

ZeroVOX can serve as a TTS backend for LLMs, enabling real-time interactions, and as an easy-to-install TTS system for home automation systems like Home Assistant. Since it is non-autoregressive like FastSpeech2 its output is generally easy to control and predictable.

License: ZeroVOX is Apache 2 licensed with many parts leveraged from other projects (see credits section below) under MIT license.

Demo
====

Please Note: model is still in alpha stage and still training.

https://huggingface.co/spaces/goooofy/zerovox-demo

Audio Corpus Stats
==================

Current ZeroVOX training corpus stats:

    german  audio corpus: 16679 speakers, 475.3 hours audio
    english audio corpus: 19899 speakers, 358.7 hours audio

ZeroVOX Model Training
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

    utils/oovtool.py -a -m zerovox-g2p-autoreg-zamia-de configs/corpora/cv_de_100

(4/5) align:

    utils/align.py --kaldi-model=tts_de_kaldi_zamia_4 configs/corpora/cv_de_100

(5/5) preprocess:

    utils/preprocess.py configs/corpora/cv_de_100

TTS Model Training
------------------

    utils/train_tts.py \
        --head=2 --reduction=1 --expansion=2 --kernel-size=5 --n-blocks=3 --block-depth=3 \
        --accelerator=gpu --threads=24 --batch-size=32 --val_epochs=8 \
        --infer-device=cpu \
        --lr=0.0001 --warmup_epochs=25 \
        --hifigan-checkpoint=VCTK_V2 \
        --out-folder=models/tts_de_zerovox_base_1 \
        configs/corpora/cv_de_100 \
        configs/corpora/de_hui/de_hui_*.yaml \
        configs/corpora/de_thorsten.yaml

Kaldi Accoustic Model Training
==============================

    utils/train_kaldi.py --model-name=tts_de_kaldi_zamia_4 --num-jobs=12 configs/corpora/cv_de_100

G2P Model Training
==================

run training:

    scripts/train_g2p_de_autoreg.sh

Credits
=======

The training setup is originally based on Efficientspeech by Rowel Atienza

https://github.com/roatienza/efficientspeech

    @inproceedings{atienza2023efficientspeech,
      title={EfficientSpeech: An On-Device Text to Speech Model},
      author={Atienza, Rowel},
      booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={1--5},
      year={2023},
      organization={IEEE}
    }

The FastSpeech2 encoder and decoder is borrowed (under MIT license) from Chung-Ming Chien's implementation of FastSpeech2

https://github.com/ming024/FastSpeech2


    @misc{ren2022fastspeech2fasthighquality,
        title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech}, 
        author={Yi Ren and Chenxu Hu and Xu Tan and Tao Qin and Sheng Zhao and Zhou Zhao and Tie-Yan Liu},
        year={2022},
        eprint={2006.04558},
        archivePrefix={arXiv},
        primaryClass={eess.AS},
        url={https://arxiv.org/abs/2006.04558}, 
    }

The StyleTTS encoder is borrowd (under MIT license) from Aaron (Yinghao) Li's implementation of StyleTTS:

https://github.com/yl4579/StyleTTS

    @misc{li2023stylettsstylebasedgenerativemodel,
        title={StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis}, 
        author={Yinghao Aaron Li and Cong Han and Nima Mesgarani},
        year={2023},
        eprint={2205.15439},
        archivePrefix={arXiv},
        primaryClass={eess.AS},
        url={https://arxiv.org/abs/2205.15439}, 
    }

The MEL decoder implementation is borrowed (under MIT license) from Tomoki Hayashi's ParallelWaveGAN project:

https://github.com/kan-bayashi/ParallelWaveGAN

The G2P transformer models are based on DeepPhonemizer by Axel Springer News Media & Tech GmbH & Co. KG - Ideas Engineering (MIT license)

https://github.com/as-ideas/DeepPhonemizer

    @inproceedings{Yolchuyeva_2019, series={interspeech_2019},
    title={Transformer Based Grapheme-to-Phoneme Conversion},
    url={http://dx.doi.org/10.21437/Interspeech.2019-1954},
    DOI={10.21437/interspeech.2019-1954},
    booktitle={Interspeech 2019},
    publisher={ISCA},
    author={Yolchuyeva, Sevinj and Németh, Géza and Gyires-Tóth, Bálint},
    year={2019},
    month=sep, pages={2095–2099},
    collection={interspeech_2019} }

The ZeroShot ResNet based speaker encoding is borrowed (under MIT license) from voxceleb_trainer by Clova AI Research

https://github.com/clovaai/voxceleb_trainer

    @inproceedings{chung2020in,
    title={In defence of metric learning for speaker recognition},
    author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
    booktitle={Proc. Interspeech},
    year={2020}
    }

    @inproceedings{he2016deep,
    title={Deep residual learning for image recognition},
    author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    pages={770--778},
    year={2016}
    }

Speaker Conditional Layer Normalization (SCLN) which is borrowed (under MIT license) from

https://github.com/keonlee9420/Cross-Speaker-Emotion-Transfer
by Keon Lee

    @misc{wu2021crossspeakeremotiontransferbased,
        title={Cross-speaker Emotion Transfer Based on Speaker Condition Layer Normalization and Semi-Supervised Training in Text-To-Speech}, 
        author={Pengfei Wu and Junjie Pan and Chenchang Xu and Junhui Zhang and Lin Wu and Xiang Yin and Zejun Ma},
        year={2021},
        eprint={2110.04153},
        archivePrefix={arXiv},
        primaryClass={eess.AS},
        url={https://arxiv.org/abs/2110.04153}, 
    }

