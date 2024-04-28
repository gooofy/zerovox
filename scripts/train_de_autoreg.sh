#!/bin/bash

python3 utils/train_g2p.py \
            --accelerator=gpu \
            --max_epochs=100 \
            --warmup_epochs=10 \
            --val_epochs=1 \
            --lr=0.0001 \
            --batch-size=512 \
            --out-dir=models/de_autoreg_1 \
            configs/g2p_de_autoreg.yaml lexicon/german_mfa.dict

