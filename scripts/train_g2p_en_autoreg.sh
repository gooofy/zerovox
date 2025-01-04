#!/bin/bash

python3 utils/train_g2p.py \
            --accelerator=gpu \
            --max_epochs=100 \
            --warmup_epochs=10 \
            --val_epochs=1 \
            --lr=0.0001 \
            --batch-size=256 \
            --lang="en" \
            --out-dir=models/g2p_en_autoreg_zamia_1 \
            configs/g2p_en_autoreg_zamia.yaml

