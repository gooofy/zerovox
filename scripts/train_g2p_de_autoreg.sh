#!/bin/bash

python3 utils/train_g2p.py \
            --accelerator=gpu \
            --max_epochs=100 \
            --warmup_epochs=10 \
            --val_epochs=1 \
            --lr=0.0001 \
            --batch-size=384 \
            --out-dir=models/g2p_de_autoreg_zamia_1 \
            configs/g2p_de_autoreg_zamia.yaml

