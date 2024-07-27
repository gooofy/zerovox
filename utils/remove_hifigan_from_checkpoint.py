#!/bin/env python3

import glob
import os

import argparse
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='remhifigan', description='remove hifigan params from zerovox model checkpoint')

    parser.add_argument("--model",
                        default=None,
                        required=True,
                        help="Path to model directory",)

    args = parser.parse_args()

    list_of_files = glob.glob(os.path.join(args.model, 'checkpoints/*.ckpt'))
    checkpoint_path = max(list_of_files, key=os.path.getctime)

    no_hifigan_checkpoint_path = os.path.join(args.model, 'checkpoints/best_no_hifigan.ckpt')

    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

        for key in list(checkpoint['state_dict']):
            if key.startswith('hifigan.'):
                del checkpoint['state_dict'][key]
                continue
            print(key)

        torch.save(checkpoint, no_hifigan_checkpoint_path)
        print (f"{no_hifigan_checkpoint_path} written.")

