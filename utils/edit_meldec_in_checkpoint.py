#!/bin/env python3

import glob
import os
import yaml
import argparse
import torch

from zerovox.tts.model import get_meldec, ZeroVox
from torchinfo import summary
from lightning import Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint",
                        help="checkpoint where to add/remove/replace meldec",)

    parser.add_argument("--meldec",
                        type=str,
                        help="Meldec model to use, default: None -> remove meldec, choices: meldec-libritts-multi-band-melgan-v2 or meldec-libritts-hifigan-v1",)

    args = parser.parse_args()

    # with open (os.path.join(args.model, "modelcfg.yaml")) as modelcfgf:
    #     modelcfg = yaml.load(modelcfgf, Loader=yaml.FullLoader)

    # # if isinstance(g2p, str) :
    # #     g2p = G2P(lang, model=g2p)

    # list_of_files = glob.glob(os.path.join(args.model, 'checkpoints/*.ckpt'))
    # checkpoint = max(list_of_files, key=os.path.getctime)

    # print (f"loading checkpoint {checkpoint} ...")

    # model = ZeroVox.load_from_checkpoint(lang=modelcfg['lang'],
    #                                      meldec_model=None,
    #                                      sampling_rate=modelcfg['audio']['sampling_rate'],
    #                                      hop_length=modelcfg['audio']['hop_size'],
    #                                      checkpoint_path=checkpoint,
    #                                      infer_device=torch.device('cpu'),                                                              
    #                                      map_location=torch.device('cpu'),
    #                                      strict=False,
    #                                      verbose=False)
    # summary(model, depth=1)

    if args.meldec:

        meldec = get_meldec(args.meldec, infer_device='cpu')

        # with open(args.meldec, "rb") as f:
        #     meldec = torch.load(f, map_location=torch.device('cpu'))

        # for key in list(meldec['model']['generator']):
        #     # if key.startswith('hifigan.'):
        #     #     del checkpoint['state_dict'][key]
        #     #     continue
        #     print(f"meldec: {key}")

    else:
        meldec = None

    # print(meldec)

    print (f"loading {args.checkpoint} ...")

    with open(args.checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

        #for key in list(checkpoint['state_dict']):
            # if key.startswith('hifigan.'):
            #     del checkpoint['state_dict'][key]
            #     continue
        #    print(key)

    if meldec:

        meldec_weights = meldec.state_dict()
        for key in meldec_weights:
            mdeckey = '_meldec.'+key
            print (f"adding meldec key {mdeckey}")
            checkpoint['state_dict'][mdeckey] = meldec_weights[key]

    else:

        for key in list(checkpoint['state_dict']):
            if key.startswith('_meldec.'):
                print (f"removing {key}")
                del checkpoint['state_dict'][key]


    torch.save(checkpoint, args.checkpoint)
    print (f"{args.checkpoint} written.")

