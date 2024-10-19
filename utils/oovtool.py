#!/bin/env python3

import argparse
import yaml
import os
import re

from tqdm import tqdm

from zerovox.g2p.g2p import G2P
from zerovox.g2p.g2p import DEFAULT_G2P_MODEL_NAME_DE

def gather_jobs_from_config(configfn: os.PathLike):

    config = yaml.load(open(configfn, "r"), Loader=yaml.FullLoader)

    out_dir = config["path"]["raw_path"]
    language = config["preprocessing"]["text"]["language"]

    jobs = []

    for labfn in os.listdir(out_dir):

        _, ext = os.path.splitext(labfn)
        if ext != '.lab':
            continue

        labpath = os.path.join(out_dir, labfn)

        with open(labpath, 'r') as labf:
            jobs.append(labf.readline().strip())

    return jobs, language

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs='+', help="path[s] to .yaml config file[s] or directorie[s]")
    parser.add_argument("-o", "--oovs", type=str, default="oovs.dict", help="name of oovs.dict file to store pronounciation predictions, default: oovs.dict")
    parser.add_argument("-m", "--g2p-model",
                        default=DEFAULT_G2P_MODEL_NAME_DE,
                        type=str,
                        help=f"G2P model, default={DEFAULT_G2P_MODEL_NAME_DE}",)                     
    parser.add_argument('-a', '--add', action='store_true',  help="auto-add generated entries to lexicon")
    args = parser.parse_args()

    jobs = []
    language = None

    for configfn in args.configs:

        if os.path.isdir(configfn):

            for cfgfn in os.listdir(configfn):

                _, ext = os.path.splitext(cfgfn)
                if ext != '.yaml':
                    continue

                j, lang = gather_jobs_from_config (os.path.join(configfn, cfgfn))
                jobs.extend(j)

                if not language:
                    language = lang
                else:
                    assert lang == language

        else:
            _, ext = os.path.splitext(configfn)
            if ext != '.yaml':
                continue

            j, lang = gather_jobs_from_config (configfn)
            jobs.extend(j)

            if not language:
                language = lang
            else:
                assert lang == language

    print(f"gathered {len(jobs)} jobs. language is {language}")

    g2p = G2P(language, model=args.g2p_model)
    
    oovs = set()

    for job in tqdm (jobs):

        tokens = g2p.tokenize(job)

        for token in tokens:
            if re.search("[a-züöäß]", token) is None:
                continue

            pron = g2p.lookup(token)

            if (not pron) and (token not in oovs):
                print (f"OOV found: {token}")
                print (f"utterance: {job}")
                oovs.add(token)

        # print(tokens)

    if oovs:
        print (f"{len(oovs)} oovs found")
        with open(args.oovs, 'w') as oovsf:
            oovs = sorted(oovs)
            lex = g2p.lex
            for oov in tqdm(oovs, desc="g2p"):
                pron, _ = g2p.predict(oov)
                oovsf.write (f"{oov}\t{' '.join(pron)}\n")
                if args.add:
                    lex[oov] = pron
        print (f"{args.oovs} written")
        if args.add:
            lex.save()
            print ("entries were added to the lexicon.")

    else:
        print ("*** no OOVs found, all is well ***")
