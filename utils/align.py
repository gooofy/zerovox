from pathlib import Path
import argparse
import yaml
import os
import shutil
import re
import sys

from zerovox.g2p.g2p import G2P

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    parser.add_argument('--cuda', action='store_true', help="enable CUDA")
    parser.add_argument("--oovs", type=str, help="name of oovs.dict file to store pronounciation predictions")

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    language = config['preprocessing']['text']['language']

    g2p = G2P(language)

    rawpath = Path(config['path']['raw_path'])

    workdir = Path(config['path']['preprocessed_path']) / 'work0'

    print (workdir)

    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, mode=0o755)

    lex = {}
    oovs = set()

    with open(workdir / 'text', 'w') as textf:

        for labfn in os.listdir(rawpath):

            uttid, ext = os.path.splitext(labfn)

            if ext != '.lab':
                continue

            with open (rawpath / labfn) as labf:
                text = labf.readline()

            tokens = g2p.tokenize(text)

            words = []
            for token in tokens:
                if re.search("[a-züöäß]", token) is None:
                    continue
                words.append(token)

                if (token not in lex) and (token not in oovs):

                    pron = g2p.lookup(token)
                    if pron:
                        lex[token] = pron
                    else:
                        oovs.add(token)

            textf.write (f"{uttid} {' '.join(words)}\n")

    if oovs:

        oovs = sorted(list(oovs))

        print (f"*** ERROR: {len(oovs)} OOVs found: ")
        print (",".join(oovs))

        if args.oovs:
            print ("generating pronounciations...")
            with open (args.oovs, 'w') as oovf:
                for oov in oovs:
                    phonemes, _ = g2p.predict(oov)
                    l=f"{oov}\t{' '.join(phonemes)}"
                    print("    " + l)
                    oovf.write(l+"\n")
        sys.exit(1)

