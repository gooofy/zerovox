#!/bin/env python3

from pathlib import Path
import argparse
import yaml
import os
import shutil
import re
import sys
import subprocess

from zerovox.g2p.g2p import G2P

SAMPLE_RATE=22050

def do_cmd(cmd:list[str], workdir: os.PathLike):

    print()
    print (" ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)

if __name__ == "__main__":

    if 'KALDI_ROOT' not in os.environ:
        print ("KALDI_ROOT environment variable not set!")
        sys.exit(1)

    kaldi_root = Path(os.environ['KALDI_ROOT'])
    if not os.path.exists(kaldi_root):
        print (f"KALDI_ROOT path {kaldi_root} does not exist!")
        sys.exit(2)

    print (f"using kaldi installation at KALDI_ROOT={kaldi_root}")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    #parser.add_argument('--cuda', action='store_true', help="enable CUDA")
    #parser.add_argument("--oovs", type=str, help="name of oovs.dict file to store pronounciation predictions")
    parser.add_argument("--num-jobs", type=int, default=12, help="number of jobs, default: 12")
    parser.add_argument("--kaldi-model", type=str, default='kaldi_model_1', help="kaldi model to use for alignment, default: kaldi_model_1")

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    language = config['preprocessing']['text']['language']

    g2p = G2P(language)

    rawpath = Path(config['path']['raw_path'])

    workdir = Path(config['path']['preprocessed_path']) / 'work0'
    print (workdir)

    kaldi_model_path = Path ('models') / args.kaldi_model / 'work' / 'exp' / 'tri4a'

    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, mode=0o755)

    os.symlink (kaldi_root / 'egs' / 'wsj' / 's5' / 'steps', workdir / 'steps')
    os.symlink (kaldi_root / 'egs' / 'wsj' / 's5' / 'utils', workdir / 'utils')
    os.symlink (kaldi_root / 'src', workdir / 'src')

    with open (os.open(path=workdir / 'path.sh', flags=(os.O_WRONLY | os.O_CREAT  | os.O_TRUNC), mode=0o755), 'w') as pathf:
        pathf.write("""
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh\n
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH\n
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1\n
. $KALDI_ROOT/tools/config/common_path.sh\n
export LC_ALL=C\n
export PYTHONUNBUFFERED=1\n
                   """)

    os.makedirs(workdir / 'exp', mode=0o755)
    os.makedirs(workdir / 'conf', mode=0o755)
    os.makedirs(workdir / 'data' / 'alignme', mode=0o755)
    os.makedirs(workdir / 'data' / 'lang', mode=0o755)
    os.makedirs(workdir / 'data' / 'local' / 'lang', mode=0o755)

    lex = {}
    oovs = set()

    with open(workdir / 'data' / 'alignme' / 'text', 'w') as textf:
        with open(workdir / 'data' / 'alignme' / 'wav.scp', 'w') as wavscpf:
            with open(workdir / 'data' / 'alignme' / 'utt2spk', 'w') as utt2spkf:

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

                    wavscpf.write (f"{uttid} {str((rawpath / (uttid+'.wav')).resolve())}\n")
                    utt2spkf.write (f"{uttid} 42\n")

    if oovs:
        print (f"*** ERROR: {len(oovs)} OOVs found - use oovtool to generate pronounciations first.")
        sys.exit(1)

    with open(workdir / 'conf' / 'mfcc.conf', 'w') as mfccf:
        mfccf.write(f'--use-energy=false\n--sample-frequency={SAMPLE_RATE}\n')

    os.makedirs(workdir / 'lang', mode=0o755)

    phoneset = set()

    with open(workdir / 'data' / 'local' / 'lang' / 'lexicon.txt', 'w') as lexf:

        lexf.write('<oov> spn\n')
        for word in sorted(lex):
            phones = lex[word]
            lexf.write(f"{word} {' '.join(phones)}\n")
            for p in phones:
                phoneset.add(p)

    with open(workdir / 'data' / 'local' / 'lang' / 'nonsilence_phones.txt', 'w') as nspf:
        #for p in sorted(phoneset):
        for p in g2p.phonemes:
            nspf.write(f"{p}\n")

    with open(workdir / 'data' / 'local' / 'lang' / 'silence_phones.txt', 'w') as spf:
        spf.write(f"sil\nspn\n")

    with open(workdir / 'data' / 'local' / 'lang' / 'optional_silence.txt', 'w') as osf:
        osf.write(f"sil\n")

    do_cmd( ['utils/prepare_lang.sh', '--position-dependent-phones', 'false', 'data/local/lang', '<oov>', 'data/local/', 'data/lang'], workdir)
    do_cmd( ['utils/fix_data_dir.sh', 'data/alignme'], workdir )
    do_cmd( ['steps/make_mfcc.sh', '--cmd', 'run.pl', '--nj', str(args.num_jobs), 'data/alignme', 'exp/make_mfcc/alignme', 'mfcc'], workdir)
    do_cmd( ['utils/fix_data_dir.sh', 'data/alignme'], workdir)
    do_cmd( ['steps/compute_cmvn_stats.sh', 'data/alignme', 'exp/make_mfcc/data/alignme', 'mfcc'], workdir)
    do_cmd( ['utils/fix_data_dir.sh', 'data/alignme'], workdir)

    do_cmd( ['steps/align_si.sh', '--nj', '1', '--cmd', 'run.pl', 'data/alignme', 'data/lang', kaldi_model_path.resolve(), 'exp/tri4a_alignme'], workdir)
    # FIXME: src/bin/ali-to-phones --ctm-output ../../../models/kaldi_model_1/work/exp/tri4a/final.mdl ark:"gunzip -c exp/tri4a_alignme/ali.1.gz|" -> foo.ctm
