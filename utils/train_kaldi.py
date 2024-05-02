#!/bin/env python3

#
# based on:
#
# Kaldi Tutorial by Eleanor Chodroff
# Written: 2015-07-15 | Last updated: 2018-11-13
# https://www.eleanorchodroff.com/tutorial/kaldi/introduction.html
#

from pathlib import Path
import argparse
import yaml
import os
import shutil
import re
import sys
import subprocess

from zerovox.g2p.g2p import G2P


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
    parser.add_argument("configs", type=str, nargs='+', help="path to preprocess.yamls")
    parser.add_argument('--model-name', type=str, default="kaldi_model_2", help="model name")
    parser.add_argument('--cuda', action='store_true', help="enable CUDA")
    parser.add_argument("--num-jobs", type=int, default=12, help="number of jobs, default: 12")

    args = parser.parse_args()

    print ("collecting .yaml files from specified paths...")

    cfgfns = []
    for cfgfn in args.configs:
        if os.path.isdir(cfgfn):
            for cfn in os.listdir(cfgfn):

                _, ext = os.path.splitext(cfn)
                if ext != '.yaml':
                    continue

                cfpath = os.path.join(cfgfn, cfn)
                #print (f"{cfpath} ...")
                cfgfns.append(cfpath)
        else:
            #print (f"{cfgfn} ...")
            cfgfns.append(cfgfn)

    if not cfgfns:
        print ("*** error: no .yaml files found!")
        sys.exit(1)
    else:
        print (f"{len(cfgfns)} .yaml files found.")

    language      = None
    sampling_rate = None

    for cfgfn in cfgfns:
        config = yaml.load(open(cfgfn, "r"), Loader=yaml.FullLoader)

        lang = config['preprocessing']['text']['language']
        sr   = config['preprocessing']['audio']['sampling_rate']

        if not language:
            language = lang
        else:
            if lang != language:
                print (f"inconsistent languages in .yaml files: {lang} vs {language} in {cfgfn}")

        if not sampling_rate:
            sampling_rate = int(sr)
        else:
            if int(sr) != sampling_rate:
                print (f"inconsistent sampling rates in .yaml files: {sr} vs {sampling_rate} in {cfgfn}")

    g2p = G2P(language)

    # workdir = Path(config['path']['preprocessed_path']) / 'work0'
    workdir = Path (f"models/{args.model_name}") / "work"

    print (workdir)

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
    os.makedirs(workdir / 'data' / 'train', mode=0o755)
    os.makedirs(workdir / 'data' / 'lang', mode=0o755)
    os.makedirs(workdir / 'data' / 'local' / 'lang', mode=0o755)

    lex = {}
    oovs = set()

    uttcnt = 0
    speaker = 0

    with open(workdir / 'data' / 'train' / 'text', 'w') as textf:
        with open(workdir / 'data' / 'train' / 'wav.scp', 'w') as wavscpf:
            with open(workdir / 'data' / 'train' / 'utt2spk', 'w') as utt2spkf:

                for cfgfn in cfgfns:
                    print (f"{cfgfn} ...")

                    config = yaml.load(open(cfgfn, "r"), Loader=yaml.FullLoader)
                    rawpath = Path(config['path']['raw_path'])

                    # speaker = config['preprocessing']['text']['speaker']
                    speaker += 1

                    for labfn in os.listdir(rawpath):

                        src, ext = os.path.splitext(labfn)

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

                        uttid = f"{speaker:06}_{uttcnt:09}"
                        uttcnt += 1

                        textf.write (f"{uttid} {' '.join(words)}\n")

                        wavscpf.write (f"{uttid} {str((rawpath / (src+'.wav')).resolve())}\n")
                        utt2spkf.write (f"{uttid} {speaker:06}\n")

    print (f"\n\n*** total: {uttcnt} utterances.\n\n")

    if oovs:
        print (f"*** ERROR: {len(oovs)} OOV()s found")
        for oov in oovs:
            print(oov)
        print ("HINT: use oovtool to generate pronounciations first.")
        sys.exit(1)

    with open(workdir / 'conf' / 'mfcc.conf', 'w') as mfccf:
        mfccf.write(f'--use-energy=false\n--sample-frequency={sampling_rate}\n')

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
        for p in sorted(phoneset):
            nspf.write(f"{p}\n")

    with open(workdir / 'data' / 'local' / 'lang' / 'silence_phones.txt', 'w') as spf:
        spf.write(f"sil\nspn\n")

    with open(workdir / 'data' / 'local' / 'lang' / 'optional_silence.txt', 'w') as osf:
        osf.write(f"sil\n")

    do_cmd( ['utils/prepare_lang.sh', '--position-dependent-phones', 'false', 'data/local/lang', '<oov>', 'data/local/', 'data/lang'], workdir)
    do_cmd( ['utils/fix_data_dir.sh', 'data/train'], workdir )
    do_cmd( ['steps/make_mfcc.sh', '--cmd', 'run.pl', '--nj', str(args.num_jobs), 'data/train', 'exp/make_mfcc/train', 'mfcc'], workdir)
    do_cmd( ['utils/fix_data_dir.sh', 'data/train'], workdir)
    do_cmd( ['steps/compute_cmvn_stats.sh', 'data/train', 'exp/make_mfcc/data/train', 'mfcc'], workdir)
    do_cmd( ['utils/fix_data_dir.sh', 'data/train'], workdir)
    do_cmd( ['utils/subset_data_dir.sh', '--first', 'data/train', '10000', 'data/train_10k'], workdir)
    do_cmd( ['steps/train_mono.sh', '--boost-silence', '1.25', '--nj', str(args.num_jobs), '--cmd', 'run.pl', 'data/train_10k', 'data/lang', 'exp/mono_10k'], workdir)
    do_cmd( ['steps/align_si.sh', '--boost-silence', '1.25', '--nj', str(args.num_jobs), '--cmd', 'run.pl', 'data/train', 'data/lang', 'exp/mono_10k', 'exp/mono_ali'], workdir)
    do_cmd( ['steps/train_deltas.sh', '--boost-silence', '1.25', '--cmd', 'run.pl', '2000', '10000', 'data/train', 'data/lang', 'exp/mono_ali', 'exp/tri1'], workdir)
    do_cmd( ['steps/align_si.sh', '--nj', str(args.num_jobs), '--cmd', 'run.pl', 'data/train', 'data/lang', 'exp/tri1', 'exp/tri1_ali'], workdir)
    do_cmd( ['steps/train_deltas.sh', '--cmd', 'run.pl', '2500', '15000', 'data/train', 'data/lang', 'exp/tri1_ali', 'exp/tri2a'], workdir)
    do_cmd( ['steps/align_si.sh', '--nj', str(args.num_jobs), '--cmd', 'run.pl', '--use-graphs', 'true', 'data/train', 'data/lang', 'exp/tri2a', 'exp/tri2a_ali'], workdir)
    do_cmd( ['steps/train_lda_mllt.sh', '--cmd', 'run.pl', '3500', '20000', 'data/train', 'data/lang', 'exp/tri2a_ali', 'exp/tri3a'], workdir)
    do_cmd( ['steps/align_fmllr.sh', '--nj', str(args.num_jobs), '--cmd', 'run.pl', 'data/train', 'data/lang', 'exp/tri3a', 'exp/tri3a_ali'], workdir)
    do_cmd( ['steps/train_sat.sh', '--cmd', 'run.pl', '4200', '40000', 'data/train', 'data/lang', 'exp/tri3a_ali', 'exp/tri4a'], workdir)  
