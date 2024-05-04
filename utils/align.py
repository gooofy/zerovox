#!/bin/env python3

from pathlib import Path
import argparse
import yaml
import os
import shutil
import re
import sys
import subprocess

from zerovox.g2p.g2p import G2PTokenizer
from zerovox.lexicon import Lexicon

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
    parser.add_argument("--num-jobs", type=int, default=12, help="number of jobs, default: 12")
    parser.add_argument("--kaldi-model", type=str, default='kaldi_model_1', help="kaldi model to use for alignment, default: kaldi_model_1")

    args = parser.parse_args()

    phonemap = {}
    # kaldi_model_path = Path ('models') / args.kaldi_model / 'work' / 'exp' / 'tri4a'
    kaldi_model_path = Path ('models') / args.kaldi_model / 'work' 

    with open (kaldi_model_path / 'exp' / 'tri4a'/ 'phones.txt') as phonesf:

        for line in phonesf:
            parts = line.strip().split(' ')
            assert(len(parts)==2)
            phonemap[int(parts[1])] = parts[0]

    print (phonemap)

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
    sys.stdout.flush()

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

    lexicon = Lexicon.load(language)
    tokenizer = G2PTokenizer(language)

    for cfgfn in cfgfns:

        print()
        print ( "**********************************************************")
        print ( "**                                                      **")
        print (f"** {cfgfn:53}**")
        print ( "**                                                      **")
        print ( "**********************************************************")

        sys.stdout.flush()

        config = yaml.load(open(cfgfn, "r"), Loader=yaml.FullLoader)

        rawpath = Path(config['path']['raw_path'])

        workdir = Path(config['path']['preprocessed_path']) / 'work0'
        print (workdir)

        aligndir = Path(config['path']['preprocessed_path']) / 'align'
        shutil.rmtree(aligndir, ignore_errors=True)
        os.makedirs(aligndir, mode=0o755)

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
        os.makedirs(workdir / 'lang', mode=0o755)

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

                        tokens = tokenizer.tokenize(text)

                        words = []
                        for token in tokens:
                            if re.search("[a-züöäß]", token) is None:
                                continue
                            words.append(token)

                            if (token not in lex) and (token not in oovs):
                                if token in lexicon:
                                    lex[token] = lexicon[token]
                                else:
                                    oovs.add(token)

                        textf.write (f"{uttid} {' '.join(words)}\n")

                        wavscpf.write (f"{uttid} {str((rawpath / (uttid+'.wav')).resolve())}\n")
                        utt2spkf.write (f"{uttid} 42\n")

        if oovs:
            print (f"*** ERROR: {len(oovs)} OOVs found - use oovtool to generate pronounciations first.")
            sys.exit(1)

        with open(workdir / 'conf' / 'mfcc.conf', 'w') as mfccf:
            mfccf.write(f'--use-energy=false\n--sample-frequency={sampling_rate}\n')

        ## for fn in ['lexicon.txt', 'lexiconp.txt', 'nonsilence_phones.txt', 'silence_phones.txt', 'optional_silence.txt']:
        ##     shutil.copyfile(Path('models') / args.kaldi_model / 'work' / 'data' / 'local' / 'lang' / fn, workdir / 'data' / 'local' / 'lang' / fn)

        # phoneset = set()

        # with open( workdir / 'data' / 'local' / 'lang' / 'lexicon.txt', 'w') as lexf:

        #     lexf.write('<oov> spn\n')
        #     for word in sorted(lex):
        #         phones = lex[word]
        #         lexf.write(f"{word} {' '.join(phones)}\n")
        #         for p in phones:
        #             phoneset.add(p)

        # with open(workdir / 'data' / 'local' / 'lang' / 'nonsilence_phones.txt', 'w') as nspf:
        #     for p in sorted(lexicon.phonemes):
        #         nspf.write(f"{p}\n")

        # with open(workdir / 'data' / 'local' / 'lang' / 'silence_phones.txt', 'w') as spf:
        #     spf.write(f"sil\nspn\n")

        # with open(workdir / 'data' / 'local' / 'lang' / 'optional_silence.txt', 'w') as osf:
        #     osf.write(f"sil\n")

        # do_cmd( ['utils/prepare_lang.sh', '--position-dependent-phones', 'false', 'data/local/lang', '<oov>', 'data/local/', 'data/lang'], workdir)
        do_cmd( ['utils/fix_data_dir.sh', 'data/alignme'], workdir )
        do_cmd( ['steps/make_mfcc.sh', '--cmd', 'run.pl', '--nj', str(args.num_jobs), 'data/alignme', 'exp/make_mfcc/alignme', 'mfcc'], workdir)
        # do_cmd( ['utils/fix_data_dir.sh', 'data/alignme'], workdir)
        do_cmd( ['steps/compute_cmvn_stats.sh', 'data/alignme', 'exp/make_mfcc/data/alignme', 'mfcc'], workdir)
        # do_cmd( ['utils/fix_data_dir.sh', 'data/alignme'], workdir)

        # do_cmd( ['steps/align_si.sh', '--nj', '1', '--cmd', 'run.pl', 'data/alignme', 'data/lang', str((kaldi_model_path / 'exp' / 'tri4a').resolve()), 'exp/tri4a_alignme'], workdir)
        do_cmd( ['steps/align_si.sh', '--nj', '1', '--cmd', 'run.pl', 'data/alignme', str( (kaldi_model_path / 'data' / 'lang').resolve() ), str((kaldi_model_path / 'exp' / 'tri4a').resolve()), 'exp/tri4a_alignme'], workdir)
        do_cmd( ['src/bin/ali-to-phones', '--ctm-output', str( (kaldi_model_path / 'exp' / 'tri4a' / 'final.mdl').resolve() ), 'ark:gunzip -c exp/tri4a_alignme/ali.1.gz|', 'foo.ctm'], workdir)

        with open (workdir / 'foo.ctm') as ctmf:
            for line in ctmf:
                #print(line)

                parts = line.strip().split(' ')
                assert len(parts)==5

                uttid = parts[0]
                start = float(parts[2])
                stop = float(parts[3])
                pid = int(parts[4])

                phone = phonemap[pid]

                l = f"{start}\t{stop}\t{phone}"
                # print(l)

                with open (aligndir / (uttid + '.tsv'), 'a') as alignf:
                    alignf.write(l + '\n')

        sys.stdout.flush()
        