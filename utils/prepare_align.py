#!/bin/env python3

'''
zerovox

    Apache 2.0 License
    2024 by Guenter Bartsch

is based on:

    EfficientSpeech: An On-Device Text to Speech Model
    https://ieeexplore.ieee.org/abstract/document/10094639
    Rowel Atienza
    Apache 2.0 License
'''

import argparse
import yaml
import os

import subprocess
import multiprocessing
from tqdm import tqdm

from zerovox.g2p.tokenizer import G2PTokenizer

def _clean_text(text:str, lang:str) -> str:

    tokenizer = G2PTokenizer(lang)

    tokens = tokenizer.tokenize(text)

    return ' '.join(tokens)

def gen_jobs_from_metadata_file(in_dir, metadata_path, out_dir, language, sampling_rate, book=None, max_length=512):

    jobs = []

    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")

            base_name = parts[0]
            if os.sep in base_name:
                base_name = os.path.basename(base_name)
            if base_name.endswith('.wav'):
                base_name = os.path.splitext(base_name)[0]

            text = parts[1] if len(parts) == 2 else parts[2]
            if len(text) > max_length:
                #print (f"skipping: {base_name}")
                continue

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):

                dest_base_name = book + '_' + base_name if book else base_name

                jobs.append({'text': text,
                             'language': language,
                             'sampling_rate': sampling_rate,
                             'wav_path': wav_path,
                             'dest_wav': os.path.join(out_dir, "{}.wav".format(dest_base_name)),
                             'dest_lab': os.path.join(out_dir, "{}.lab".format(dest_base_name))
                             })

    print (f"{metadata_path} -> {len(jobs)} jobs")

    return jobs

def process_job (job):

    # print (f"processing: {job}")

    text = _clean_text(job['text'], job['language'])

    cmd = [ 'ffmpeg', '-y', '-v', 'quiet',
            '-i', job['wav_path'],
            '-filter', f'acompressor,loudnorm=I=-14.0,aresample={job["sampling_rate"]}',
            '-ac', '1',
            job['dest_wav'] ]

    subprocess.run(cmd)
    with open(job['dest_lab'], 'w') as f1:
        f1.write(text)

def gather_jobs_from_config(configfn: os.PathLike, limit: int=0):

    config = yaml.load(open(configfn, "r"), Loader=yaml.FullLoader)

    if "LJSpeech" not in config["dataset"]:
        raise Exception (f"unknown dataset format '{config['dataset']}")

    in_dir = config["path"]["corpus_path"]

    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    language = config["preprocessing"]["text"]["language"]
    max_length = config["preprocessing"]["text"]["max_length"]

    os.makedirs(out_dir, exist_ok=True)

    # auto-detect corpus format: single or multi-book?

    metadata_path = os.path.join(in_dir, "metadata.csv")

    if os.path.isfile(metadata_path):

        jobs = gen_jobs_from_metadata_file (in_dir, metadata_path, out_dir, language, sampling_rate, max_length=max_length)

    else:

        jobs = []

        for book in os.listdir(in_dir):

            bookdir = os.path.join(in_dir, book)

            metadata_path = os.path.join(bookdir, 'metadata.csv')

            if os.path.isfile(metadata_path):
                jobs.extend(gen_jobs_from_metadata_file (bookdir, metadata_path, out_dir, language, sampling_rate, book=book, max_length=max_length))

    if limit:
        jobs = jobs[:limit]

    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs='+', help="path[s] to .yaml config file[s] or directorie[s]")
    parser.add_argument("--limit", type=int, default=100, help="limit number auf audio files to process per config, default: 100 (0=unlimited)")
    parser.add_argument("--num-workers", type=int, default=12, help="number of parallel processes, default: 12")
    args = parser.parse_args()

    jobs = []

    for configfn in args.configs:

        if os.path.isdir(configfn):

            for cfgfn in os.listdir(configfn):

                _, ext = os.path.splitext(cfgfn)
                if ext != '.yaml':
                    continue

                j = gather_jobs_from_config (os.path.join(configfn, cfgfn), args.limit)
                jobs.extend(j)

        else:
            _, ext = os.path.splitext(configfn)
            if ext != '.yaml':
                continue

            j = gather_jobs_from_config (configfn, args.limit)
            jobs.extend(j)

    print(f"gathered {len(jobs)} jobs.")

    with multiprocessing.Pool(processes=args.num_workers) as pool:

        for _ in tqdm (pool.imap(process_job, jobs), total=len(jobs)):
            pass
