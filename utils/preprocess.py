#!/bin/env python3

# zerovox

#    Apache 2.0 License
#    2024, 2025 by Guenter Bartsch

#
# CTC forced alignment based ZeroVOX II preprocessing
#
# see https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html
#
# 2025/01/05
#

#import warnings
# FIXME: keep enabled?
#warnings.filterwarnings("error")

import os
import sys
import multiprocessing
import argparse
import subprocess
import shutil

import yaml
import json

from tqdm import tqdm

import torch
import torchaudio
import numpy as np
import scipy

import librosa
import pyworld

from zerovox.tts.mels import get_mel_from_wav
from zerovox.tts.symbols import Symbols
from zerovox.tts.normalize import zerovox_normalize
from zerovox.tts.data import PREPROCESSED_DATA_PATH

MEL_LEN_HEADROOM = 10 # reduce max_mel_len by this margin to have some headroom in training
MIN_TXT_LEN      = 5  # characters

def batch_list(data, batch_size):
  """
  Transforms a list into a list of batches of up to a given max batch size.

  Args:
    data: The input list.
    batch_size: The maximum size of each batch.

  Returns:
    A list of batches, where each batch is a sublist of `data`.
  """
  batches = []
  current_batch = []
  for item in data:
    current_batch.append(item)
    if len(current_batch) == batch_size:
      batches.append(current_batch)
      current_batch = []
  if current_batch:
    batches.append(current_batch)
  return batches

def load_and_preprocess_audio(job):

    audio_path = job['audio_path']
    target_sr  = job['target_sr']

    # Load the WAV file
    sr, audio = scipy.io.wavfile.read(audio_path)

    # Normalize the audio to floating-point values between -1 and 1
    if audio.dtype != 'float32':
        audio = audio / 32768.0  # Assuming 16-bit signed integer data

    audio = audio.astype(np.float32)

    # Calculate the number of samples in the resampled data
    num_samples = int(len(audio) * (target_sr / sr))

    # Resample the audio data
    audio_resampled = scipy.signal.resample(audio, num_samples)
  
    return audio_resampled


def first_and_last_hop_above_threshold(audio, hop_size, threshold):
  """
  Computes the last hop index in a mono audio waveform (torch tensor) 
  that contains audio above a given noise threshold.

  Args:
    audio: A 1-dimensional torch tensor representing the audio waveform.
    hop_size: The hop size in samples.
    threshold: The noise threshold.

  Returns:
    The index of the last hop that contains audio above the threshold, 
    or -1 if no hop exceeds the threshold.
  """

  # Calculate the number of hops
  num_hops = max(0, (audio.shape[0] - 1) // hop_size)

  # Create a boolean mask indicating hops with audio above threshold
  hop_masks = torch.zeros(num_hops, dtype=torch.bool)
  for i in range(num_hops):
    start = i * hop_size
    end = min((i + 1) * hop_size, audio.shape[0])
    hop_audio = audio[start:end]
    hop_masks[i] = torch.any(torch.abs(hop_audio) > threshold)

  # Find the index of the first and last True value in the mask
  first_hop_idx = torch.where(hop_masks)[0].min().item() if torch.any(hop_masks) else 0
  last_hop_idx = torch.where(hop_masks)[0].max().item() if torch.any(hop_masks) else num_hops-1

  return first_hop_idx, last_hop_idx

def zerovox_normalize_helper(job):

    transcript, lang = job

    return zerovox_normalize(transcript=transcript, lang=lang)

class AudioPreprocessor:

    def __init__(self, modelcfg, use_cuda, verbose):

        self._modelcfg             = modelcfg
        self._verbose              = verbose

        self._target_sampling_rate = modelcfg['audio']['sampling_rate']
        self._fft_size             = modelcfg['audio']["fft_size"]
        self._hop_size             = modelcfg['audio']["hop_size"]
        self._win_length           = modelcfg['audio']["win_length"]
        self._num_mels             = modelcfg['audio']["num_mels"]
        self._fmin                 = modelcfg['audio']["fmin"]
        self._fmax                 = modelcfg['audio']["fmax"]

    def process(self, job):

        if 'durations' not in job:
            return None

        out_dir = job['out_dir']

        destwav = f"{out_dir}/wavs/{job['dest_wav']}"

        cmd = [ 'ffmpeg', '-y', '-v', 'quiet',
                '-i', job['wav_path'],
                '-filter', f'acompressor,loudnorm=I=-14.0,aresample={self._target_sampling_rate}',
                '-ac', '1',
                destwav ]

        subprocess.run(cmd)

        # mel

        if self._verbose:
            print (f"{destwav}: librosa.load")
            sys.stdout.flush()

        # Read and trim wav
        wav, _ = librosa.load(destwav, sr=self._target_sampling_rate)
        wav = wav[
            job['start_hop'] * self._hop_size : job['end_hop'] * self._hop_size
        ].astype(np.float32)

        # Compute fundamental frequency
        if self._verbose:
            print (f"{destwav}: pyworld.dio")
            sys.stdout.flush()
        pitch, t = pyworld.dio(
            wav.astype(np.float64),
            self._target_sampling_rate,
            frame_period=self._hop_size / self._target_sampling_rate * 1000,
        )
        if self._verbose:
            print (f"{destwav}: pyworld.stonemask")
            sys.stdout.flush()
        pitch = pyworld.stonemask(wav.astype(np.float64), pitch, t, self._target_sampling_rate)

        # librosa alternative
        # f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # pitch = pitch[: sum(durations)]
        # if np.sum(pitch != 0) <= 1:
        #     continue

        # Compute mel-scale spectrogram and energy

        if self._verbose:
            print (f"{destwav}: get_mel_from_wav")
            sys.stdout.flush()
        mel_spectrogram, energy = get_mel_from_wav(audio=wav,
                    sampling_rate=self._target_sampling_rate,
                    fft_size=self._fft_size, # =1024,
                    hop_size=self._hop_size, # =256,
                    win_length=self._win_length, # =1024,
                    num_mels=self._num_mels, #=80,
                    fmin=self._fmin, #=0,
                    fmax=self._fmax, #=8000,
                    )

        durations = job['durations']

        # mel_spectrogram = mel_spectrogram[:, : sum(durations)]
        # energy = energy[: sum(durations)]

        # compute pitch per phoneme
        if self._verbose:
            print (f"{destwav}: compute pitch per phoneme")
            sys.stdout.flush()
        phoneme_pitches = np.zeros(len(durations), dtype=pitch.dtype)

        # perform linear pitch interpolation
        nonzero_ids = np.where(pitch != 0)[0]
        if not nonzero_ids.any():
            return None
        interp_fn = scipy.interpolate.interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        # Phoneme-level average
        if self._verbose:
            print (f"{destwav}: Phoneme-level average")
            sys.stdout.flush()
        pos = 0
        for i, d in enumerate(durations):
            #pos = phone_positions[i]
            if (d>0) and (pos+d < len(pitch)):
                phoneme_pitches[i] = np.mean(pitch[pos : pos + d])
            else:
                if pos < len(pitch):
                    phoneme_pitches[i] = pitch[pos]
                else:
                    phoneme_pitches[i] = pitch[-1]
            pos += d

        # compute mean energy per phoneme
        if self._verbose:
            print (f"{destwav}: compute mean energy per phoneme")
            sys.stdout.flush()
        pos = 0
        phoneme_energy = np.zeros(len(durations), dtype=pitch.dtype)
        for i, d in enumerate(durations):
            #pos = phone_positions[i]
            if (d>0) and (pos+d < len(energy)):
                phoneme_energy[i] = np.mean(energy[pos : pos + d])
            else:
                if pos < len(energy):
                    phoneme_energy[i] = energy[pos]
                else:
                    phoneme_energy[i] = energy[-1]
            pos += d

        # make sure sum(durations) matches mel_spectrogram precisely
        diff =  mel_spectrogram.shape[1] - sum(durations)
        # print(diff)
        durations[-1] += diff
        assert sum(durations) == mel_spectrogram.shape[1]
        if min(durations)<0:
            print (f"{destwav}: negative duration detected: {durations} -> skipping")
            return None

        # Save files

        if self._verbose:
            print (f"{destwav}: save")
            sys.stdout.flush()

        basename = os.path.basename(destwav)
        basename = os.path.splitext(basename)[0]

        dur_filename = f"duration-{basename}.npy"
        np.save(os.path.join(out_dir, "duration", dur_filename), durations)

        pitch_filename = f"pitch-{basename}.npy"
        np.save(os.path.join(out_dir, "pitch", pitch_filename), phoneme_pitches)

        energy_filename = f"energy-{basename}.npy"
        np.save(os.path.join(out_dir, "energy", energy_filename), phoneme_energy)

        mel_filename = f"mel-{basename}.npy"
        np.save(os.path.join(out_dir, "mel", mel_filename), mel_spectrogram.T)

        startstop_filename = f"startstop-{basename}.json"
        with open(os.path.join(out_dir, "mel", startstop_filename), "w") as f:
            f.write(json.dumps({'start_hop':job['start_hop'], 'end_hop': job['end_hop']}))

        pmin = np.min(pitch)
        pmax = np.max(pitch)

        emin = np.min(energy)
        emax = np.max(energy)

        return pmin, pmax, emin, emax



class Preprocessor:

    def __init__(self, modelcfg, lang, min_avg_score):

        self._device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lang           = lang
        self._min_avg_score  = min_avg_score

        self._syms = Symbols (phones=modelcfg['model']['phones'], puncts=modelcfg['model']['puncts'])
        self._extra_puncts = set()

        self._modelcfg             = modelcfg
        self._max_txt_len          = modelcfg['model']["max_txt_len"]
        self._max_mel_len          = modelcfg['model']["max_mel_len"] - MEL_LEN_HEADROOM
        self._min_mel_len          = modelcfg['model']["min_mel_len"]
        self._target_sampling_rate = modelcfg['audio']['sampling_rate']
        self._hop_size             = modelcfg['audio']["hop_size"]

        # alignment model

        print ("loading the alignment model...")

        self._bundle = torchaudio.pipelines.MMS_FA
        self._model = self._bundle.get_model(with_star=False).to(self._device)

        self._align_sample_rate = int(self._bundle.sample_rate)
        self._align_hop_size    = 320

        self._labels = self._bundle.get_labels(star=None)
        self._dictionary = self._bundle.get_dict(star=None)

        print ("loading the alignment model... done.")

    def ahop2thop (self, hop):
        # convert alignment model hop(hop_size=320, sr=16000) to target hop (hop_size=256, sr=22050)

        aframe = hop * self._align_hop_size

        tframe = aframe * self._target_sampling_rate / self._align_sample_rate

        thop = int(round(tframe / self._hop_size))

        return thop

    def align (self, jobs, out_dir, batch_size, max_txt_len, pool, lang):

        print ("normalizing...")

        # normalize text

        with tqdm(total=len(jobs), desc="normalize") as pbar:

            transcripts_uroman = []
            transcripts_normalized = []

            for transcript_uroman, transcript_normalized in p.map(zerovox_normalize_helper, [(job['transcript'], lang) for job in jobs]):

                pbar.update()
                pbar.refresh()

                transcripts_uroman.append(transcript_uroman)
                transcripts_normalized.append(transcript_normalized)

            for transcript_uroman, transcript_normalized, job in zip(transcripts_uroman, transcripts_normalized, jobs):
                job['transcript_uroman'] = transcript_uroman
                job['transcript_normalized'] = transcript_normalized

        # filter out transcripts that are either too short or too long

        jobs_filtered = []

        for job in jobs:

            if len(job['transcript_normalized']) < MIN_TXT_LEN:
                print (f"dropping sample {job['base_name']} because it is too short")
                continue
            if len(job['transcript_normalized']) > max_txt_len:
                print (f"dropping sample {job['base_name']} because it exceeds max_txt_len ({max_txt_len})")
                continue

            jobs_filtered.append(job)

        batches = batch_list(jobs_filtered, batch_size=batch_size)

        for batch in tqdm(batches, desc="align"):

            waveforms = pool.map(load_and_preprocess_audio, [{'audio_path': job['wav_path'], 'target_sr':self._align_sample_rate} for job in batch])

            max_len = max(len(arr) for arr in waveforms)

            padded_wfs = []
            for wf in waveforms:
                pad_width = [(0, max_len - len(wf))] 
                padded_wf = np.pad(wf, pad_width, mode='constant', constant_values=0.0)
                padded_wfs.append(padded_wf)

            # Convert to PyTorch tensor
            audio_tensor = torch.from_numpy(np.stack(padded_wfs))

            with torch.inference_mode():
                emissions, _ = self._model(audio_tensor.to(self._device))

            # forced alignment

            for emission, job, audio in zip(emissions, batch, audio_tensor):

                try:
                    tt = [self._dictionary[c] for word in job['transcript_normalized'].split(' ') for c in word]
                    targets = torch.tensor([tt], dtype=torch.int32, device=self._device)

                    aligned_tokens, alignment_scores = torchaudio.functional.forced_align(emission.unsqueeze(0), targets, blank=0)

                    tokens = aligned_tokens[0]
                    scores = alignment_scores[0]

                    scores = scores.exp()  # convert back to probability

                    if len(scores)==0:
                        print (f"{job['wav_path']}: *** dropping sample because alignment failed")
                        continue

                    # min_score = min(scores)
                    avg_score = sum(scores) / len(scores)
                    # print (f"{job['wav_path']}: min_score: {min_score}, avg_score: {avg_score}")

                    if avg_score < self._min_avg_score:
                        print (f"{job['wav_path']}: *** dropping sample because avg alignment score is too low: {avg_score} < {self._min_avg_score}")
                        continue

                except RuntimeError as e:
                    print (f"{job['wav_path']}: *** dropping sample due to RuntimeError: {e}")
                    continue


                # Token-level alignments

                token_spans = torchaudio.functional.merge_tokens(tokens, scores)

                ts_pos           = 0
                end_hop          = 0
                durations        = []
                puncts           = []
                phones           = []

                # include some extra hops at start and end (model tends to truncate phones)

                start_hop, end_hop_th = first_and_last_hop_above_threshold(audio, self._align_hop_size, threshold=0.004)
                if token_spans:
                    s = token_spans[0]
                    if s.start<start_hop:
                        start_hop = s.start

                # convert all time markers to target hops so we don't have to worry about that anymore

                last_token_start = start_hop = self.ahop2thop(start_hop)
                end_hop_th = self.ahop2thop(end_hop_th)
                for s in token_spans:
                    s.start = self.ahop2thop(s.start)
                    s.end   = self.ahop2thop(s.end)

                transcript_uroman = job['transcript_uroman']

                for s_idx, s in enumerate(token_spans):

                    if ts_pos >= len(transcript_uroman):
                        raise Exception ("alignment error: ran out of transcript_uroman!")

                    # print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")

                    token = self._labels[s.token]

                    # collect punctuation leading up to this token
                    punct = self._syms.encode_punct(Symbols.NO_PUNCT)
                    
                    while (ts_pos < len(transcript_uroman)) and (transcript_uroman[ts_pos] != token):

                        cp = transcript_uroman[ts_pos]

                        # print (f"punctuation: {cp}")

                        if self._syms.is_punct(cp):
                            punct_id = self._syms.encode_punct(cp)
                            if punct_id >punct:
                                punct = punct_id
                        else:
                            self._extra_puncts.add(cp)

                        ts_pos += 1

                    if (ts_pos >= len(transcript_uroman)) or (transcript_uroman[ts_pos] != token):
                        raise Exception ("alignment error: transcript_uroman mismatch!")

                    ts_pos += 1

                    if s_idx > 0:
                        # extend phonemes to include pauses between them

                        extra_hops = s.start - last_token_start - durations[s_idx-1]
                        assert extra_hops >= 0
                        extra_hops_next_token = extra_hops // 2
                        extra_hops_prev_token = extra_hops - extra_hops_next_token
                        durations[s_idx-1] += extra_hops_prev_token
                        s.start -= extra_hops_next_token

                        puncts[s_idx-1] = punct
                        last_token_start = s.start

                    durations.append(s.end-s.start)
                    puncts.append(0)
                    phones.append(self._syms.encode_phone(token))

                if not durations:
                    return None

                s = token_spans[-1]
                end_hop = s.end

                # maybe include some extra hops (model tends to truncate phones)
                if end_hop_th>end_hop:
                    end_hop = end_hop_th

                # deal with extra puncts at the end and last token / total duration
                durations[-1] = end_hop-s.start

                assert min(durations)>=0

                assert sum(durations) == end_hop-start_hop

                punct = self._syms.encode_punct(Symbols.NO_PUNCT)
                while ts_pos < len(transcript_uroman):

                    cp = transcript_uroman[ts_pos]

                    # print (f"punctuation: {cp}")

                    if self._syms.is_punct(cp):
                        punct_id = self._syms.encode_punct(cp)
                        if punct_id >punct:
                            punct = punct_id
                    else:
                        self._extra_puncts.add(cp)

                    ts_pos += 1

                puncts[-1] = punct

                # check and write final result

                total_hops = end_hop - start_hop
                if total_hops <= self._max_mel_len and total_hops >= self._min_mel_len:

                    job['start_hop'] = start_hop
                    job['end_hop'] = end_hop
                    job['durations'] = durations

                    metafn = f"{out_dir}/train.txt"
                    with open(metafn, 'a') as metaf:
                        metaf.write(f"{job['dest_wav']}|{','.join([str(p) for p in phones])}|{','.join([str(p) for p in puncts])}|{job['transcript']}\n")

                    destwav = f"{out_dir}/wavs/{job['dest_wav']}"
                    with open (destwav + ".txt", 'w') as labelf:
                        pos = start_hop
                        for phone, punct, dur in zip (phones, puncts, durations):
                            labelf.write(f"{float(pos*self._hop_size)/self._target_sampling_rate}\t{float((pos+dur)*self._hop_size)/self._target_sampling_rate}\t{self._syms.decode_phone(phone)}\n")
                            pos += dur
                else:
                    print (f"*** {job['wav_path']}: dropping sample because it exceeds mel len limits: {total_hops} vs [{self._min_mel_len}:{self._max_mel_len}]")



def gen_jobs_from_metadata_file(in_dir, out_dir, metadata_path, limit, book=None):

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

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):

                dest_base_name = book + '_' + base_name if book else base_name

                jobs.append({'transcript': text,
                             'wav_path': wav_path,
                             'dest_wav': f"{dest_base_name}.wav",
                             'out_dir': out_dir,
                             'base_name': base_name
                             })
                if len(jobs) >= limit:
                    break

    print (f"{metadata_path} -> {len(jobs)} jobs")

    return jobs

def gather_jobs_from_config(config, limit: int):

    if "LJSpeech" not in config["dataset"]:
        raise Exception (f"unknown dataset format '{config['dataset']}")

    in_dir = config["path"]["corpus_path"]

    out_dir = os.path.join(PREPROCESSED_DATA_PATH, config["path"]["preprocessed_path"])

    shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs((os.path.join(out_dir, "wavs")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "mel")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "pitch")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "energy")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "duration")), exist_ok=True)
    # os.makedirs((os.path.join(out_dir, "phonemepos")), exist_ok=True)

    # auto-detect corpus format: single or multi-book?

    metadata_path = os.path.join(in_dir, "metadata.csv")

    # max_txt_len=23

    if os.path.isfile(metadata_path):

        jobs = gen_jobs_from_metadata_file (in_dir=in_dir, out_dir = out_dir, metadata_path=metadata_path, limit=limit)

    else:

        jobs = []

        for book in os.listdir(in_dir):

            bookdir = os.path.join(in_dir, book)

            metadata_path = os.path.join(bookdir, 'metadata.csv')

            if os.path.isfile(metadata_path):
                jobs.extend(gen_jobs_from_metadata_file (in_dir=bookdir, out_dir = out_dir, metadata_path=metadata_path, book=book, limit=limit-len(jobs)))

    return jobs


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("modelcfg", type=str, help="model config to preprocess for")
    parser.add_argument("corpora", type=str, nargs='+', help="path[s] to corpus .yaml config file[s] or directorie[s]")
    parser.add_argument("-l", "--limit", type=int, default=1000, help="limit number auf audio files to process per config, default: 1000 (0=unlimited)")
    parser.add_argument("-j", "--num-jobs", type=int, default=multiprocessing.cpu_count(), help=f"number of parallel jobs, default: {multiprocessing.cpu_count()}")
    parser.add_argument("-m", "--min-alignment-score", type=float, default=0.9, help="minimum required alignment score to keep sample, default: 0.9 (90%)")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help=f"number of parallel jobs, default: 4")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    modelcfg = yaml.load(open(args.modelcfg, "r"), Loader=yaml.FullLoader)

    print (f"audio cfg:\n{modelcfg['audio']}")
    print (f"max txt len: {modelcfg['model']['max_txt_len']}, max mel len: {modelcfg['model']['max_mel_len']}")

    # gather corpus configs

    corpus_configs = []
    for corpusfn in tqdm(args.corpora, desc="collect corpora"):
        if os.path.isdir(corpusfn):
            for cfn in os.listdir(corpusfn):

                _, ext = os.path.splitext(cfn)
                if ext != '.yaml':
                    continue

                cfpath = os.path.join(corpusfn, cfn)
                cfg = yaml.load(open(cfpath, "r"), Loader=yaml.FullLoader)
                corpus_configs.append(cfg)
        else:
            cfg = yaml.load(open(corpusfn, "r"), Loader=yaml.FullLoader)
            corpus_configs.append(cfg)

    if not corpus_configs:
        raise Exception ("*** error: no .yaml files found!")
    
    print (f"{len(corpus_configs)} corpora found.")

    lang = None
    for corpus in corpus_configs:
        if not lang:
            lang = corpus['language']
        else:
            if lang != corpus['language']:
                raise Exception ('inconsistent languages detected')

    print (f"language is {lang}")

    pproc = Preprocessor (modelcfg, lang=lang, min_avg_score=args.min_alignment_score)
    aproc = AudioPreprocessor(modelcfg=modelcfg, use_cuda=False, verbose=args.verbose)

    with multiprocessing.Pool(args.num_jobs) as p:
        
        for cfg in corpus_configs:

            jobs = gather_jobs_from_config (cfg, limit=args.limit)

            print(f"gathered {len(jobs)} jobs.")

            pproc.align (jobs, out_dir = os.path.join(PREPROCESSED_DATA_PATH, cfg["path"]["preprocessed_path"]), batch_size=args.batch_size, max_txt_len=modelcfg['model']["max_txt_len"], pool=p, lang=lang)

            pitch_min = np.finfo(np.float64).max
            pitch_max = np.finfo(np.float64).min
            energy_min = np.finfo(np.float64).max
            energy_max = np.finfo(np.float64).min

            with tqdm(total=len(jobs), desc="audio") as pbar:

                for stats in p.imap_unordered(aproc.process, jobs):

                    pbar.update()
                    pbar.refresh()

                    # update statistics
                    if not stats:
                        continue

                    pmin, pmax, emin, emax = stats

                    if pmin < pitch_min:
                        pitch_min = pmin
                    if pmax > pitch_max:
                        pitch_max = pmax

                    if emin < energy_min:
                        energy_min = emin
                    if emax > energy_max:
                        energy_max = emax


            with open(os.path.join(PREPROCESSED_DATA_PATH, cfg["path"]["preprocessed_path"], "stats.json"), "w") as f:
                stats = {
                    "pitch": [
                        float(pitch_min),
                        float(pitch_max)
                    ],
                    "energy": [
                        float(energy_min),
                        float(energy_max)
                    ],
                }
                f.write(json.dumps(stats))


        print (f"extra puncts : {pproc._extra_puncts}")
