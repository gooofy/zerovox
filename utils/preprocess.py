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

import os
import re
import argparse
import subprocess
import shutil

import uroman
import yaml
import json

from tqdm import tqdm
from num2words import num2words

import torch
import torchaudio
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

import librosa
import pyworld

from zerovox.tts.mels import get_mel_from_wav, TacotronSTFT
from zerovox.tts.symbols import Symbols

MEL_LEN_HEADROOM = 10 # reduce max_mel_len by this margin to have some headroom in training

#DEBUG_OFFSET = 433
DEBUG_OFFSET = 0

def spell_out_numbers(text, lang):
    """
    Spells out numbers within a string using num2words.

    Args:
        text: The input string.

    Returns:
        The string with numbers spelled out, or the original string if no numbers are found.
        Returns an error message if num2words raises an exception (like for very large numbers).
    """
    try:
        def replace_number(match):
            num_str = match.group(0)
            try:
                num = int(num_str)
                return num2words(num, lang=lang)
            except ValueError:  # Handle floats
                try:
                    num = float(num_str)
                    if num.is_integer(): #check if it is an integer represented as float
                        return num2words(int(num), lang=lang)
                    else:
                        parts = str(num).split('.')
                        integer_part = num2words(int(parts[0]), lang=lang)
                        decimal_part = parts[1]
                        decimal_as_int = int(decimal_part)
                        decimal_spelled = num2words(decimal_as_int, lang=lang)
                        return f"{integer_part} point {decimal_spelled}"
                except ValueError:
                    return num_str  # Return original if not a valid number
            except OverflowError: #handle numbers too big for int
                return f"Number too large to spell out"
            except Exception as e:
                return f"Error spelling out number: {e}"
                
        pattern = r"\b\d+(\.\d+)?\b"  # Matches whole numbers or decimals
        new_text = re.sub(pattern, replace_number, text)

        pattern = r"\d+"  # Matches remaining digits or numbers
        new_text = re.sub(pattern, replace_number, new_text)

        return new_text
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def normalize_uroman(text_uroman):
    text = re.sub("([^a-z' ])", " ", text_uroman)
    text = re.sub(' +', ' ', text)
    return text.strip()

def load_and_preprocess_audio(audio_path, target_sr):
  """
  Loads a mono WAV file and resamples it to the target sample rate

  Args:
    audio_path: Path to the WAV file.
    target_sr: Target sample rate for resampling.

  Returns:
    A PyTorch tensor representing the audio samples.
  """

  # Load audio using librosa
  audio, sr = librosa.load(audio_path, mono=True) 

  # Resample audio to target sample rate
  if sr != target_sr:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

  # Convert to PyTorch tensor
  audio_tensor = torch.from_numpy(audio).float()

  return audio_tensor

def last_hop_above_threshold(audio, hop_size, threshold):
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

  # Find the index of the last True value in the mask
  last_hop_idx = torch.where(hop_masks)[0].max().item() if torch.any(hop_masks) else -1

  return last_hop_idx



class Preprocessor:

    def __init__(self, modelcfg, use_cuda=True):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._uromanizer = uroman.Uroman()
        self._syms = Symbols (phones=modelcfg['model']['phones'], puncts=modelcfg['model']['puncts'])
        self._extra_puncts = set()

        self._modelcfg             = modelcfg
        self._target_sampling_rate = modelcfg['audio']['sampling_rate']

        # mel spectrograms

        self._max_txt_len   = modelcfg['model']["max_txt_len"]
        self._max_mel_len   = modelcfg['model']["max_mel_len"] - MEL_LEN_HEADROOM
        self._fft_size      = modelcfg['audio']["fft_size"]
        self._hop_size      = modelcfg['audio']["hop_size"]
        self._win_length    = modelcfg['audio']["win_length"]
        self._window        = modelcfg['audio']["window"]
        self._num_mels      = modelcfg['audio']["num_mels"]
        self._fmin          = modelcfg['audio']["fmin"]
        self._fmax          = modelcfg['audio']["fmax"]
        self._eps           = float(modelcfg['audio']["eps"])
        self._log_base      = float(modelcfg['audio']["log_base"])
        self._filter_length = modelcfg['audio']["filter_length"]        

        self._stft = TacotronSTFT(filter_length=self._filter_length, 
                                  hop_length=self._hop_size,
                                  win_length=self._win_length,
                                  n_mel_channels=self._num_mels,
                                  sampling_rate=self._target_sampling_rate,
                                  mel_fmin=self._fmin,
                                  mel_fmax=self._fmax,
                                  use_cuda=use_cuda)


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
        # convert alignment model hop(hop_size=320, sr=16000) to target hop (hop_size=300, sr=24000)

        aframe = hop * self._align_hop_size

        tframe = aframe * self._target_sampling_rate / self._align_sample_rate

        thop = int(round(tframe / self._hop_size))

        return thop

    def get_alignment(self, wav_path, transcript, lang):

        waveform = load_and_preprocess_audio(wav_path, target_sr=self._align_sample_rate)
        waveform = waveform.unsqueeze(dim=0)

        with torch.inference_mode():
            emission, _ = self._model(waveform.to(self._device))

        # print(f"emissions: {emission.shape}")

        # normalize the transcript

        transcript_uroman = transcript.replace("’", "'")
        transcript_uroman = spell_out_numbers(transcript_uroman, lang=lang)
        transcript_uroman = str(self._uromanizer.romanize_string(transcript_uroman)).lower().strip()

        if len(transcript_uroman) > self._max_txt_len:
            return None

        transcript_normalized = normalize_uroman(transcript_uroman).split(' ')

        tokenized_transcript = [self._dictionary[c] for word in transcript_normalized for c in word]

        # print (f"tokenized transcript: {tokenized_transcript}")

        # Computing alignments

        # Frame-level alignments

        targets = torch.tensor([tokenized_transcript], dtype=torch.int32, device=self._device)
        alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)

        aligned_tokens, alignment_scores = alignments[0], scores[0]  # remove batch dimension for simplicity
        alignment_scores = alignment_scores.exp()  # convert back to probability

        # Token-level alignments

        # Next step is to resolve the repetation, so that each alignment does not depend 
        # on previous alignments. torchaudio.functional.merge_tokens() computes the 
        # TokenSpan object, which represents which token from the transcript is present 
        # at what time span.

        token_spans = torchaudio.functional.merge_tokens(aligned_tokens, alignment_scores)

        ts_pos           = 0
        start_hop        = 0
        end_hop          = 0
        durations        = []
        puncts           = []
        phones           = []
        last_token_start = 0

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

            if s_idx == 0:
                start_hop = self.ahop2thop(s.start)
            else:
                durations[s_idx-1] = self.ahop2thop(s.start) - last_token_start
                puncts[s_idx-1] = punct

            durations.append(0)
            puncts.append(0)
            phones.append(self._syms.encode_phone(token))
            last_token_start = self.ahop2thop(s.start)

        if not durations:
            return None

        s = token_spans[-1]
        end_hop = self.ahop2thop(s.end)

        # maybe include some extra hops (model tends to truncate phones)
        end_hop_th = self.ahop2thop(last_hop_above_threshold(waveform[0], self._align_hop_size, threshold=0.02))
        if end_hop_th>end_hop:
            end_hop = end_hop_th

        # deal with extra puncts at the end and last token / total duration
        durations[-1] = end_hop-self.ahop2thop(s.start)

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

        #print (phones)
        #print (puncts)
        #print (durations)

        return phones, puncts, durations, start_hop, end_hop

    def process (self, jobs, out_dir, lang):

        pitch_min = np.finfo(np.float64).max
        pitch_max = np.finfo(np.float64).min
        energy_min = np.finfo(np.float64).max
        energy_max = np.finfo(np.float64).min

        cnt = 0

        for job in tqdm(jobs):

            cnt += 1
            if DEBUG_OFFSET and cnt<DEBUG_OFFSET:
                continue

            # audio preprocessing

            destwav = f"{out_dir}/wavs/{job['dest_wav']}"

            cmd = [ 'ffmpeg', '-y', '-v', 'quiet',
                    '-i', job['wav_path'],
                    '-filter', f'acompressor,loudnorm=I=-14.0,aresample={self._target_sampling_rate}',
                    '-ac', '1',
                    destwav ]

            subprocess.run(cmd)

            # transcript processing

            alignment = self.get_alignment(wav_path=destwav, transcript=job['text'], lang=lang)

            if not alignment:
                continue

            phones, puncts, durations, start_hop, end_hop = alignment

            with open (destwav + ".txt", 'w') as labelf:
                pos = start_hop
                for phone, punct, dur in zip (phones, puncts, durations):
                    labelf.write(f"{float(pos*self._hop_size)/self._target_sampling_rate}\t{float((pos+dur)*self._hop_size)/self._target_sampling_rate}\t{self._syms.decode_phone(phone)}\n")
                    pos += dur

            # mel

            # Read and trim wav
            wav, _ = librosa.load(destwav, sr=self._target_sampling_rate)
            wav = wav[
                start_hop * self._hop_size : end_hop * self._hop_size
            ].astype(np.float32)

            #print (f"wav shape: {wav.shape}")

            # Compute fundamental frequency
            pitch, t = pyworld.dio(
                wav.astype(np.float64),
                self._target_sampling_rate,
                frame_period=self._hop_size / self._target_sampling_rate * 1000,
            )
            pitch = pyworld.stonemask(wav.astype(np.float64), pitch, t, self._target_sampling_rate)

            # pitch = pitch[: sum(durations)]
            if np.sum(pitch != 0) <= 1:
                continue

            # Compute mel-scale spectrogram and energy

            mel_spectrogram, energy = get_mel_from_wav(audio=wav,
                        sampling_rate=self._target_sampling_rate,
                        fft_size=self._fft_size, # =2048,
                        hop_size=self._hop_size, # =300,
                        win_length=self._win_length, # =1200,
                        window=self._window, # ="hann",
                        num_mels=self._num_mels, #=80,
                        fmin=self._fmin, #=80,
                        fmax=self._fmax, #=7600,
                        eps=self._eps, #=1e-10,
                        log_base=self._log_base, #=10.0,
                        stft=self._stft)

            if mel_spectrogram.shape[1] > self._max_mel_len:
                print (f"*** dropping sample because it exceeds mel max_mel_len: {destwav}")
                continue
            
            # mel_spectrogram = mel_spectrogram[:, : sum(durations)]
            # energy = energy[: sum(durations)]

            # compute pitch per phoneme
            phoneme_pitches = np.zeros(len(durations), dtype=pitch.dtype)

            # perform linear pitch interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = scipy.interpolate.interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
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

            # Save files

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

            metafn = f"{out_dir}/train.txt"
            with open(metafn, 'a') as metaf:
                metaf.write(f"{job['dest_wav']}|{','.join([str(p) for p in phones])}|{','.join([str(p) for p in puncts])}|{job['text']}\n")

            # update statistics

            pmin = np.min(pitch)
            if pmin < pitch_min:
                pitch_min = pmin
            pmax = np.max(pitch)
            if pmax > pitch_max:
                pitch_max = pmax

            emin = np.min(energy)
            if emin < energy_min:
                energy_min = emin
            emax = np.max(energy)
            if emax > energy_max:
                energy_max = emax


        with open(os.path.join(out_dir, "stats.json"), "w") as f:
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

        print (f"extra puncts : {self._extra_puncts}")

def gen_jobs_from_metadata_file(in_dir, metadata_path, book=None):

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

                jobs.append({'text': text,
                             'wav_path': wav_path,
                             'dest_wav': f"{dest_base_name}.wav",
                             })

    print (f"{metadata_path} -> {len(jobs)} jobs")

    return jobs

def gather_jobs_from_config(config, limit: int):

    if "LJSpeech" not in config["dataset"]:
        raise Exception (f"unknown dataset format '{config['dataset']}")

    in_dir = config["path"]["corpus_path"]

    out_dir = config["path"]["preprocessed_path"]

    shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs((os.path.join(out_dir, "wavs")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "mel")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "pitch")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "energy")), exist_ok=True)
    os.makedirs((os.path.join(out_dir, "duration")), exist_ok=True)
    # os.makedirs((os.path.join(out_dir, "phonemepos")), exist_ok=True)

    # auto-detect corpus format: single or multi-book?

    metadata_path = os.path.join(in_dir, "metadata.csv")

    if os.path.isfile(metadata_path):

        jobs = gen_jobs_from_metadata_file (in_dir=in_dir, metadata_path=metadata_path)

    else:

        jobs = []

        for book in os.listdir(in_dir):

            bookdir = os.path.join(in_dir, book)

            metadata_path = os.path.join(bookdir, 'metadata.csv')

            if os.path.isfile(metadata_path):
                jobs.extend(gen_jobs_from_metadata_file (in_dir=bookdir, metadata_path=metadata_path, book=book))

    if limit:
        jobs = jobs[:limit]

    return jobs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("modelcfg", type=str, help="model config to preprocess for")
    parser.add_argument("corpora", type=str, nargs='+', help="path[s] to corpus .yaml config file[s] or directorie[s]")
    parser.add_argument("-l", "--limit", type=int, default=1000, help="limit number auf audio files to process per config, default: 1000 (0=unlimited)")
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

    pproc = Preprocessor (modelcfg)

    for cfg in corpus_configs:

        jobs = gather_jobs_from_config (cfg, limit=args.limit)

        print(f"gathered {len(jobs)} jobs.")

        pproc.process (jobs, out_dir = cfg["path"]["preprocessed_path"], lang=cfg['language'])


