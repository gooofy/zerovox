#!/bin/env python3

import argparse
import scipy.signal
import yaml
import os
import random
import sys
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torchaudio
import librosa
import pyworld
import scipy

#import torchaudio
#from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.encoders import MelSpectrogramEncoder

from zerovox.g2p.g2p  import G2PTokenizer
from zerovox.g2p.data import G2PSymbols
from zerovox.lexicon  import Lexicon

def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec, energy


def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = scipy.signal.get_window(window, win_length, fftbins=True)
    win_sq = librosa.util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa.util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann", use_cuda=False):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.use_cuda = use_cuda
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = scipy.signal.get_window(window, win_length, fftbins=True)
            # fft_window = pad_center(fft_window, filter_length)
            fft_window = librosa.util.pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = torch.nn.functional.conv1d(
            input_data.cuda() if self.use_cuda else input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda() if self.use_cuda else torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = torch.nn.functional.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > librosa.util.tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction



class TacotronSTFT(torch.nn.Module):

    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
        use_cuda
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length, use_cuda=use_cuda)
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy


class Preprocessor:
    def __init__(self, config: dict[str, any], lexicon: Lexicon, tokenizer: G2PTokenizer, use_cuda: bool):

        np.seterr(all='raise')

        self._config        = config
        self._use_cuda      = use_cuda
        self._in_dir        = config["path"]["raw_path"]
        self._out_dir       = config["path"]["preprocessed_path"]
        self._val_size      = config["preprocessing"]["val_size"]
        self._sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self._hop_length    = config["preprocessing"]["stft"]["hop_length"]
        self._language      = config['preprocessing']['text']['language']
        self._speaker       = config['preprocessing']['text']['speaker']

        self._lexicon       = lexicon
        self._tokenizer     = tokenizer
        self._symbols       = G2PSymbols (self._lexicon.graphemes, self._lexicon.phonemes)

        self._spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb-mel-spec")

        self._pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self._energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self._stft = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
            use_cuda=use_cuda
        )

    def build_from_path(self):

        os.makedirs((os.path.join(self._out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self._out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self._out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self._out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self._out_dir, "spkemb")), exist_ok=True)

        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram

        for wav_name in tqdm(os.listdir(self._in_dir), desc=self._speaker):
            if ".wav" not in wav_name:
                continue

            basename = wav_name.split(".")[0]
            align_path = os.path.join(
                self._out_dir, "align", f"{basename}.tsv"
            )
            if os.path.exists(align_path):
                ret = self.process_utterance(basename)
                if ret is None:
                    continue
                else:
                    info, pitch, energy, n = ret
                out.append(info)
            else:
                continue

            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

            n_frames += n

        #print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self._pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self._energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self._out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self._out_dir, "energy"), energy_mean, energy_std
        )

        with open(os.path.join(self._out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self._hop_length / self._sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self._out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self._val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self._out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self._val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, basename):
        wav_path   = os.path.join(self._in_dir, f"{basename}.wav")
        text_path  = os.path.join(self._in_dir, f"{basename}.lab")
        align_path = os.path.join(self._out_dir, "align", f"{basename}.tsv")

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n").lower()

        # Get alignments, reconstruct punctuation from tokens
        alignment = []
        with open(align_path, 'r') as alignf:
            for line in alignf:
                parts = line.strip().split('\t')
                assert len(parts)==3
                alignment.append({'start': float(parts[0]), 'duration': float(parts[1]), 'phoneme': parts[2]})

        tokens = self._tokenizer.tokenize(raw_text)

        res = self.get_alignment(tokens, alignment)
        if not res:
            return None
        phones, puncts, phone_positions, durations, start, end = res
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self._sampling_rate * start) : int(self._sampling_rate * end)
        ].astype(np.float32)

        # Compute fundamental frequency
        pitch, t = pyworld.dio(
            wav.astype(np.float64),
            self._sampling_rate,
            frame_period=self._hop_length / self._sampling_rate * 1000,
        )
        pitch = pyworld.stonemask(wav.astype(np.float64), pitch, t, self._sampling_rate)

        # pitch = pitch[: sum(durations)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = get_mel_from_wav(wav, self._stft)
        mel_spectrogram = mel_spectrogram[:, : sum(durations)]
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
        for i, d in enumerate(durations):
            pos = phone_positions[i]
            if (d>0) and (pos+d < len(pitch)):
                phoneme_pitches[i] = np.mean(pitch[pos : pos + d])
            else:
                if pos < len(pitch):
                    phoneme_pitches[i] = pitch[pos]
                else:
                    phoneme_pitches[i] = pitch[-1]

        # compute mean energy per phoneme
        phoneme_energy = np.zeros(len(durations), dtype=pitch.dtype)
        for i, d in enumerate(durations):
            pos = phone_positions[i]
            if (d>0) and (pos+d < len(energy)):
                phoneme_energy[i] = np.mean(energy[pos : pos + d])
            else:
                if pos < len(energy):
                    phoneme_energy[i] = energy[pos]
                else:
                    phoneme_energy[i] = energy[-1]

        # compute speaker embedding
        signal, _ = torchaudio.load(wav_path)
        spkemb = self._spk_emb_encoder.encode_waveform(signal)[0][0]
        spkemb = spkemb.cpu().detach().numpy()

        # Save files
        dur_filename = f"duration-{basename}.npy"
        np.save(os.path.join(self._out_dir, "duration", dur_filename), durations)

        pitch_filename = f"pitch-{basename}.npy"
        np.save(os.path.join(self._out_dir, "pitch", pitch_filename), phoneme_pitches)

        energy_filename = f"energy-{basename}.npy"
        np.save(os.path.join(self._out_dir, "energy", energy_filename), phoneme_energy)

        mel_filename = f"mel-{basename}.npy"
        np.save(
            os.path.join(self._out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        spkemb_filename = f"spkemb-{basename}.npy"
        np.save(os.path.join(self._out_dir, "spkemb", spkemb_filename), spkemb)

        return (
            "|".join([basename, " ".join(phones), raw_text, " ".join(puncts)]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tokens : list[str], alignment : list[dict[str, any]]):

        if not tokens or not alignment:
            return None

        token_pos = 0
        alignment_pos = 0
   
        phones           = []
        puncts           = []
        phone_positions  = []
        durations        = []
        start_time       = -1
        end_time         = -1
        last_token_start = -1

        while True:

            cur_token = None
            punct = self._symbols.encode_punct(' ')
            while token_pos < len(tokens) and not cur_token:
                t = tokens[token_pos]
                if self._symbols.is_punct(t):
                    if t != ' ':
                        punct = self._symbols.encode_punct(t)

                #elif len(t)>0 and self._symbols.is_grapheme(t[0]):
                elif t in self._lexicon:
                    cur_token = t
                    break

                token_pos += 1

            #print (cur_token)
            if puncts:
                puncts[-1] = punct

            if not cur_token:
                break

            token_phones = self._lexicon[cur_token]
            for phone in token_phones:

                cur_align = alignment[alignment_pos]
                while (self._symbols.is_silence(cur_align['phoneme'])):
                    alignment_pos += 1
                    cur_align = alignment[alignment_pos]

                if phone != cur_align['phoneme']:
                    print (f"*** error: alignment async: {phone} vs {cur_align['phoneme']}")
                    print (f"   utt: {' '.join(tokens)}")
                    return None

                start = cur_align['start']
                stop = start + cur_align['duration']

                if start_time < 0:
                    start_time = start

                # now that we know when in time this token starts we can compute
                # the duration of the last phone
                # thereby adding any silences to the duration of the last phone
                if phones:
                    duration = start - last_token_start
                    durations.append(int(np.round(duration * self._sampling_rate / self._hop_length)))

                if stop > end_time:
                    end_time = stop

                phones.append(phone)
                puncts.append(punct)
                punct = self._symbols.encode_punct('')

                phone_positions.append(int(np.round((start-start_time) * self._sampling_rate / self._hop_length)))

                last_token_start = start

                alignment_pos += 1

            token_pos += 1

        durations.append(int(np.round(cur_align['duration'] * self._sampling_rate / self._hop_length)))
        #print (phones, puncts, durations)

        return phones, puncts, phone_positions, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs='+', help="path to preprocess.yamls")
    parser.add_argument("--cuda", action='store_true')
    # parser.add_argument("--num-jobs", type=int, default=12, help="number of jobs, default: 12")

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
    sys.stdout.flush()

    language  = None

    for cfgfn in cfgfns:

        config = yaml.load(open(cfgfn, "r"), Loader=yaml.FullLoader)

        if not language:
            language = config['preprocessing']['text']['language']
            lexicon = Lexicon.load(language)
            tokenizer = G2PTokenizer(language)
        else:
            if language != config['preprocessing']['text']['language']:
                print (f"inconsistent languages: {language} vs {config['preprocessing']['text']['language']} from {cfgfn}")
                sys.exit(1)

        pproc = Preprocessor (config, lexicon, tokenizer, args.cuda)

        pproc.build_from_path()


