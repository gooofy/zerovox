
import numpy as np
import librosa

# import torch
# import scipy

# import numpy as np

# def window_sumsquare(
#     window,
#     n_frames,
#     hop_length,
#     win_length,
#     n_fft,
#     dtype=np.float32,
#     norm=None,
# ):
#     """
#     # from librosa 0.6
#     Compute the sum-square envelope of a window function at a given hop length.

#     This is used to estimate modulation effects induced by windowing
#     observations in short-time fourier transforms.

#     Parameters
#     ----------
#     window : string, tuple, number, callable, or list-like
#         Window specification, as in `get_window`

#     n_frames : int > 0
#         The number of analysis frames

#     hop_length : int > 0
#         The number of samples to advance between frames

#     win_length : [optional]
#         The length of the window function.  By default, this matches `n_fft`.

#     n_fft : int > 0
#         The length of each analysis frame.

#     dtype : np.dtype
#         The data type of the output

#     Returns
#     -------
#     wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
#         The sum-squared envelope of the window function
#     """
#     if win_length is None:
#         win_length = n_fft

#     n = n_fft + hop_length * (n_frames - 1)
#     x = np.zeros(n, dtype=dtype)

#     # Compute the squared window at the desired length
#     win_sq = scipy.signal.get_window(window, win_length, fftbins=True)
#     win_sq = librosa.util.normalize(win_sq, norm=norm) ** 2
#     win_sq = librosa.util.pad_center(win_sq, n_fft)

#     # Fill the envelope
#     for i in range(n_frames):
#         sample = i * hop_length
#         x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
#     return x

# def dynamic_range_compression(x, C=1, clip_val=1e-5):
#     """
#     PARAMS
#     ------
#     C: compression factor
#     """
#     return torch.log(torch.clamp(x, min=clip_val) * C)


# def dynamic_range_decompression(x, C=1):
#     """
#     PARAMS
#     ------
#     C: compression factor used to compress
#     """
#     return torch.exp(x) / C

# class STFT(torch.nn.Module):
#     """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

#     def __init__(self, filter_length, hop_length, win_length, window="hann", use_cuda=False):
#         super(STFT, self).__init__()
#         self.filter_length = filter_length
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.window = window
#         self.use_cuda = use_cuda
#         self.forward_transform = None
#         scale = self.filter_length / self.hop_length
#         fourier_basis = np.fft.fft(np.eye(self.filter_length))

#         cutoff = int((self.filter_length / 2 + 1))
#         fourier_basis = np.vstack(
#             [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
#         )

#         forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
#         inverse_basis = torch.FloatTensor(
#             np.linalg.pinv(scale * fourier_basis).T[:, None, :]
#         )

#         if window is not None:
#             assert filter_length >= win_length
#             # get window and zero center pad it to filter_length
#             fft_window = scipy.signal.get_window(window, win_length, fftbins=True)
#             # fft_window = pad_center(fft_window, filter_length)
#             fft_window = librosa.util.pad_center(fft_window, size=filter_length)
#             fft_window = torch.from_numpy(fft_window).float()

#             # window the bases
#             forward_basis *= fft_window
#             inverse_basis *= fft_window

#         self.register_buffer("forward_basis", forward_basis.float())
#         self.register_buffer("inverse_basis", inverse_basis.float())

#     def transform(self, input_data):
#         num_batches = input_data.size(0)
#         num_samples = input_data.size(1)

#         self.num_samples = num_samples

#         # similar to librosa, reflect-pad the input
#         input_data = input_data.view(num_batches, 1, num_samples)
#         input_data = torch.nn.functional.pad(
#             input_data.unsqueeze(1),
#             (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
#             mode="reflect",
#         )
#         input_data = input_data.squeeze(1)

#         forward_transform = torch.nn.functional.conv1d(
#             input_data.cuda() if self.use_cuda else input_data,
#             torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda() if self.use_cuda else torch.autograd.Variable(self.forward_basis, requires_grad=False),
#             stride=self.hop_length,
#             padding=0,
#         ).cpu()

#         cutoff = int((self.filter_length / 2) + 1)
#         real_part = forward_transform[:, :cutoff, :]
#         imag_part = forward_transform[:, cutoff:, :]

#         magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
#         phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

#         return magnitude, phase

#     def inverse(self, magnitude, phase):
#         recombine_magnitude_phase = torch.cat(
#             [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
#         )

#         inverse_transform = torch.nn.functional.conv_transpose1d(
#             recombine_magnitude_phase,
#             torch.autograd.Variable(self.inverse_basis, requires_grad=False),
#             stride=self.hop_length,
#             padding=0,
#         )

#         if self.window is not None:
#             window_sum = window_sumsquare(
#                 self.window,
#                 magnitude.size(-1),
#                 hop_length=self.hop_length,
#                 win_length=self.win_length,
#                 n_fft=self.filter_length,
#                 dtype=np.float32,
#             )
#             # remove modulation effects
#             approx_nonzero_indices = torch.from_numpy(
#                 np.where(window_sum > librosa.util.tiny(window_sum))[0]
#             )
#             window_sum = torch.autograd.Variable(
#                 torch.from_numpy(window_sum), requires_grad=False
#             )
#             window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
#             inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
#                 approx_nonzero_indices
#             ]

#             # scale by hop ratio
#             inverse_transform *= float(self.filter_length) / self.hop_length

#         inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
#         inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

#         return inverse_transform

#     def forward(self, input_data):
#         self.magnitude, self.phase = self.transform(input_data)
#         reconstruction = self.inverse(self.magnitude, self.phase)
#         return reconstruction

# class TacotronSTFT(torch.nn.Module):

#     def __init__(
#         self,
#         filter_length,
#         hop_length,
#         win_length,
#         n_mel_channels,
#         sampling_rate,
#         mel_fmin,
#         mel_fmax,
#         use_cuda
#     ):
#         super(TacotronSTFT, self).__init__()
#         self.n_mel_channels = n_mel_channels
#         self.sampling_rate = sampling_rate
#         self.stft_fn = STFT(filter_length, hop_length, win_length, use_cuda=use_cuda)
#         mel_basis = librosa.filters.mel(
#             sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
#         )
#         mel_basis = torch.from_numpy(mel_basis).float()
#         self.register_buffer("mel_basis", mel_basis)

#     def spectral_normalize(self, magnitudes):
#         output = dynamic_range_compression(magnitudes)
#         return output

#     def spectral_de_normalize(self, magnitudes):
#         output = dynamic_range_decompression(magnitudes)
#         return output

#     def mel_spectrogram(self, y):
#         """Computes mel-spectrograms from a batch of waves
#         PARAMS
#         ------
#         y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

#         RETURNS
#         -------
#         mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
#         """
#         assert torch.min(y.data) >= -1
#         assert torch.max(y.data) <= 1

#         magnitudes, phases = self.stft_fn.transform(y)
#         magnitudes = magnitudes.data
#         mel_output = torch.matmul(self.mel_basis, magnitudes)
#         mel_output = self.spectral_normalize(mel_output)
#         energy = torch.norm(magnitudes, dim=1)

#         return mel_output, energy


# def get_mel_from_wav(audio,
#                      sampling_rate,
#                      fft_size, # =1024,
#                      hop_size, # =256,
#                      win_length, # =None,
#                      window, # ="hann",
#                      num_mels, #=80,
#                      fmin, #=None,
#                      fmax, #=None,
#                      eps, #=1e-10,
#                      log_base, #=10.0,
#                      stft
#                      ):

#     mel = logmelfilterbank(
#         audio=audio,
#         sampling_rate=sampling_rate,
#         fft_size=fft_size,
#         hop_size=hop_size,
#         win_length=win_length,
#         window=window,
#         num_mels=num_mels,
#         fmin=fmin,
#         fmax=fmax,
#         eps=eps,
#         log_base=log_base
#     )

#     # energy computation

#     audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
#     audio = torch.autograd.Variable(audio, requires_grad=False)

#     mel2, energy = stft.mel_spectrogram(audio)

#     energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

#     return mel.T, energy

# mel_basis = None
# hann_window = None
# mel_fmax = None

# def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
#     return torch.log(torch.clamp(x, min=clip_val) * C)

# def spectral_normalize_torch(magnitudes):
#     output = dynamic_range_compression_torch(magnitudes)
#     return output

# def get_mel_from_wav(audio,
#                      sampling_rate,
#                      fft_size, # =1024,
#                      hop_size, # =256,
#                      win_length, # =None,
#                      num_mels, #=80,
#                      fmin, #=None,
#                      fmax, #=None,
#                      ):

#     audio = torch.FloatTensor(audio).unsqueeze(0)

#     if torch.min(audio) < -1.:
#         print(f"WARNING: get_mel_from_wav: audio min value < -1.0 : {torch.min(audio)}")
#     if torch.max(audio) > 1.:
#         print(f"WARNING: get_mel_from_wav: audio max value >  1.0 : {torch.max(audio)}")

#     global mel_basis, hann_window, mel_fmax
#     if mel_basis is None:
#         mel = librosa.filters.mel (sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
#         mel_basis = torch.from_numpy(mel).float().to(audio.device)
#         hann_window = torch.hann_window(win_length).to(audio.device)
#         mel_fmax = fmax
#     else:
#         assert mel_fmax == fmax

#     y = torch.nn.functional.pad(audio.unsqueeze(1), (int((fft_size-hop_size)/2), int((fft_size-hop_size)/2)), mode='reflect')
#     y = y.squeeze(1)

#     magnitudes = torch.stft(y, fft_size, hop_length=hop_size, win_length=win_length, window=hann_window,
#                             center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

#     magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)+(1e-9))

#     spec = torch.matmul(mel_basis, magnitudes)
#     spec = spectral_normalize_torch(spec)

#     energy = torch.norm(magnitudes, dim=1)

#     return spec.squeeze(0).transpose(0,1), energy.squeeze(0)


mel_basis = None
#hann_window = None
mel_fmax = None

def dynamic_range_compression_numpy(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def spectral_normalize_numpy(magnitudes):
    output = dynamic_range_compression_numpy(magnitudes)
    return output

def get_mel_from_wav(audio,
                     sampling_rate,
                     fft_size, # =1024,
                     hop_size, # =256,
                     win_length, # =None,
                     num_mels, #=80,
                     fmin, #=None,
                     fmax, #=None,
                     ):

    # audio = np.array(audio, dtype=np.float32)

    if np.min(audio) < -1.:
        print(f"WARNING: get_mel_from_wav: audio min value < -1.0 : {np.min(audio)}")
    if np.max(audio) > 1.:
        print(f"WARNING: get_mel_from_wav: audio max value >  1.0 : {np.max(audio)}")

    global mel_basis, mel_fmax
    if mel_basis is None:
        mel_basis = librosa.filters.mel (sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
        #hann_window = librosa.util.hann_window(win_length, dtype=np.float32)
        #hann_window = librosa.filters.get_window(window='hann', Nx=win_length, fftbins=True)
        mel_fmax = fmax
    else:
        assert mel_fmax == fmax

    padding = (fft_size - hop_size) // 2
    audio_padded = np.pad(audio, (padding, padding), mode='reflect')

    stft_result = librosa.stft(audio_padded, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window='hann', center=False)

    magnitudes = np.abs(stft_result)

    spec = np.dot(mel_basis, magnitudes)
    spec = spectral_normalize_numpy(spec)

    energy = np.linalg.norm(magnitudes, axis=0)

    return spec, energy