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

import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LRScheduler
import math
import time
import traceback
import psutil
import gc
from pathlib import Path

from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from scipy.io import wavfile

from zerovox.g2p.data import G2PSymbols
from zerovox.tts.networks import PhonemeEncoder
from zerovox.tts.GST import GST
from zerovox.tts.ResNetSE34V2 import ResNetSE34V2
from zerovox.tts.fs2 import FS2Encoder, FS2Decoder
from zerovox.tts.postnet import PostNet

from zerovox.parallel_wavegan.utils import load_model as load_meldec_model

def write_to_file(wavs, sampling_rate, hop_length, lengths=None, wav_path="outputs", filename="tts"):
    wavs = (wavs * 32760).astype("int16")
    wavs = [wav for wav in wavs]
    if lengths is not None:
        lengths *= hop_length
        for i in range(len(wavs)):
            wavs[i] = wavs[i][: lengths[i]]
            
    # create dir if not exists
    os.makedirs(wav_path, exist_ok=True)
    if len(wavs) == 1:
        path = os.path.join(wav_path, filename)
        print("Writing wav to {}".format(path))
        wavfile.write(path, sampling_rate, wavs[0])
    else:
        for i, wav in enumerate(wavs):
            path = os.path.join(wav_path, "{}-{}.wav".format(filename, i+1))
            wavfile.write(path, sampling_rate, wav)
    
    return wavs, sampling_rate


def download_model_file(model:str, relpath:str) -> Path:

    cache_path = Path(os.getenv("CACHED_PATH_ZEROVOX", Path.home() / ".cache" / "zerovox"))

    target_dir  = cache_path / "model_repo" / model
    target_path = target_dir / relpath

    if target_path.exists():
        return target_path

    os.makedirs (target_dir, exist_ok=True)

    url = f"https://huggingface.co/goooofy/{model}/resolve/main/{relpath}?download=true"

    torch.hub.download_url_to_file(url, str(target_path))

    return target_path

DEFAULT_MELDEC_MODEL_NAME = "meldec-libritts-multi-band-melgan-v2"
#DEFAULT_MELDEC_MODEL_NAME = "meldec-libritts-hifigan-v1"

def get_meldec(model: str|os.PathLike, infer_device=None, verbose=False):

    if os.path.isdir(model):

        config_path = Path(Path(model) / 'config.yml')
        gen_path  = Path(Path(model) / 'checkpoint.pkl')
        stats_path  = Path(Path(model) / 'stats.h5')

    else:

        config_path = download_model_file(model=model, relpath="config.yml")
        gen_path  = download_model_file(model=model, relpath="checkpoint.pkl")
        stats_path  = download_model_file(model=model, relpath="stats.h5")

    # get the main path
    if verbose:
        print("meldec: using config    : ", config_path)
        print("meldec: using checkpoint: ", gen_path)
        print("meldec: using stats     : ", stats_path)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    model = load_meldec_model(gen_path, config)
    if verbose:
        print(f"meldec: loaded model parameters from {gen_path}.")

    model.remove_weight_norm()

    device = torch.device(infer_device)
    model = model.eval().to(device)
    model.to(device)

    if config["generator_params"]["out_channels"] == 1:
        model.pqmf = None

    return model

class LinearWarmUpCosineDecayLR(LRScheduler):
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        min_lr: float,
        warmup_epochs: int,
        total_epochs: int
    ):  
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr  = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        super().__init__(optimizer)

    def state_dict(self):
        return {'epoch': self.last_epoch}

    def load_state_dict(self, state_dict):
        #pass
        if 'epoch' in state_dict:
            self.last_epoch=state_dict['epoch']

    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up
            f = float(self.last_epoch+1) / float(self.warmup_epochs)
        else:
            # Cosine learning rate decay
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            f = max(self.min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return [ self.base_lr * f ]

class ZeroVox(LightningModule):
    def __init__(self,
                 symbols: G2PSymbols,
                 stats, 
                 meldec_model,
                 sampling_rate,
                 hop_length,
                 n_mels,
                 lr, #=1e-3,
                 weight_decay, #=1e-6, 
                 max_epochs, #=5000,
                 warmup_epochs, #=50,

                 embed_dim, #=128, 
                 punct_embed_dim, #=16,
                 dpe_embed_dim, #=64,
                 emb_reduction, #=1,
                 max_seq_len, #=1000,

                 encoder_kind, #="efficientspeech",
                 encoder_depth, #=2,
                 encoder_n_heads, #=1,
                 encoder_kernel_size, #=3, 
                 encoder_expansion, #=1,

                 fs2enc_layer, # 4
                 fs2enc_head, # 2
                 fs2enc_dropout, # 0.2
                 vp_filter_size, # 256
                 vp_kernel_size, # 3
                 vp_dropout, # 0.5
                 ve_pitch_quantization, # 'linear'
                 ve_energy_quantization, # 'linear'
                 ve_n_bins, # 256

                 spkemb_kind, # 'GST' or 'ResNetSE34V2'
                 gst_n_style_tokens, #=2000,
                 gst_n_heads, #=8,
                 gst_ref_enc_filters, #=[32, 32, 64, 64, 128, 128],
                 resnet_layers, #=[3, 4, 6, 3]
                 resnet_num_filters, #=[32, 64, 128, 256]
                 resnet_encoder_type, #='ASP' or 'SAP'

                 decoder_n_layers, #=6,
                 decoder_n_head, #=2,
                 decoder_conv_filter_size, #=1024,
                 decoder_conv_kernel_size, #=[9, 1],
                 decoder_dropout, #=0.2,
                 decoder_scln, #=True,

                 postnet_embedding_dim, #=512, # 0 to disable
                 postnet_kernel_size, #=5,
                 postnet_n_convolutions, #=5,

                 wav_path="wavs", 
                 infer_device=None, 
                 verbose=False,
                 ):
        super(ZeroVox, self).__init__()

        self.save_hyperparameters(ignore=['meldec_model', 'infer_device', 'verbose'])

        if encoder_kind == 'efficientspeech':

            self._phoneme_encoder = PhonemeEncoder(symbols=symbols,
                                                   stats=stats,
                                                   depth=encoder_depth,
                                                   reduction=emb_reduction,
                                                   head=encoder_n_heads,
                                                   embed_dim=embed_dim,
                                                   kernel_size=encoder_kernel_size,
                                                   expansion=encoder_expansion,
                                                   punct_embed_dim=punct_embed_dim,
                                                   dpe_embed_dim=dpe_embed_dim)

            emb_size = (embed_dim+punct_embed_dim)//emb_reduction
            dec_hidden = emb_size+3*dpe_embed_dim

        elif encoder_kind == 'fastspeech2':

            self._phoneme_encoder = FS2Encoder(symbols=symbols,
                                               max_seq_len=max_seq_len,
                                               embed_dim=embed_dim,
                                               encoder_layer=fs2enc_layer,
                                               encoder_head=fs2enc_head,
                                               conv_filter_size=decoder_conv_filter_size,
                                               conv_kernel_size=decoder_conv_kernel_size,
                                               encoder_dropout=fs2enc_dropout,
                                               punct_embed_dim=punct_embed_dim,
                                               vp_filter_size=vp_filter_size,
                                               vp_kernel_size=vp_kernel_size,
                                               vp_dropout=vp_dropout,
                                               ve_pitch_quantization=ve_pitch_quantization,
                                               ve_energy_quantization=ve_energy_quantization,
                                               ve_n_bins=ve_n_bins,
                                               stats=stats)

            emb_size = embed_dim+punct_embed_dim
            dec_hidden = emb_size

        else:
            raise Exception (f"unknown encoder kind: '{encoder_kind}'")

        if spkemb_kind == 'GST':
            self._spkemb = GST(emb_size, n_mels, gst_n_style_tokens, gst_n_heads, gst_ref_enc_filters)
        elif spkemb_kind == 'ResNetSE34V2':
            self._spkemb = ResNetSE34V2(layers=resnet_layers, num_filters=resnet_num_filters, nOut=emb_size, encoder_type=resnet_encoder_type, n_mels=n_mels, log_input=False)
        else:
            raise Exception (f"unknown speaker embedding kind: '{spkemb_kind}'")

        self._mel_decoder = FS2Decoder(dec_max_seq_len=max_seq_len,
                                       dec_hidden = dec_hidden,
                                       dec_n_layers = decoder_n_layers,
                                       dec_n_head = decoder_n_head,
                                       dec_conv_filter_size = decoder_conv_filter_size,
                                       dec_conv_kernel_size = decoder_conv_kernel_size,
                                       dec_dropout = decoder_dropout,
                                       dec_scln = decoder_scln,
                                       n_mel_channels=n_mels,
                                       spk_emb_size=emb_size)

        # FIXME
        # self._fake_mel_decoder = torch.nn.Linear(emb_size+3*dpe_embed_dim, n_mels)

        if postnet_embedding_dim:
            self._postnet = PostNet(
                                 n_mel_channels=n_mels,
                                 postnet_embedding_dim=postnet_embedding_dim,
                                 postnet_kernel_size=postnet_kernel_size,
                                 postnet_n_convolutions=postnet_n_convolutions,
            )
        else:
            self._postnet = None

        if meldec_model:
            self._meldec = get_meldec(model=meldec_model, infer_device=infer_device, verbose=verbose)
        else:
            self._meldec = None

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self._min_mel_len = 689 # 689 * 256 / 24000 -> 7s
        self._hop_length = hop_length

        self._verbose = verbose


    def forward(self, x, force_duration=False, normalize_before=True):

        # [bs, 1, 528]             [bs, 440, 80]
        style_embed = self._spkemb(x["ref_mel"])

        pred = self._phoneme_encoder(x, style_embed=style_embed, train=self.training, force_duration=force_duration)

        mask = pred["masks"]

        if mask is None:
            max_len = pred["features"].shape[1] # pred["mel_len"].cpu().max().item()
            range_tensor = torch.arange(max_len).expand(len(pred["mel_len"]), max_len).to(device=pred['mel_len'].device)
            dec_mask = range_tensor < pred["mel_len"].unsqueeze(1)
            dec_mask = ~dec_mask
        else:
            dec_mask = mask[:,:,0]

        # FIXME
        # pred["features"].shape torch.Size([8, 1221, 240])
        # mel.shape torch.Size([8, 1221, 80])

        mel, dec_mask = self._mel_decoder(pred["features"], dec_mask, spk_emb=style_embed) 
        
        if mask is not None and mel.size(0) > 1:
            mask = mask[:, :, :mel.shape[-1]]
            mel = mel.masked_fill(mask, 0)

        # FIXME
        # mel = self._fake_mel_decoder(pred["features"])

        if self._postnet:
            mel = self._postnet(mel) + mel

        pred["mel"] = mel

        if self.training:
            return pred

        mel_len  = pred["mel_len"]
        log_duration = pred["log_duration"]

        if normalize_before:
            mel = (mel - self._meldec.mean) / self._meldec.scale
        mel = mel.transpose(1, 2)
        wav = self._meldec(c=mel)
        if self._meldec.pqmf is not None:
            wav = self._meldec.pqmf.synthesis(wav)
        wav = wav.squeeze(1)

        return wav, mel, mel_len, log_duration

    def inference(self, x, style_embed, normalize_before=True):

        start_time = time.time()

        pred = self._phoneme_encoder(x, style_embed=style_embed, train=False)

        pe_time = time.time()

        max_len = pred["features"].shape[1] # pred["mel_len"].cpu().max().item()
        range_tensor = torch.arange(max_len).expand(len(pred["mel_len"]), max_len).to(device=pred['mel_len'].device)
        dec_mask = range_tensor < pred["mel_len"].unsqueeze(1)
        dec_mask = ~dec_mask

        mel, dec_mask = self._mel_decoder(pred["features"], dec_mask, spk_emb=style_embed) 
        
        dec_time = time.time()

        mel_len  = int(pred["mel_len"].cpu().detach().numpy())
        log_duration = pred["log_duration"]

        if normalize_before:
            mel = (mel - self._meldec.mean) / self._meldec.scale
        mel = mel.transpose(1, 2)

        # try to keep mel size constant to reduce meldec latency (no torch/coda recompile on every utterance)
        if mel_len < self._min_mel_len:
            mel = torch.nn.functional.pad(mel, (0, self._min_mel_len - mel_len))
        elif mel_len > self._min_mel_len:
            self._min_mel_len = mel_len

        wav = self._meldec(c=mel)
        if self._meldec.pqmf is not None:
            wav = self._meldec.pqmf.synthesis(wav)
        wav = wav.squeeze((0,1))

        meldec_time = time.time()

        if self._verbose:
            print (f"synthesis timing stats: pe={pe_time-start_time}s, dec={dec_time-pe_time}s, meldec={meldec_time-dec_time}s")

        return wav[:mel_len * self._hop_length], mel_len, log_duration


    def loss(self, y_hat, y, x):
        pitch_pred = y_hat["pitch"] # [16, 180, 1]
        energy_pred = y_hat["energy"] # [16, 180, 1]
        log_duration_pred = y_hat["log_duration"] # [16, 180]
        mel_pred = y_hat["mel"]

        phoneme_mask = x["phoneme_mask"]
        mel_mask = x["mel_mask"]

        pitch = x["pitch"]
        energy = x["energy"]
        duration_targets = x["duration"]
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel = y["mel"]

        mel_mask = ~mel_mask
        mel_mask = mel_mask.unsqueeze(-1)
        target = mel.masked_select(mel_mask)
        pred = mel_pred.masked_select(mel_mask)
        mel_loss = nn.L1Loss()(pred, target)
    
        phoneme_mask = ~phoneme_mask

        pitch_pred = pitch_pred[:,:pitch.shape[-1]]
        pitch_pred = torch.squeeze(pitch_pred)
        pitch = pitch.masked_select(phoneme_mask)
        pitch_pred = pitch_pred.masked_select(phoneme_mask)
        pitch_loss = nn.MSELoss()(pitch_pred, pitch)

        energy_pred = energy_pred[:,:energy.shape[-1]]
        energy_pred = torch.squeeze(energy_pred)
        energy      = energy.masked_select(phoneme_mask)
        energy_pred = energy_pred.masked_select(phoneme_mask)
        energy_loss = nn.MSELoss()(energy_pred, energy)

        # duration_pred = duration_pred[:,:duration.shape[-1]]
        # duration_pred = torch.squeeze(duration_pred)
        # duration      = duration.masked_select(phoneme_mask)
        # duration_pred = duration_pred.masked_select(phoneme_mask)
        # duration      = torch.log(duration.float() + 1)
        # duration_pred = torch.log(duration_pred.float() + 1)
        # duration_loss = nn.MSELoss()(duration_pred, duration)

        log_duration_pred    = log_duration_pred.masked_select(phoneme_mask)
        log_duration_targets = log_duration_targets.masked_select(phoneme_mask)
        duration_loss = nn.MSELoss()(log_duration_pred, log_duration_targets)

        return mel_loss, pitch_loss, energy_loss, duration_loss
 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(y_hat, y, x)
        loss = (10. * mel_loss) + (2. * pitch_loss) + (2. * energy_loss) + duration_loss
        
        losses = {"loss": loss.detach(), 
                  "mel_loss": mel_loss.detach(), 
                  "pitch_loss": pitch_loss.detach(),
                  "energy_loss": energy_loss.detach(), 
                  "duration_loss": duration_loss.detach()}
        self.training_step_outputs.append(losses)
        
        return loss


    def on_train_epoch_end(self):

        gc.collect()
        process = psutil.Process(os.getpid())
        resident_size = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        print(f"on_train_epoch_end: resident size = {resident_size} MB")

        if not self.training_step_outputs:
            return
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_mel_loss = torch.stack([x["mel_loss"] for x in self.training_step_outputs]).mean()
        avg_pitch_loss = torch.stack([x["pitch_loss"] for x in self.training_step_outputs]).mean()
        avg_energy_loss = torch.stack(
            [x["energy_loss"] for x in self.training_step_outputs]).mean()
        avg_duration_loss = torch.stack(
            [x["duration_loss"] for x in self.training_step_outputs]).mean()
        self.log("mel", avg_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pitch", avg_pitch_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("energy", avg_energy_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("dur", avg_duration_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.training_step_outputs.clear()


    def validation_step(self, batch, batch_idx):

        try:
            # if batch_idx==0 and self.current_epoch>=1 :
            if self.current_epoch>=1 :
                with torch.no_grad():
                    x, y = batch
                    wavs, mel_pred, len_pred, _ = self.forward(x, force_duration=True)
                    wavs = wavs.to(torch.float).cpu().numpy()
                    write_to_file(wavs, self.hparams.sampling_rate, self.hparams.hop_length, lengths=len_pred.cpu().numpy(), \
                        wav_path=self.hparams.wav_path, filename=f"prediction-{batch_idx}")

                    mel_target = y["mel"]
                    #wavs = self._meldec(c=mel_target.transpose(1, 2), normalize_before=True).squeeze(1)

                    mel_target_scaled = (mel_target - self._meldec.mean) / self._meldec.scale
                    wavs = self._meldec(c=mel_target_scaled.transpose(1, 2))
                    if self._meldec.pqmf is not None:
                        wavs = self._meldec.pqmf.synthesis(wavs)
                    wavs = wavs.squeeze(1)

                    wavs = wavs.to(torch.float).cpu().numpy()

                    len_target = x["mel_len"]
                    write_to_file(wavs, self.hparams.sampling_rate, self.hparams.hop_length, lengths=len_target.cpu().numpy(), wav_path=self.hparams.wav_path, filename=f"reconstruction-{batch_idx}")

                    # write the text to be converted to file
                    path = os.path.join(self.hparams.wav_path, f"prediction-{batch_idx}.txt")
                    with open(path, "w") as f:
                        text = x["text"]
                        for i in range(len(text)):
                            f.write(text[i] + "\n")

                    # compute validation mel loss

                    max_len = torch.cat((len_pred, len_target)).cpu().max().item()
                    mel_pred = mel_pred.transpose(1,2)
                    mel_pred = torch.nn.functional.pad(input=mel_pred, pad=(0, 0, 0, max_len-mel_pred.shape[1], 0, 0), mode='constant', value=0)
                    mel_target = torch.nn.functional.pad(input=mel_target, pad=(0, 0, 0, max_len-mel_target.shape[1], 0, 0), mode='constant', value=0)

                    range_tensor = torch.arange(max_len).expand(len(len_target), max_len)

                    mask_pred    = range_tensor < len_pred.cpu().unsqueeze(1)
                    mask_target  = range_tensor < len_target.cpu().unsqueeze(1)

                    mask_pred   = ~mask_pred.unsqueeze(-1)
                    mask_target = ~mask_target.unsqueeze(-1)
                    target = mel_target.cpu().masked_fill(mask_target, 0)
                    pred = mel_pred.cpu().masked_fill(mask_pred, 0)
                    mel_loss = nn.L1Loss()(pred, target)

                    self.validation_step_outputs.append(mel_loss)

        except Exception as e:
            print ("*** validation failed (this is ok in early training steps), exception caught:")
            print (traceback.format_exc())

    def on_test_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        avg_mel_loss = torch.stack([mel_loss for mel_loss in self.validation_step_outputs]).mean()
        self.log("val_mel", avg_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    # def on_save_checkpoint(self, checkpoint):
    #     # remove MELGAN
    #     # print (checkpoint['state_dict'].keys())

    #     for key in list(checkpoint['state_dict'].keys()):
    #         if not key.startswith('_meldec.'):
    #             continue
    #         del checkpoint['state_dict'][key]

    #     #print (checkpoint['state_dict'].keys())



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        self.scheduler = LinearWarmUpCosineDecayLR (optimizer,
                                                    base_lr=self.hparams.lr,
                                                    min_lr=0.1,
                                                    warmup_epochs=self.hparams.warmup_epochs,
                                                    total_epochs=self.hparams.max_epochs)

        return [optimizer], [self.scheduler]
