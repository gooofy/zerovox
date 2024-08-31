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
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
import math
import time
import traceback
from pathlib import Path

from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from scipy.io import wavfile

import zerovox.hifigan
from zerovox.g2p.data import G2PSymbols
from zerovox.tts.networks import PhonemeEncoder, MelDecoder
from zerovox.tts.GST import GST

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

DEFAULT_HIFIGAN_MODEL_NAME = "zerovox-hifigan-vctk-v2-en-1"

def get_hifigan(model: str|os.PathLike, infer_device=None, verbose=False):

    if os.path.isdir(model):

        json_path = Path(Path(model) / 'config.json')
        gen_path  = Path(Path(model) / 'generator.ckpt')

    else:

        json_path = download_model_file(model=model, relpath="config.json")
        gen_path  = download_model_file(model=model, relpath="generator.ckpt")

    # get the main path
    if verbose:
        print("Using config: ", json_path)
        print("Using hifigan checkpoint: ", gen_path)
    with open(json_path, "r") as f:
        config = json.load(f)

    config = zerovox.hifigan.AttrDict(config)
    torch.manual_seed(config.seed)
    vocoder = zerovox.hifigan.Generator(config)
    if infer_device is not None:
        vocoder.to(infer_device)
        ckpt = torch.load(gen_path, map_location=torch.device(infer_device))
    else:
        ckpt = torch.load(gen_path)
        
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    for p in vocoder.parameters():
        p.requires_grad = False

    return vocoder

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
        return {}

    def load_state_dict(self, state_dict):
        pass

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
                 hifigan_model,
                 sampling_rate,
                 hop_length,
                 n_mels,
                 lr=1e-3,
                 weight_decay=1e-6, 
                 max_epochs=5000,
                 warmup_epochs=50,
                 encoder_depth=2,
                 decoder_n_blocks=2,
                 decoder_block_depth=2,
                 decoder_x2_fix=True,
                 reduction=4, 
                 encoder_n_heads=1,
                 embed_dim=128, 
                 encoder_kernel_size=3, 
                 decoder_kernel_size=3,
                 encoder_expansion=1,
                 wav_path="wavs", 
                 infer_device=None, 
                 verbose=False,
                 punct_embed_dim=16,
                 gst_n_style_tokens=10,
                 gst_n_heads=8,
                 gst_ref_enc_filters=[32, 32, 64, 64, 128, 128]):
        super(ZeroVox, self).__init__()

        self.save_hyperparameters()

        self._phoneme_encoder = PhonemeEncoder(symbols=symbols,
                                               stats=stats,
                                               depth=encoder_depth,
                                               reduction=reduction,
                                               head=encoder_n_heads,
                                               embed_dim=embed_dim,
                                               kernel_size=encoder_kernel_size,
                                               expansion=encoder_expansion,
                                               punct_embed_dim=punct_embed_dim)

        emb_size = (embed_dim+punct_embed_dim)//reduction

        self._gst = GST(emb_size, n_mels, gst_n_style_tokens, gst_n_heads, gst_ref_enc_filters)

        self._mel_decoder = MelDecoder(dim=emb_size, 
                                       kernel_size=decoder_kernel_size,
                                       n_blocks=decoder_n_blocks, 
                                       block_depth=decoder_block_depth,
                                       x2_fix=decoder_x2_fix)

        self.hifigan = get_hifigan(model=hifigan_model,
                                   infer_device=infer_device, verbose=verbose)

        self.training_step_outputs = []
        self.validation_step_outputs = []

        # self._min_mel_len = 1500 # 1500 * 256 / 22050 -> 17.4s
        self._min_mel_len = 689 # 689 * 256 / 22050 -> 8s
        self._hop_length = hop_length


    def forward(self, x, force_duration=False):

        style_embed = self._gst(x["ref_mel"])

        pred = self._phoneme_encoder(x, style_embed=style_embed, train=self.training, force_duration=force_duration)

        mel = self._mel_decoder(pred["features"], style_embed) 
        
        mask = pred["masks"]
        if mask is not None and mel.size(0) > 1:
            mask = mask[:, :, :mel.shape[-1]]
            mel = mel.masked_fill(mask, 0)
        
        pred["mel"] = mel

        if self.training:
            return pred

        mel_len  = pred["mel_len"]
        duration = pred["duration"]

        mel = mel.transpose(1, 2)
        wav = self.hifigan(mel).squeeze(1)
        
        return wav, mel, mel_len, duration

    def inference(self, x, style_embed):

        #start_time = time.time()

        pred = self._phoneme_encoder(x, style_embed=style_embed, train=False)

        #pe_time = time.time()

        mel = self._mel_decoder(features=pred["features"], style_embed=style_embed) 
        
        #dec_time = time.time()

        mel_len  = int(pred["mel_len"].cpu().detach().numpy())
        duration = pred["duration"]

        mel = mel.transpose(1, 2)

        # try to keep mel size constant to reduce hifigan latency (no torch/coda recompile on every utterance)
        if mel_len < self._min_mel_len:
            mel = torch.nn.functional.pad(mel, (0, self._min_mel_len - mel_len))
        elif mel_len > self._min_mel_len:
            self._min_mel_len = mel_len

        wav = self.hifigan(mel).squeeze()

        #hifigan_time = time.time()

        #print (f"phoneme_encoder: {pe_time-start_time}s, mel_decoder: {dec_time-pe_time}, hifigan: {hifigan_time-dec_time}")

        return wav[:mel_len * self._hop_length], mel_len, duration


    def loss(self, y_hat, y, x):
        pitch_pred = y_hat["pitch"]
        energy_pred = y_hat["energy"]
        duration_pred = y_hat["duration"]
        mel_pred = y_hat["mel"]

        phoneme_mask = x["phoneme_mask"]
        mel_mask = x["mel_mask"]

        pitch = x["pitch"]
        energy = x["energy"]
        duration = x["duration"]
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

        duration_pred = duration_pred[:,:duration.shape[-1]]
        duration_pred = torch.squeeze(duration_pred)
        duration      = duration.masked_select(phoneme_mask)
        duration_pred = duration_pred.masked_select(phoneme_mask)
        duration      = torch.log(duration.float() + 1)
        duration_pred = torch.log(duration_pred.float() + 1)
        duration_loss = nn.MSELoss()(duration_pred, duration)

        return mel_loss, pitch_loss, energy_loss, duration_loss
 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(y_hat, y, x)
        loss = (10. * mel_loss) + (2. * pitch_loss) + (2. * energy_loss) + duration_loss
        
        losses = {"loss": loss, 
                  "mel_loss": mel_loss, 
                  "pitch_loss": pitch_loss,
                  "energy_loss": energy_loss, 
                  "duration_loss": duration_loss}
        self.training_step_outputs.append(losses)
        
        return loss


    def on_train_epoch_end(self):
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
                x, y = batch
                wavs, mel_pred, len_pred, _ = self.forward(x)
                wavs = wavs.to(torch.float).cpu().numpy()
                write_to_file(wavs, self.hparams.sampling_rate, self.hparams.hop_length, lengths=len_pred.cpu().numpy(), \
                    wav_path=self.hparams.wav_path, filename=f"prediction-{batch_idx}")

                mel_target = y["mel"]
                with torch.no_grad():
                    wavs = self.hifigan(mel_target.transpose(1, 2)).squeeze(1)
                    wavs = wavs.to(torch.float).cpu().numpy()

                len_target = x["mel_len"]
                write_to_file(wavs, self.hparams.sampling_rate, self.hparams.hop_length, lengths=len_target.cpu().numpy(),\
                        wav_path=self.hparams.wav_path, filename=f"reconstruction-{batch_idx}")

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

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        self.scheduler = LinearWarmUpCosineDecayLR (optimizer,
                                                    base_lr=self.hparams.lr,
                                                    min_lr=0.1,
                                                    warmup_epochs=self.hparams.warmup_epochs,
                                                    total_epochs=self.hparams.max_epochs)

        return [optimizer], [self.scheduler]
