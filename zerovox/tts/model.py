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
import math
from pathlib import Path

from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from scipy.io import wavfile

import zerovox.hifigan
from zerovox.g2p.data import G2PSymbols
from zerovox.tts.networks import PhonemeEncoder, MelDecoder, Phoneme2Mel

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

def get_hifigan(checkpoint : str, infer_device=None, verbose=False):

    json_path = download_model_file(model='zerovox-hifigan-vctk-1', relpath=checkpoint + "_config.json")
    gen_path  = download_model_file(model='zerovox-hifigan-vctk-1', relpath=checkpoint + "_generator")

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

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=0):
    """
    Create a learning rate scheduler with linear warm-up and cosine learning rate decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        warmup_steps (int): The number of warm-up steps.
        total_steps (int): The total number of steps.
        min_lr (float, optional): The minimum learning rate at the end of the decay. Default: 0.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warm-up
            f = float(current_epoch+1) / float(warmup_epochs)
        else:
            # Cosine learning rate decay
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            f = max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return f

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


class ZeroVox(LightningModule):
    def __init__(self,
                 symbols: G2PSymbols,
                 stats, 
                 hifigan_checkpoint,
                 sampling_rate,
                 hop_length,
                 lr=1e-3,
                 weight_decay=1e-6, 
                 max_epochs=5000,
                 warmup_epochs=50,
                 depth=2, 
                 n_blocks=2, 
                 block_depth=2, 
                 reduction=4, 
                 head=1,
                 embed_dim=128, 
                 kernel_size=3, 
                 decoder_kernel_size=3, 
                 expansion=1,
                 wav_path="wavs", 
                 infer_device=None, 
                 verbose=False,
                 punct_embed_dim=16,
                 speaker_embed_dim=192):
        super(ZeroVox, self).__init__()

        self.save_hyperparameters()

        phoneme_encoder = PhonemeEncoder(symbols=symbols,
                                         stats=stats,
                                         depth=depth,
                                         reduction=reduction,
                                         head=head,
                                         embed_dim=embed_dim,
                                         kernel_size=kernel_size,
                                         expansion=expansion,
                                         punct_embed_dim=punct_embed_dim,
                                         speaker_embed_dim=speaker_embed_dim)

        mel_decoder = MelDecoder(dim=(embed_dim+punct_embed_dim)//reduction+speaker_embed_dim, 
                                 kernel_size=decoder_kernel_size,
                                 n_blocks=n_blocks, 
                                 block_depth=block_depth)

        self.phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                                       decoder=mel_decoder)

        self.hifigan = get_hifigan(checkpoint=hifigan_checkpoint,
                                   infer_device=infer_device, verbose=verbose)

        self.training_step_outputs = []


    def forward(self, x):

        # return self.phoneme2mel(x, train=True) if self.training else self.predict_step(x)

        if self.training:

            return self.phoneme2mel(x, train=True)

        else:
            mel, mel_len, duration = self.phoneme2mel(x, train=False)
            mel = mel.transpose(1, 2)
            wav = self.hifigan(mel).squeeze(1)
            
            return wav, mel_len, duration

    # def predict_step(self, batch, batch_idx=0,  dataloader_idx=0):
    #     mel, mel_len, duration = self.phoneme2mel(batch, train=False)
    #     mel = mel.transpose(1, 2)
    #     wav = self.hifigan(mel).squeeze(1)
        
    #     return wav, mel_len, duration


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
        # TODO: use predict step for wav file generation

        # if batch_idx==0 and self.current_epoch>=1 :
        if self.current_epoch>=1 :
            x, y = batch
            wavs, lengths, _ = self.forward(x)
            wavs = wavs.to(torch.float).cpu().numpy()
            write_to_file(wavs, self.hparams.sampling_rate, self.hparams.hop_length, lengths=lengths.cpu().numpy(), \
                wav_path=self.hparams.wav_path, filename=f"prediction-{batch_idx}")

            mel = y["mel"]
            mel = mel.transpose(1, 2)
            lengths = x["mel_len"]
            with torch.no_grad():
                wavs = self.hifigan(mel).squeeze(1)
                wavs = wavs.to(torch.float).cpu().numpy()

            write_to_file(wavs, self.hparams.sampling_rate, self.hparams.hop_length, lengths=lengths.cpu().numpy(),\
                    wav_path=self.hparams.wav_path, filename=f"reconstruction-{batch_idx}")

            # write the text to be converted to file
            path = os.path.join(self.hparams.wav_path, f"prediction-{batch_idx}.txt")
            with open(path, "w") as f:
                text = x["text"]
                for i in range(len(text)):
                    f.write(text[i] + "\n")

    def on_test_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = get_lr_scheduler(optimizer, warmup_epochs=self.hparams.warmup_epochs, total_epochs=self.hparams.max_epochs, min_lr=0.1)

        return [optimizer], [self.scheduler]
