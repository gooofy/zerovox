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

import torch
import torch.nn.functional as F
from torch import nn
from .blocks import MixFFN, SelfAttention
from zerovox.g2p.data import G2PSymbols

class Encoder(nn.Module):
    """ Phoneme Encoder """

    def __init__(self, symbols: G2PSymbols, depth=2, embed_dim=128, kernel_size=3, \
                 expansion=1, reduction=4, head=1, punct_embed_dim=16):
        super().__init__()

        small_embed_dim = (embed_dim+punct_embed_dim) // reduction
        dim_ins = [small_embed_dim*(2**i) for i in range(depth-1)]
        dim_ins.insert(0, embed_dim+punct_embed_dim)
        self.dim_outs = [small_embed_dim*2**i for i in range(depth)]
        heads = [head*(i+1) for i in range(depth)]
        kernels = [kernel_size-(2 if i > 0 else 0) for i in range(depth)]
        paddings = [k//2 for k in kernels]
        strides = [2 for _ in range(depth-1)]
        strides.insert(0, 1)
        
        self._symbols = symbols

        self.embed = nn.Embedding(self._symbols.num_phonemes + 1, embed_dim, padding_idx=0)

        self.punct_embed = nn.Embedding(self._symbols.num_puncts + 1, punct_embed_dim, padding_idx=0)

        self.attn_blocks = nn.ModuleList([])
        for dim_in, dim_out, head, kernel, stride, padding in zip(dim_ins, self.dim_outs,\
                                                                  heads, kernels, strides, paddings):
            self.attn_blocks.append(
                    nn.ModuleList([
                        #depthwise separable-like convolution
                        nn.Conv1d(dim_in, dim_in, kernel_size=kernel, stride=stride, \
                                  padding=padding, bias=False),
                        nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=False), 
                        SelfAttention(dim_out, num_heads=head),
                        MixFFN(dim_out, expansion),
                        nn.LayerNorm(dim_out),
                        nn.LayerNorm(dim_out),
                        ]))

    def get_feature_dims(self):
        return self.dim_outs

    def forward(self, phoneme, puncts, mask=None):
        features = []

        x = self.embed(phoneme)
        x_punct = self.punct_embed(puncts)
        x = torch.cat((x, x_punct), 2)
        # merge, attn and mixffn operates on n or seqlen dim
        # b = batch, n = sequence len, c = channel (1st layer is embedding)
        # (b, n, c)
        n = x.shape[-2]
        decoder_mask = None
        pool = 1

        for merge3x3, merge1x1, attn, mixffn, norm1, norm2 in self.attn_blocks:
            # after each encoder block, merge features
            x = x.permute(0, 2, 1)
            x = merge3x3(x)
            x = merge1x1(x)
            x = x.permute(0, 2, 1)
            # self-attention with skip connect
            if mask is not None:
                pool = int(torch.round(torch.tensor([n / x.shape[-2]], requires_grad=False)).item())
            
            y, attn_mask = attn(x, mask=mask, pool=pool)
            x = norm1(y + x)
            if attn_mask is not None:
                x = x.masked_fill(attn_mask, 0)
                if decoder_mask is None:
                    decoder_mask = attn_mask
           
            # Mix-FFN with skip connect
            x = norm2(mixffn(x) + x)
            
            if attn_mask is not None:
                x = x.masked_fill(attn_mask, 0)
            # mlp decoder operates on c or channel dim
            features.append(x)

        return features, decoder_mask


class AcousticDecoder(nn.Module):
    """ Pitch, Duration, Energy Predictor """

    def __init__(self, dim, dim_out,
                 pitch_stats=None, 
                 energy_stats=None,
                 n_mel_channels=80, 
                 duration=False):
        super().__init__()
        
        self.n_mel_channels = n_mel_channels

        self.conv1 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.ReLU())
        self.norm1 = nn.LayerNorm(dim)
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.ReLU())
        self.norm2 = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 1)
        self.duration = duration
        
        if pitch_stats is not None:
            pitch_min, pitch_max = pitch_stats
            self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, dim_out - 1),\
                                           requires_grad=False,)
            self.pitch_embedding = nn.Embedding(dim_out, dim_out)
        else:
            self.pitch_bins = None
            self.pitch_embedding = None

        if energy_stats is not None:
            energy_min, energy_max = energy_stats
            self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, dim_out - 1), \
                                            requires_grad=False,)
            self.energy_embedding = nn.Embedding(dim_out, dim_out)
        else:
            self.energy_bins = None
            self.energy_embedding = None

        if duration:
            self.dur_lin = nn.Linear(dim, dim_out)


    def get_pitch_embedding(self, pred, target, mask, control=1.):
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            #pred = pred * control
            embedding = self.pitch_embedding(torch.bucketize(pred, self.pitch_bins))
        return embedding

    def get_energy_embedding(self, pred, target, mask, control=1.):
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            #pred = pred * control
            embedding = self.energy_embedding(torch.bucketize(pred, self.energy_bins))
        return embedding

    def get_embedding(self, pred, target, mask, control=1.):
        if self.pitch_embedding is not None:
            return self.get_pitch_embedding(pred, target, mask, control)
        elif self.energy_embedding is not None:
            return self.get_energy_embedding(pred, target, mask, control)
        return None

    def forward(self, fused_features):
        y = fused_features.permute(0, 2, 1)
        y = self.conv1(y)
        y = y.permute(0, 2, 1)
        y = nn.ReLU()(self.norm1(y))        
        y = y.permute(0, 2, 1)
        y = self.conv2(y)
        y = y.permute(0, 2, 1)
        features = self.norm2(y)
        y = self.linear(y)
        if self.duration:
            y = nn.ReLU()(y)
            features = self.dur_lin(features)
            return y, features

        return y


class Fuse(nn.Module):
    """ Fuse Attn Features"""

    def __init__(self, dims, kernel_size=3):
        super().__init__()

        assert(len(dims)>0)

        dim = dims[0]
        self.mlps = nn.ModuleList([])
        for d in dims:
            upsample = d // dim
            self.mlps.append(
                    nn.ModuleList([
                        nn.Linear(d, dim),
                        nn.ConvTranspose1d(dim, dim, kernel_size=kernel_size, stride=upsample) \
                                           if upsample>1 else nn.Identity()
                        ]))

        self.fuse = nn.Linear(dim*len(dims), dim)

    def forward(self, features, mask=None):

        fused_features = []
        
        # each feature from encoder block
        for feature, mlps in zip(features, self.mlps):
            mlp, upsample = mlps
            # linear projection to uniform channel size (eg 256)
            x = mlp(feature)
            # upsample operates on the n or seqlen dim
            x = x.permute(0, 2, 1)
            # upsample sequence len downsampled by encoder blocks
            x = upsample(x)
            
            if mask is not None:
                x = x[:,:,:mask.shape[1]]
            elif len(fused_features) > 0:
                x = x[:,:,:fused_features[0].shape[-1]] 

            fused_features.append(x)
            #print(x.size())

        # cat on the feature dim
        fused_features = torch.cat(fused_features, dim=-2)
        fused_features = fused_features.permute(0, 2, 1)

        fused_features = self.fuse(fused_features)
        if mask is not None:
            fused_features = fused_features.masked_fill(mask, 0)

        return fused_features


class FeatureUpsampler(nn.Module):
    """ Upsample fused features using target or predicted duration"""

    def __init__(self):
        super().__init__()

    def forward(self, fused_features, fused_masks, duration, max_mel_len=None):
        mel_len = list()
        features = list()
        masks = list()

        for feature, mask, repetition in zip(fused_features, fused_masks, duration):
            repetition = repetition.squeeze().int()
            feature = feature.repeat_interleave(repetition, dim=0)
            mask = mask.repeat_interleave(repetition, dim=0)
            mel_len.append(feature.shape[0])
            if max_mel_len is not None:
                feature = F.pad(feature, (0, 0, 0, max_mel_len -
                                feature.shape[0]), "constant", 0.0)
                mask = F.pad(mask, (0, 0, 0,  max_mel_len -
                             mask.shape[0]), "constant", True)
            features.append(feature)
            masks.append(mask)

        if max_mel_len is None:
            max_mel_len = max(mel_len)
            features = [F.pad(feature, (0, 0, 0, max_mel_len - feature.shape[0]),
                              "constant", 0.0) for feature in features]
            masks = [F.pad(mask, (0, 0, 0, max_mel_len - mask.shape[0]),
                           "constant", True) for mask in masks]

        features = torch.stack(features)
        masks = torch.stack(masks)
        len_pred = torch.IntTensor(mel_len).to(features.device)
        #len_pred = torch.LongTensor(mel_len).to(features.device)

        return features, masks, len_pred

class PhonemeEncoder(nn.Module):
    """ Encodes phonemes to acoustic features """

    def __init__(self,
                 symbols:G2PSymbols,
                 stats, 
                 depth, 
                 reduction,
                 head,
                 embed_dim, 
                 kernel_size,
                 expansion,
                 punct_embed_dim,
                 dpe_embed_dim):
        super().__init__()

        self.encoder = Encoder(symbols=symbols,
                               depth=depth,
                               reduction=reduction, 
                               head=head, 
                               embed_dim=embed_dim, 
                               kernel_size=kernel_size, 
                               expansion=expansion,
                               punct_embed_dim=punct_embed_dim)
        
        dim = ((embed_dim+punct_embed_dim) // reduction)
        self.fuse = Fuse(self.encoder.get_feature_dims(), kernel_size=kernel_size)
        self.feature_upsampler = FeatureUpsampler()
        self.pitch_decoder = AcousticDecoder(dim, dpe_embed_dim, pitch_stats=stats[0:2] if stats else None)
        self.energy_decoder = AcousticDecoder(dim, dpe_embed_dim, energy_stats=stats[2:4] if stats else None)
        self.duration_decoder = AcousticDecoder(dim, dpe_embed_dim, duration=True)
        self._dpe_embed_dim = dpe_embed_dim
        

    def forward(self, x, style_embed, train=False, force_duration=False):
        phoneme = x["phoneme"]
        puncts = x["puncts"]
        phoneme_mask = x["phoneme_mask"] if phoneme.shape[0] > 1 else None

        pitch_target = x["pitch"] if train else None
        energy_target = x["energy"] if train  else None
        duration_target = x["duration"] if train or force_duration else None
        mel_len = x["mel_len"] if train  else None
        max_mel_len = max(mel_len) if train else None

        features, mask = self.encoder(phoneme, puncts, mask=phoneme_mask)
        fused_features = self.fuse(features, mask=mask)

        if style_embed is not None:
            # add GST Tokens (style/speaker) embedding to fused_features
            style_embed = style_embed.expand_as(fused_features)
            fused_features = fused_features + style_embed

        pitch_pred = self.pitch_decoder(fused_features)
        pitch_features = self.pitch_decoder.get_embedding(pitch_pred, pitch_target, mask)
        pitch_features = pitch_features.squeeze(2)
        if mask is not None:
            pitch_features = pitch_features.masked_fill(mask[:,:,:self._dpe_embed_dim], 0)
        # elif pitch_features.dim() != 3:
        #    pitch_features = pitch_features.unsqueeze(0)

        energy_pred = self.energy_decoder(fused_features)
        energy_features = self.energy_decoder.get_embedding(energy_pred, energy_target, mask)
        energy_features = energy_features.squeeze(2)

        if mask is not None:
            energy_features = energy_features.masked_fill(mask[:,:,:self._dpe_embed_dim], 0)
        # elif energy_features.dim() != 3:
        #    energy_features = energy_features.unsqueeze(0)

        log_duration_pred, duration_features = self.duration_decoder(fused_features)
        if mask is not None:
            duration_features = duration_features.masked_fill(mask[:,:,:self._dpe_embed_dim], 0)
        log_duration_pred = log_duration_pred.squeeze(-1)
        
        fused_features = torch.cat([fused_features, pitch_features, \
                                    energy_features, duration_features], dim=-1)

        # TODO: Use fused_masks of all False for inference of bs=1
        if mask is None:
            fused_masks = torch.zeros_like(fused_features).bool()
        else:
            fused_masks = torch.cat([mask, mask[:,:,:self._dpe_embed_dim], mask[:,:,:self._dpe_embed_dim], mask[:,:,:self._dpe_embed_dim]], dim=-1)
        
        if duration_target is None:
            #duration_target = torch.round(duration_pred).squeeze()
            duration_target = torch.clamp(
                (torch.round(torch.exp(log_duration_pred) - 1)),
                min=0,
            )
        if phoneme_mask is not None:
            duration_target = duration_target.masked_fill(phoneme_mask, 0).clamp(min=0)
        else:
            duration_target = duration_target.unsqueeze(0)

        features, masks, mel_len_pred = self.feature_upsampler(fused_features,
                                                               fused_masks,
                                                               duration=duration_target,
                                                               max_mel_len=max_mel_len,)
    
        if mask is None:
            masks = None

        y = {"pitch": pitch_pred,
             "energy": energy_pred,
             "log_duration": log_duration_pred,
             "mel_len": mel_len_pred,
             "features": features,
             "masks": masks, } # [16, 1453, 240]

        return y

    