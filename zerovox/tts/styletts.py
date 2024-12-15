# source:
# https://github.com/yl4579/StyleTTS
# by Aaron (Yinghao) Li (Columbia University) under MIT license
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p
        
        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)
            
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        #self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        #if upsample == 'none':
        self.pool = nn.Identity()
        #else:
        #    self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        # x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class StyleTTSDecoder(nn.Module):
#    def __init__(self, dim_in=512, style_dim=64, residual_dim=64, dim_out=80):
    def __init__(self, dim_in, style_dim, residual_dim, dim_out):
        super().__init__()
        
        self.decode = nn.ModuleList()
        
        self.bottleneck_dim = dim_in * 2
        
        self.encode = nn.Sequential(ResBlk1d(dim_in, self.bottleneck_dim, normalize=True),
                                    ResBlk1d(self.bottleneck_dim, self.bottleneck_dim, normalize=True))
        
        self.decode.append(AdainResBlk1d(self.bottleneck_dim + residual_dim, self.bottleneck_dim, style_dim))
        self.decode.append(AdainResBlk1d(self.bottleneck_dim + residual_dim, self.bottleneck_dim, style_dim))
        self.decode.append(AdainResBlk1d(self.bottleneck_dim + residual_dim, dim_in, style_dim, upsample=True))
        self.decode.append(AdainResBlk1d(dim_in, dim_in, style_dim))
        self.decode.append(AdainResBlk1d(dim_in, dim_in, style_dim))

        # self.F0_conv = nn.Sequential(
        #     ResBlk1d(1, residual_dim, normalize=True, downsample=True),
        #     weight_norm(nn.Conv1d(residual_dim, 1, kernel_size=1)),
        #     nn.InstanceNorm1d(1, affine=True)
        # )
        
        # self.N_conv = nn.Sequential(
        #     ResBlk1d(1, residual_dim, normalize=True, downsample=True),
        #     weight_norm(nn.Conv1d(residual_dim, 1, kernel_size=1)),
        #     nn.InstanceNorm1d(1, affine=True)
        # )
        
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, residual_dim, kernel_size=1)),
            nn.InstanceNorm1d(residual_dim, affine=True)
        )
        
        self.to_out = nn.Sequential(weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0)))
        
    #def forward(self, asr, F0, N, s):        
    #def forward(self, x, s):
    def forward(self, enc_seq, mask, spk_emb):

        # F0 = self.F0_conv(F0.unsqueeze(1))
        # N = self.N_conv(N.unsqueeze(1))
        
        # x = torch.cat([asr, F0, N], axis=1)
        enc_seq = enc_seq.transpose(1,2)
        spk_emb = spk_emb.squeeze(1)

        x = self.encode(enc_seq)
        
        asr_res = self.asr_res(enc_seq)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res], axis=1)
            x = block(x, spk_emb)
            if block.upsample_type != "none":
                res = False
                
        x = self.to_out(x)

        x = x.transpose(1,2)

        return x, None
