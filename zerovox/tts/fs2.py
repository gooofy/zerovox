#
# FastSpeech 2 Encoder + MEL Decoder
#
# borrowed (under MIT license) from Chung-Ming Chien's implementation of FastSpeech2
#
# https://github.com/ming024/FastSpeech2
#

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn

# source: https://github.com/keonlee9420/Cross-Speaker-Emotion-Transfer
# by Keon Lee

class SCLN(nn.Module):
    """ Speaker Condition Layer Normalization """

    def __init__(self, s_size, hidden_size, eps=1e-8, bias=False):
        super(SCLN, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            s_size,
            2 * hidden_size,  # For both b (bias) and g (gain)
            bias,
        )
        self.eps = eps

    def forward(self, x, s):

        # Normalize Input Features
        mu, sigma = torch.mean(
            x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True)
        y = (x - mu) / (sigma + self.eps)  # [B, T, H_m]

        # Get Bias and Gain
        # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]
        b, g = torch.split(self.affine_layer(s), self.hidden_size, dim=-1)

        # Perform Scailing and Shifting
        o = g * y + b  # [B, T, H_m]

        return o


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, spk_emb_size, scln, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        if scln:
            self.layer_norm = SCLN(s_size=spk_emb_size, hidden_size=d_model)
        else:
            self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scln = scln

    def forward(self, q, k, v, spk_emb, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if self.scln:
            output = self.layer_norm(output + residual, spk_emb)
        else:
            output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, spk_emb_size, scln, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise

        self.w_1 = nn.Conv1d(
            in_channels=d_in,
            out_channels=d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        if scln:
            self.layer_norm = SCLN(s_size=spk_emb_size, hidden_size=d_in)
        else:
            self.layer_norm = nn.LayerNorm(d_in)
        self._scln = scln
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spk_emb):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_1(output)
        output = F.relu(output)
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self._scln:
            output = self.layer_norm(output + residual, s=spk_emb)
        else:
            output = self.layer_norm(output + residual)

        return output

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, spk_emb_size, scln, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, spk_emb_size=spk_emb_size, scln=scln, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, spk_emb_size=spk_emb_size, scln=scln, dropout=dropout
        )

    def forward(self, enc_input, spk_emb, mask, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, spk_emb, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output, spk_emb)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn

class FS2Decoder(nn.Module):

    def __init__(self,
                 dec_max_seq_len      : int, # 2000
                 dec_hidden           : int, # 256
                 dec_n_layers         : int, # 6
                 dec_n_head           : int, # 2
                 dec_conv_filter_size : int, # 1024
                 dec_conv_kernel_size : list[int], # [9, 1]
                 dec_dropout          : float, # 0.2
                 dec_scln             : bool, # true
                 n_mel_channels       : int, # 80
                 spk_emb_size         : int
                ):
        super(FS2Decoder, self).__init__()

        n_position = dec_max_seq_len + 1   # config["max_seq_len"] + 1
        d_word_vec = dec_hidden            # config["transformer"]["decoder_hidden"]
        n_layers   = dec_n_layers          # config["transformer"]["decoder_layer"]
        n_head     = dec_n_head            # config["transformer"]["decoder_head"]
        d_k = d_v = (
            # config["transformer"]["decoder_hidden"] // config["transformer"]["decoder_head"]
            dec_hidden // dec_n_head
        )
        d_model = dec_hidden               # config["transformer"]["decoder_hidden"]
        d_inner = dec_conv_filter_size     # config["transformer"]["conv_filter_size"]
        kernel_size = dec_conv_kernel_size # config["transformer"]["conv_kernel_size"]
        dropout = dec_dropout              # config["transformer"]["decoder_dropout"]

        self.max_seq_len = dec_max_seq_len # config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, spk_emb_size=spk_emb_size, scln=dec_scln, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.mel_linear = nn.Linear(dec_hidden, n_mel_channels)

    # enc_seq [bs, 1546, 576] mask
    def forward(self, enc_seq, mask, spk_emb, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask, spk_emb=spk_emb
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        dec_output = self.mel_linear(dec_output)

        return dec_output, mask

class Encoder(nn.Module):
    """ Encoder """

    def __init__(self,
                 max_seq_len,
                 symbols,
                 embed_dim,
                 encoder_layer,
                 encoder_head,
                 conv_filter_size,
                 conv_kernel_size,
                 encoder_dropout,
                 punct_embed_dim):

        super(Encoder, self).__init__()

        n_position = max_seq_len + 1 # config["max_seq_len"] + 1
        #n_src_vocab = symbols.num_phonemes + 1 # len(symbols) + 1
        #d_word_vec = encoder_hidden # config["transformer"]["encoder_hidden"]
        #n_layers = encoder_layer # config["transformer"]["encoder_layer"]
        #n_head = encoder_head # config["transformer"]["encoder_head"]

        encoder_hidden = embed_dim + punct_embed_dim

        d_k = d_v = (encoder_hidden//encoder_head)
        #d_model = encoder_hidden # config["transformer"]["encoder_hidden"]
        d_inner = conv_filter_size # config["transformer"]["conv_filter_size"]
        kernel_size = conv_kernel_size #config["transformer"]["conv_kernel_size"]
        dropout = encoder_dropout # config["transformer"]["encoder_dropout"]

        self.max_seq_len = max_seq_len # config["max_seq_len"]
        self.d_model = encoder_hidden

        self.src_word_emb = nn.Embedding(symbols.num_phonemes + 1, embed_dim, padding_idx=0)
        # self.src_word_emb = nn.Embedding(
        #     n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        # )
        self.punct_embed = nn.Embedding(symbols.num_puncts + 1, punct_embed_dim, padding_idx=0)

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, encoder_hidden).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model, encoder_head, d_k, d_v, d_inner, kernel_size, spk_emb_size=0, scln=False, dropout=dropout
                )
                for _ in range(encoder_layer)
            ]
        )

    def forward(self, src_seq, puncts, mask, return_attns=False):

        x = self.src_word_emb(src_seq) # [16, 126, 128]
        x_punct = self.punct_embed(puncts) # [16, 126, 16]
        x = torch.cat((x, x_punct), 2) # [16, 126, 144]

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = x + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = x + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, spk_emb=None, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output # [16, 126, 144]

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):       
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        else:
            assert False
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()
        #self._device = device

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self,
                 emb_size,
                 vp_filter_size,
                 vp_kernel_size,
                 vp_dropout):
        super(VariancePredictor, self).__init__()

        self.input_size = emb_size # model_config["transformer"]["encoder_hidden"]
        self.filter_size = vp_filter_size #model_config["variance_predictor"]["filter_size"]
        self.kernel = vp_kernel_size #model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = vp_filter_size # model_config["variance_predictor"]["filter_size"]
        self.dropout = vp_dropout # model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self,
                 emb_size,
                 vp_filter_size,
                 vp_kernel_size,
                 vp_dropout,
                 ve_n_bins):
        
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(emb_size=emb_size,
                                                    vp_filter_size=vp_filter_size,
                                                    vp_kernel_size=vp_kernel_size,
                                                    vp_dropout=vp_dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor  = VariancePredictor(emb_size=emb_size,
                                                  vp_filter_size=vp_filter_size,
                                                  vp_kernel_size=vp_kernel_size,
                                                  vp_dropout=vp_dropout)
        self.energy_predictor = VariancePredictor(emb_size=emb_size,
                                                  vp_filter_size=vp_filter_size,
                                                  vp_kernel_size=vp_kernel_size,
                                                  vp_dropout=vp_dropout)

        # pitch_min  = stats[0]
        # pitch_max  = stats[1]
        # energy_min = stats[2]
        # energy_max = stats[3]

        # self.pitch_bins = nn.Parameter(
        #     torch.exp(
        #         torch.linspace(np.log(pitch_min), np.log(pitch_max), ve_n_bins - 1)
        #     ),
        #     requires_grad=False,
        # )

        # self.energy_bins = nn.Parameter(
        #     torch.exp(
        #         torch.linspace(np.log(energy_min), np.log(energy_max), ve_n_bins - 1)
        #     ),
        #     requires_grad=False,
        # )

        self.pitch_embedding = nn.Embedding(
            ve_n_bins, emb_size
        )
        self.energy_embedding = nn.Embedding(
            ve_n_bins, emb_size
        )
        self._ve_n_bins = ve_n_bins

    def get_pitch_embedding(self, x, target, mask):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            # embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
            # out = torch.argmax(out, dim=2)
            # embedding = self.pitch_embedding(torch.argmax(target, dim=2))
            embedding = self.pitch_embedding(torch.round(target*(self._ve_n_bins-1)).long())
        else:
            # embedding = self.pitch_embedding(
            #     torch.bucketize(prediction, self.pitch_bins)
            # )
            #embedding = self.pitch_embedding(torch.argmax(prediction, dim=2))
            embedding = self.pitch_embedding(torch.clamp(torch.round(prediction*(self._ve_n_bins-1)).long(), min=0, max=self._ve_n_bins-1))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            #embedding = self.energy_embedding(torch.argmax(target, dim=2))
            embedding = self.energy_embedding(torch.round(target*(self._ve_n_bins-1)).long())
        else:
            #embedding = self.energy_embedding(torch.argmax(prediction, dim=2))
            embedding = self.energy_embedding(torch.clamp(torch.round(prediction*(self._ve_n_bins-1)).long(), min=0, max=self._ve_n_bins-1))
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, src_mask
        )
        x = x + pitch_embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, src_mask
        )
        x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1)),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )



class FS2Encoder(nn.Module):

    def __init__(self,
                 symbols,
                 max_seq_len,
                 embed_dim,
                 encoder_layer,
                 encoder_head,
                 conv_filter_size,
                 conv_kernel_size,
                 encoder_dropout,
                 punct_embed_dim,
                 vp_filter_size,
                 vp_kernel_size,
                 vp_dropout,
                 ve_n_bins):

        super(FS2Encoder, self).__init__()

        self._encoder = Encoder(max_seq_len=max_seq_len,
                                symbols=symbols,
                                embed_dim=embed_dim,
                                encoder_layer=encoder_layer,
                                encoder_head=encoder_head,
                                conv_filter_size=conv_filter_size,
                                conv_kernel_size=conv_kernel_size,
                                encoder_dropout=encoder_dropout,
                                punct_embed_dim=punct_embed_dim)

        self._variance_adaptor = VarianceAdaptor(emb_size=embed_dim+punct_embed_dim,
                                                 vp_filter_size=vp_filter_size,
                                                 vp_kernel_size=vp_kernel_size,
                                                 vp_dropout=vp_dropout,
                                                 ve_n_bins=ve_n_bins)

    def forward(self, x, style_embed, train=False, force_duration=False):
        phoneme = x["phoneme"]
        puncts = x["puncts"]
        phoneme_mask = x["phoneme_mask"] if 'phoneme_mask' in x else torch.zeros_like(phoneme, dtype=torch.bool) # if phoneme.shape[0] > 1 else None

        features = self._encoder(src_seq=phoneme, puncts=puncts, mask=phoneme_mask, return_attns=False)

        # add speaker embedding to features
        style_embed = style_embed.expand_as(features)
        features = features + style_embed

        pitch_target = x["pitch"] if train else None
        energy_target = x["energy"] if train  else None
        duration_target = x["duration"] if train or force_duration else None
        mel_len = x["mel_len"] if train  else None
        max_mel_len = max(mel_len) if train else None
        mel_masks = x['mel_mask'] if 'mel_mask' in x else None
        (
            features,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_len_pred,
            mel_masks,
        ) = self._variance_adaptor(
            features,
            phoneme_mask, #src_masks,
            mel_masks,
            max_mel_len,
            pitch_target=pitch_target,       # p_targets,
            energy_target=energy_target,     # e_targets,
            duration_target=duration_target, # d_targets,
        )

        y = {"pitch": p_predictions,
             "energy": e_predictions,
             "log_duration": log_d_predictions,
             "mel_len": mel_len_pred,
             "features": features,
             "masks": mel_masks.unsqueeze(2).expand(-1,-1,features.shape[2]) if mel_masks is not None else None  # [16, 1477, 272]
            }
        
        return y