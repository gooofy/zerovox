#
# FastSpeech 2 MEL Decoder
#
# borrowed (under MIT license) from Chung-Ming Chien's implementation of FastSpeech2
#
# https://github.com/ming024/FastSpeech2
#

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

    def forward(self, q, k, v, spk_emb, mask=None):

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
            d_in,
            d_hid,
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
        output = self.w_2(F.relu(self.w_1(output)))
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

    def forward(self, enc_input, spk_emb, mask=None, slf_attn_mask=None):
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
