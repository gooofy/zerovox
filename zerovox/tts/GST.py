#
# this code originates from:
#
# GST-Tacotron by Chengqi Deng
#
# https://github.com/KinglittleQ/GST-Tacotron
#
# which is an implementation of
#
# 	@misc{wang2018style,
# 		  title={Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis},
# 		  author={Yuxuan Wang and Daisy Stanton and Yu Zhang and RJ Skerry-Ryan and Eric Battenberg and Joel Shor and Ying Xiao and Fei Ren and Ye Jia and Rif A. Saurous},
# 		  year={2018},
# 		  eprint={1803.09017},
# 		  archivePrefix={arXiv},
# 		  primaryClass={cs.CL}
# 	}
#
# License: MIT
#


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class GST(nn.Module):
    # emb_size=144, n_mels=80, n_style_tokens=2000, n_heads=8, ref_enc_filters=[32, 32, 64, 64, 128, 128]
    def __init__(self, emb_size:int, n_mels:int, n_style_tokens: int, n_heads: int, ref_enc_filters):

        super().__init__()
        self.encoder = ReferenceEncoder(emb_size, n_mels, ref_enc_filters)
        self.stl = STL(emb_size, n_style_tokens, n_heads)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)   # inputs [8, 487, 80] enc_out [8, 72]
        style_embed = self.stl(enc_out)  # style_embed [8, 1, 144]
        return style_embed


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, emb_size:int, n_mels:int, ref_enc_filters):

        super().__init__()

        self._n_mels          = n_mels
        self._emb_size        = emb_size
        self._ref_enc_filters = ref_enc_filters

        K = len(self._ref_enc_filters)
        filters = [1] + self._ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=self._ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(self._n_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=self._ref_enc_filters[-1] * out_channels,
                          hidden_size=emb_size // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self._n_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, emb_size:int, num_style_tokens: int, num_heads: int):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(num_style_tokens, emb_size // num_heads))
        d_q = emb_size // 2
        d_k = emb_size // num_heads
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=emb_size, num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


