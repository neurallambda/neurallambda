'''

- t10_swap found antiparallel vecs in a sequence and swapped em
- t10_swap_02, do the same on an ordered set
- t10_swap_03 (THIS FILE). Find sorted prefix. Find the minimal thing beyond that, and insert/append it into the prefix


RESULTS:
- length-generalization requires random subsequences of symbols and position encoding

NOTES:

* for the sort-insertion task, adding pos embeddings helps a ton in both transformer and neurallambda
* responds well to dropout on embeddings. val acc increases, stability increases.
* weight decay hurts
* having a large vocabulary really helps
* Transformer3D has nebulous benefit
* TransformerDefeasible has nebulous benefit

TODO: [ ] Test insertion fn
TODO: [ ] simpler problem (like find sorted prefix)
TODO: [ ] sprinkle dropout throughout
TODO: [ ] optimize for loop in Hrrformer


CURRENTLY USING ALIBI together with SINUSOID (but not on v) Also alibi isn't different on different heads

TODO: refactor/cleanup?
TODO: fix alibi
TODO: fix RoPE
TODO: concat position encodings, instead of adding?

'''


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.fft import fft, ifft

import matplotlib.pyplot as plt
import math
import warnings

import neurallambda.model.transformer01 as Transformer2D
import neurallambda.model.transformer_binding_attention as TransformerBind
import neurallambda.model.transformer3d as Transformer
import neurallambda.lab.common as Common
import neurallambda.lab.datasets as Data
import random

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEBUG = False

LOSS_FN = 'cross_entropy'
# LOSS_FN = 'cross_entropy_select_from_inputs'
# LOSS_FN = 'cosine_distance'
# LOSS_FN = 'nllloss'

BATCH_SIZE = 128
DEVICE = 'cuda'
GRAD_CLIP = None

os.environ['DEBUG_MULTIHEAD_ATTENTION'] = 'False'



##################################################
# Defeasibility

class MultiHeadAttentionDefeasible(nn.Module):
    ''' softmax(Q K * Qdef Kdef)V '''

    def __init__(self, emb_dim, num_heads, use_wq=True, use_wk=True, use_wv=True, use_wout=True):
        super(MultiHeadAttentionDefeasible, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wv = use_wv
        self.use_wout = use_wout

        if use_wq:
            self.query = nn.Linear(emb_dim, emb_dim, bias=False)
            self.query_def = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wk:
            self.key = nn.Linear(emb_dim, emb_dim, bias=False)
            self.key_def = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wv:
            self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wout:
            self.out = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, qi, ki, vi, mask=None, attn_nonlin=None):
        '''
        Args:
          attn_nonlin: {None, 'softmax', 'linear', 'sigmoid', 'tanh', 'relu'}
          use_wq, use_wk, use_wv: whether or not to project QKV first
        '''
        assert attn_nonlin in {None, 'none', 'softmax', 'sigmoid', 'tanh', 'relu'}
        B, S, D = qi.size()
        device = qi.device
        # Linear projections
        q = self.query(qi) if self.use_wq else qi  # [B, S, D]
        q = q.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        q_def = self.query_def(qi) if self.use_wq else qi  # [B, S, D]
        q_def = q_def.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        q_def = q_def.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        k = self.key(ki) if self.use_wk else ki  # [B, S, D]
        k = k.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        k = k.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        k_def = self.key_def(ki) if self.use_wk else ki  # [B, S, D]
        k_def = k_def.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        k_def = k_def.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        v = self.value(vi) if self.use_wv else vi  # [B, S, D]
        v = v.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        v = v.transpose(1, 2)  # [B, NUM_HEADS, S, HEAD_DIM]


        # Scaled dot-product attention
        scores = einsum('bhse, bhte -> bhst', q, k)  # [B, N_HEADS, S, S]
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, device=device))  # [B, NUM_HEADS, S, S]

        def_scores = einsum('bhse, bhte -> bhst', q_def, k_def)  # [B, N_HEADS, S, S]
        def_scores = def_scores / torch.sqrt(torch.tensor(self.head_dim, device=device))  # [B, NUM_HEADS, S, S]

        # if mask is not None and mask.ndim == 2:  # [S, S]
        #     mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)  # [B, N_HEADS, S, S]
        #     scores += mask  # [B, S, S]
        # elif mask is not None and mask.ndim == 3:  # [B, S, S]
        #     mask = mask.unsqueeze(1).expand(B, self.num_heads, -1, -1)  # [B, N_HEADS, S, S]
        #     scores += mask  # [B, NUM_HEADS, S, S]

        match attn_nonlin:
            case 'none' | None:
                attention_weights = scores
            case 'softmax':
                # correct softmax dim is -1:
                #   https://github.com/pytorch/pytorch/blob/03725a05127296c581a434cebe727e95caf5c14f/torch/nn/functional.py#L5026

                # attention_weights = F.softmax(scores, dim=-1)

                # def_weights = F.softmax(def_scores, dim=3).sum(dim=2).unsqueeze(3).expand(B, self.num_heads, S, S)
                # def_weights = def_scores.sigmoid()  # 0.975
                # def_weights = def_scores.sigmoid().sum(dim=2).unsqueeze(3).expand(B, self.num_heads, S, S)  # 0.974
                # def_weights = def_scores.sigmoid().sum(dim=3).unsqueeze(3).expand(B, self.num_heads, S, S)  # 0.978
                # def_weights = def_scores.sigmoid().sum(dim=2).unsqueeze(2).expand(B, self.num_heads, S, S)  # 0.980

                def_weights = def_scores.sigmoid().sum(dim=3).unsqueeze(2).expand(B, self.num_heads, S, S)  # 0.980
                attention_weights = F.softmax(scores * def_weights, dim=-1)

            case 'sigmoid':
                attention_weights = scores.sigmoid()
            case 'tanh':
                attention_weights = scores.tanh()
            case 'relu':
                attention_weights = scores.relu()

        # attention_weights = F.dropout(attention_weights, p=dropout_prob, training=self.training)  # [B, NUM_HEADS, S, S]
        # attended_values = torch.matmul(attention_weights, v)  # [B, N_HEADS, S, HEAD_DIM]
        attended_values = torch.einsum('bhst, bhtd -> bshd', attention_weights, v)

        # Concatenation and linear transformation
        # concatenated = attended_values.transpose(1, 2)  # [B, S, N_HEADS, HEAD_DIM]
        concatenated = attended_values.contiguous().view(B, -1, self.emb_dim)  # [B, S, D]
        output = self.out(concatenated) if self.use_wout else concatenated  # [B, S, D]

        return output


class DecoderLayerDefeasible(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward=512, dropout=0.1, use_wq=True, use_wk=True, use_wv=True, use_wout=True):
        super(DecoderLayerDefeasible, self).__init__()
        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wv = use_wv
        self.use_wout = use_wout

        self.self_attn = MultiHeadAttentionDefeasible(emb_dim, num_heads, self.use_wq, self.use_wk, self.use_wv, self.use_wout)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(dim_feedforward, emb_dim, bias=False)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, q, k, v, mask=None, attn_nonlin=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(q, k, v, mask, attn_nonlin)
        xs = self.norm1(q + attn_output)

        # Feed-forward network with residual connection
        ffnn_output = self.ffnn(xs)
        xs = self.norm2(xs + ffnn_output)

        return xs



##################################################
# 3D

class MultiHeadAttention3D(nn.Module):
    ''' softmax(outer(Q, K, L)) repeat(V)'''

    def __init__(self, emb_dim, num_heads, use_wq=True, use_wk=True, use_wl=True, use_wv=True, use_wout=True):
        super(MultiHeadAttention3D, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wl = use_wl
        self.use_wv = use_wv
        self.use_wout = use_wout

        if use_wq:
            self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wk:
            self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wl:
            self.lock = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wv:
            self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wout:
            self.out = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, q, k, l, v, mask=None, attn_nonlin=None):
        '''
        Args:
          attn_nonlin: {None, 'softmax', 'linear', 'sigmoid', 'tanh', 'relu'}
          use_wq, use_wk, use_wv: whether or not to project QKV first
        '''
        assert attn_nonlin in {None, 'none', 'softmax', 'sigmoid', 'tanh', 'relu'}
        B, QS, D = q.shape
        KS = k.shape[1]
        LS = l.shape[1]

        device = q.device

        # Linear projections
        q = self.query(q) if self.use_wq else q  # [B, S, D]
        q = q.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        q = q.transpose(1, 2)  # [batch, num_heads, QS, head_dim]

        k = self.key(k) if self.use_wk else k  # [B, KS, D]
        k = k.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        k = k.transpose(1, 2)  # [batch, num_heads, S, head_dim]

        l = self.key(l) if self.use_wl else l  # [B, LS, D]
        l = l.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        l = l.transpose(1, 2)  # [batch, num_heads, S, head_dim]

        v = self.value(v) if self.use_wv else v  # [B, VS, D]
        v = v.view(B, -1, self.num_heads, self.head_dim)  # [B, VS, N_HEADS, HEAD_DIM]
        v = v.transpose(1, 2)  # [B, NUM_HEADS, S, HEAD_DIM]

        # Scaled dot-product attention
        scores = einsum('bhqe, bhke, bhle -> bhqkl', q, k, l)  # [B, N_HEADS, Q, K, L]
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, device=device))  # [B, NUM_HEADS, Q, K, L]

        if mask is not None:
            raise Warning('mask not implemented for 3d transformer yet')

        match attn_nonlin:
            case 'none' | None:
                attention_weights = scores
            case 'softmax':
                # correct softmax dim for 2d is -1:
                #   https://github.com/pytorch/pytorch/blob/03725a05127296c581a434cebe727e95caf5c14f/torch/nn/functional.py#L5026
                warnings.warn('unprincipled softmax in transformer3d. maybe do the technique to get a 2d softmax?')

                # attention_weights = F.softmax(scores.view(B, self.num_heads, QS * KS, LS), dim=-2).view(B, self.num_heads, QS, KS, LS)  # [B, NUM_HEADS, Q, K, L]
                # attention_weights = torch.einsum('bhqkl -> bhqkl', attention_weights)  # 0.967
                # attention_weights = torch.einsum('bhqkl -> bhqlk', attention_weights)  # 0.962
                # attention_weights = torch.einsum('bhqkl -> bhklq', attention_weights)  # 0.961
                # attention_weights = torch.einsum('bhqkl -> bhkql', attention_weights)  # 0.965
                # attention_weights = torch.einsum('bhqkl -> bhlqk', attention_weights)  # 0.952
                # attention_weights = torch.einsum('bhqkl -> bhlkq', attention_weights)  # 0.971

                # attention_weights = F.softmax(scores.view(B, self.num_heads, QS * KS, LS), dim=-1).view(B, self.num_heads, QS, KS, LS)  # [B, NUM_HEADS, Q, K, L]
                # attention_weights = torch.einsum('bhqkl -> bhqkl', attention_weights)  # 0.953
                # attention_weights = torch.einsum('bhqkl -> bhqlk', attention_weights)  # 0.917
                # attention_weights = torch.einsum('bhqkl -> bhklq', attention_weights)  # 0.888
                # attention_weights = torch.einsum('bhqkl -> bhkql', attention_weights)  # 0.918
                # attention_weights = torch.einsum('bhqkl -> bhlqk', attention_weights)  # 0.902
                # attention_weights = torch.einsum('bhqkl -> bhlkq', attention_weights)  # 0.890

                attention_weights = F.softmax(scores.view(B, self.num_heads, QS, KS * LS), dim=-1).view(B, self.num_heads, QS, KS, LS)  # [B, NUM_HEADS, Q, K, L]
                # attention_weights = torch.einsum('bhqkl -> bhqkl', attention_weights)  # 0.966
                # attention_weights = torch.einsum('bhqkl -> bhqlk', attention_weights)  # 0.967
                # attention_weights = torch.einsum('bhqkl -> bhklq', attention_weights)  # 0.961
                # attention_weights = torch.einsum('bhqkl -> bhkql', attention_weights)  # 0.949
                # attention_weights = torch.einsum('bhqkl -> bhlqk', attention_weights)  # 0.951
                # attention_weights = torch.einsum('bhqkl -> bhlkq', attention_weights)  # 0.957

                # attention_weights = F.softmax(scores.view(B, self.num_heads, QS, KS * LS), dim=-2).view(B, self.num_heads, QS, KS, LS)  # [B, NUM_HEADS, Q, K, L]
                # attention_weights = torch.einsum('bhqkl -> bhqkl', attention_weights)  # 0.914
                # attention_weights = torch.einsum('bhqkl -> bhqlk', attention_weights)  # 0.891
                # attention_weights = torch.einsum('bhqkl -> bhklq', attention_weights)  # 0.963
                # attention_weights = torch.einsum('bhqkl -> bhkql', attention_weights)  # 0.926
                # attention_weights = torch.einsum('bhqkl -> bhlqk', attention_weights)  # 0.918
                # attention_weights = torch.einsum('bhqkl -> bhlkq', attention_weights)  # 0.963

            case 'sigmoid':
                attention_weights = scores.sigmoid()
            case 'tanh':
                attention_weights = scores.tanh()
            case 'relu':
                attention_weights = scores.relu()

        vv = v.unsqueeze(2).expand(B, self.num_heads, LS, KS, self.head_dim)  # [B, NUM_HEADS, L * K, HEAD_DIM]
        # vv = v.repeat(1, 1, KS, 1)  # [B, NUM_HEADS, L * K, HEAD_DIM]

        attended_values = torch.einsum('bhqkl, bhlkd -> bqhd', attention_weights, vv)

        # Concatenation and linear transformation
        # concatenated = attended_values.transpose(1, 2)  # [B, S, N_HEADS, HEAD_DIM]
        concatenated = attended_values.contiguous().view(B, -1, self.emb_dim)  # [B, S, D]
        output = self.out(concatenated) if self.use_wout else concatenated  # [B, S, D]

        return output


class DecoderLayer3D(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward=512, dropout=0.1, use_wq=True, use_wk=True, use_wl=True, use_wv=True, use_wout=True):
        super(DecoderLayer3D, self).__init__()
        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wl = use_wl
        self.use_wv = use_wv
        self.use_wout = use_wout

        self.self_attn = MultiHeadAttention3D(emb_dim, num_heads, self.use_wq, self.use_wk, self.use_wl, self.use_wv, self.use_wout)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=True),
            nn.ReLU(),
            nn.Linear(dim_feedforward, emb_dim, bias=True)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, q, k, l, v, mask=None, attn_nonlin=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(q, k, l, v, mask, attn_nonlin)
        xs = self.norm1(q + attn_output)

        # Feed-forward network with residual connection
        ffnn_output = self.ffnn(xs)
        xs = self.norm2(xs + ffnn_output)

        return xs


##################################################
# HrrFormer

def bind(x, y):
    return ifft(fft(x) * fft(y)).real


def approx_transpose(x):
    ''' actual transpose is numerically unstable '''
    x = torch.flip(x, dims=[-1])
    return torch.roll(x, 1, dims=-1)


def unbind(x, y):
    ''' y gets inverted '''
    return bind(x, approx_transpose(y))


def hrr_project(x):
    f = fft(x).abs()
    p = ifft(fft(x) / f).real()
    return torch.nan_to_num(p)


def up(x):
    return torch.roll(x, shifts=(1,), dims=(-1,))


def down(x):
    return torch.roll(x, shifts=(-1,), dims=(-1,))

# # @@@@@@@@@@

# n = 10000
# dim = 1024
# sym = torch.randn(n, dim)

# a = sym[0]
# b = sym[1]
# c = sym[2]
# d = sym[3]
# e = sym[4]
# f = sym[5]
# g = sym[6]
# h = sym[7]
# i = sym[8]

# val = bind(a, up(b)) + bind(c, up(d)) + bind(e, up(f)) + bind(g, up(h))
# out = down(unbind(val, c))
# for i, c in enumerate('a b c d e f g h i'.split(' ')):
#     most_sim = torch.cosine_similarity(out, sym[i], dim=0).abs()
#     print(f'{i} {c} {most_sim:>.2f}')

# sims = torch.cosine_similarity(out.unsqueeze(0), sym, dim=1)
# so = torch.softmax(sims * math.sqrt(1000), dim=0).max()
# print(so)

# BRK
# # @@@@@@@@@@


class HRRAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, use_wq=True, use_wk=True, use_wv=True, use_wout=True):
        super(HRRAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wv = use_wv
        self.use_wout = use_wout

        if use_wq:
            self.query = nn.Linear(emb_dim, emb_dim, bias=True)
        if use_wk:
            self.key = nn.Linear(emb_dim, emb_dim, bias=True)
        if use_wv:
            self.value = nn.Linear(emb_dim, emb_dim, bias=True)
        if use_wout:
            self.out = nn.Linear(emb_dim, emb_dim, bias=True)

    def forward(self, q, k, v, mask=None, attn_nonlin=None):
        '''
        Args:
          attn_nonlin: {None, 'softmax', 'linear', 'sigmoid', 'tanh', 'relu'}
          use_wq, use_wk, use_wv: whether or not to project QKV first
        '''
        assert attn_nonlin in {None, 'none', 'softmax', 'sigmoid', 'tanh', 'relu'}
        B, S, D = q.shape
        device = q.device

        # Linear projections
        q = self.query(q) if self.use_wq else q  # [B, S, D]
        q = q.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]

        k = self.key(k) if self.use_wk else k  # [B, S, D]
        k = k.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]

        v = self.value(v) if self.use_wv else v  # [B, S, D]
        v = v.view(B, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]

        beta = []
        for i in range(S): # OPTIM: remove loop
            bv = v[:, i]
            # bv = up(bv)
            beta.append(bind(k[:, i], bv))
        beta = torch.stack(beta, dim=1)
        beta = beta.sum(dim=1) # sum across time
        # beta = beta.view(B, self.num_heads, self.head_dim)

        a = []
        for i in range(S): # OPTIM: remove loop
            v_hat = unbind(beta, q[:, i]) # 2nd term gets transposed
            # v_hat = down(v_hat)
            a.append(torch.cosine_similarity(v[:, i], v_hat, dim=-1))
        a = torch.stack(a, dim=1) # [B, S, N_HEADS]
        w = torch.softmax(a, dim=-2) # softmax across time

        attn = torch.einsum('bth, bthd -> bthd', w, v).view(B, S, D)
        attn = self.out(attn)
        return attn


class HRRLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward=512, dropout=0.1, use_wq=True, use_wk=True, use_wv=True, use_wout=True):
        super(HRRLayer, self).__init__()
        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wv = use_wv
        self.use_wout = use_wout

        self.self_attn = HRRAttention(emb_dim, num_heads, self.use_wq, self.use_wk, self.use_wv, self.use_wout)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(dim_feedforward, emb_dim, bias=False)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, q, k, v, mask=None, attn_nonlin=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(q, k, v, mask, attn_nonlin)
        xs = self.norm1(q + attn_output)

        # Feed-forward network with residual connection
        ffnn_output = self.ffnn(xs)
        xs = self.norm2(xs + ffnn_output)

        return xs



##################################################
# Data

print()

lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
# lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' ')
# lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt uu vv ww xx yy zz 00 11 22 33 44 55 66 77 88 99 aaa bbb'.split(' ')  # len=100
# lang = '0 1 2 3 4 5 6 7 8 9'.split(' ')
train_lang = lang
val_lang = lang

MIN_LENGTH = 3
TRAIN_MAX_SEQUENCE_LENGTH = 10
VAL_MAX_SEQUENCE_LENGTH = 10


# # @@@@@@@@@@
# samples = Data.insert_min_into_sorted_prefix(10, 10, 10, lang='0 1 2 3 4 5 6 7 8 9'.split(' '), mask_type='all', sample_with_replacement=False)
# for s in samples:
#     print('----------')
#     print(' '.join(s['inputs']))
#     print(' '.join(s['outputs']))


##################################################
#

def swap(x, swap1, swap2):
    ''' swap1 and swap2 are softmax vectors (think onehot) of rows of x that will
    be swapped. '''
    # Combine swap1 and swap2 into a single matrix
    P = torch.einsum('bx, by -> bxy', swap1, swap2)
    P = P + P.transpose(1, 2)  # swap both directions
    # identity matrix to keep non-swapped data
    Id = torch.diag_embed(torch.ones_like(swap1) - swap1 - swap2)
    x_swapped = torch.einsum('bij, bjd -> bid', P + Id, x)
    return x_swapped


def permute_one(xs, from_prob, to_prob):
    '''insert the item in a sequence at `from_prob` to `to_prob`.

    CAVEAT: if moving forward, the item will be placed *after* the element at
    to_prob, and if moving an item backward, it will be placed at the mentioned
    index, ie, before the thing that's currently at the to-index. This allows
    to move items from the very start of the list to the very end of the list,
    and vice versa.

    args:
      xs: [batch, sequence, embedding dim]
      from_prob, to_prob: [batch, sequence]

    '''
    S = xs.size(1)

    fcs2 = torch.cumsum(to_prob, dim=1)
    fcs1 = ((fcs2 - 1) * -1)

    tcs2 = torch.cumsum(from_prob, dim=1)
    tcs1 = ((tcs2 - 1) * -1)

    # left block
    lb = torch.diag_embed(fcs1 * tcs1)

    # mid block
    mb = (
        # shift down for case where moving ix backward. If it's actually moving
        # ix forward, this should become all 0s.
        F.pad(torch.diag_embed(fcs2 * tcs1), [0, 0, 1, 0], value=0)[:, :-1] +

        # shift I up and to the right for case where moving ix forward. If it's
        # actually moving ix backward, this should become all 0s.
        F.pad(torch.diag_embed(F.pad(fcs1, [1, 0], value=0)[:, :S] *
                               F.pad(tcs2, [1, 0], value=0)[:, :S]), [0, 0, 0, 1])[:, 1:]
    )

    # value that's moved
    val = torch.einsum('bs, bt -> bst', to_prob, from_prob)

    # right block
    rb = torch.diag_embed(F.pad(tcs2, [1, 0], value=0)[:, :S] *
                          F.pad(fcs2, [1, 0], value=0)[:, :S])
    perm = (lb + mb + val + rb)

    return torch.einsum('bst, btd -> bsd', perm, xs)


# B = 2
# S = 10
# D = 3

# xs = torch.arange(B * S, dtype=torch.float).reshape(B, S).unsqueeze(2).repeat(1, 1, D)

# from_prob = torch.zeros((B, S)).float()
# from_prob[:, 8] = 0.1
# from_prob[:, 9] = 0.9

# to_prob = torch.zeros((B, S)).float()
# to_prob[:, 1] = 0.1
# to_prob[:, 2] = 0.9


# print()
# print('permute one')
# ys = permute_one(xs, from_prob, to_prob)
# print(ys)




# # @@@@@@@@@@
# # Hand check generated data

# val_raw = Data.insert_min_into_sorted_prefix(10, 5, 10, lang=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mask_type='all', sample_with_replacement=False)
# tokenizer, create_dataloaders = Common.build_tokenizer_dataloader([train_raw], data_keys=['inputs', 'outputs'])
# train_dl, = create_dataloaders(BATCH_SIZE)


# for inp, out, attn in train_dl:
#     break

# BRK
# # @@@@@@@@@@



##################################################
# LSTM

class LSTMModel(nn.Module):
    def __init__(self, tokenizer, emb_dim, hidden_dim, num_layers, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        if LOSS_FN in {'cross_entropy', 'cross_entropy_select_from_inputs'}:
            self.fc_out = nn.Linear(hidden_dim, self.vocab_size)
        elif LOSS_FN == 'cosine_distance':
            self.fc_out = nn.Linear(hidden_dim, emb_dim)

    def forward(self, xs_ids):
        B, S = xs_ids.shape
        device = xs_ids.device

        xs = self.embeddings(xs_ids) * math.sqrt(self.emb_dim)
        xs = self.dropout(xs)

        lstm_out, _ = self.lstm(xs)
        lstm_out = self.dropout(lstm_out)

        if LOSS_FN == 'cosine_distance':
            output = self.fc_out(lstm_out)
            assert output.size(0) == xs_ids.size(0)
            assert output.size(1) == xs_ids.size(1)
            return output
        elif LOSS_FN == 'cross_entropy':
            logits = self.fc_out(lstm_out)
            assert logits.size(0) == xs_ids.size(0)
            assert logits.size(1) == xs_ids.size(1)
            assert logits.size(2) == self.vocab_size
            return logits


##########


class FullLSTM(nn.Module):
    ''' Transductive LLM, records and stacks each output, so, same sequence length as input. '''
    def __init__(self, input_size, hidden_size, dropout):
        super(FullLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h, c = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(x.device), torch.zeros(batch_size, self.lstm_cell.hidden_size).to(x.device)
        outputs = []

        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        # output = self.layer_norm(output)
        return output


class LSTMModel(nn.Module):
    def __init__(self, tokenizer, emb_dim, hidden_dim, num_layers, num_heads=8, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        self.max_symbols = 100
        self.symbol_embeddings = torch.randn(self.max_symbols, emb_dim)


        ##########

        # Transformer
        self.initial_decoder = Transformer2D.DecoderLayer(emb_dim, num_heads, hidden_dim, dropout)
        self.final_decoder = Transformer2D.DecoderLayer(emb_dim, num_heads, hidden_dim, dropout)

        # # MultiheadAttention
        # self.initial_decoder = Transformer2D.MultiHeadAttention(emb_dim, num_heads, use_wq=True, use_wk=True, use_wv=True, use_wout=True)
        # self.final_decoder = Transformer2D.MultiHeadAttention(emb_dim, num_heads, use_wq=True, use_wk=True, use_wv=True, use_wout=True)

        self.lstm_layers = nn.ModuleList([
            nn.ModuleList([FullLSTM(emb_dim, emb_dim, dropout), nn.LayerNorm(emb_dim)])
            for i in range(num_layers)
        ])


        self.fc_out = nn.Linear(emb_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, xs_ids, mask=None):
        B, S = xs_ids.shape
        device = xs_ids.device
        self.symbol_embeddings = self.symbol_embeddings.to(device)

        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)

        # Embedding
        xs = self.embeddings(xs_ids) * math.sqrt(self.emb_dim)
        xs = self.dropout(xs)

        # Generate ALiBi bias
        m = alibi_slope(self.num_heads, device)
        alibi_bias = (m * relative_positions(S, device)).unsqueeze(0).expand(B, -1, -1, -1)  # [B, NUM_HEAD, S, S]

        # Initial Decoder layer
        xs = self.initial_decoder(xs, xs, v, mask=mask, attn_nonlin='softmax', alibi_bias=alibi_bias)

        # LSTM layers processing full sequences
        for lstm_layer, norm in self.lstm_layers:
            # xs = lstm_layer(xs) + xs  # add residual
            xs = lstm_layer(xs)
            xs = norm(xs)

        # Final Decoder layer
        xs = self.final_decoder(xs, xs, xs, mask=mask, attn_nonlin='softmax', alibi_bias=alibi_bias)

        # Final output
        logits = self.fc_out(xs)

        assert logits.size(0) == xs_ids.size(0)
        assert logits.size(1) == xs_ids.size(1)
        assert logits.size(2) == self.vocab_size

        return logits



##################################################


class LSTMSwapModel(nn.Module):
    '''

    uses a differentiable `permute_one` function to swap
    `permute_one` takes 2 args referring to swap locations, each of which is calculated by a separate Net

    '''
    def __init__(self, tokenizer, emb_dim, hidden_dim, num_layers, num_heads=8, dropout=0.1):
        super(LSTMSwapModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        self.max_symbols = 100
        self.symbol_embeddings = torch.randn(self.max_symbols, emb_dim)

        ##########

        # Net 1
        self.trans_i1 = Transformer2D.DecoderLayer(emb_dim, num_heads, hidden_dim, dropout)
        self.lstm1 = nn.ModuleList([nn.ModuleList([FullLSTM(emb_dim, emb_dim, dropout), nn.LayerNorm(emb_dim)]) for i in range(num_layers)])
        self.trans_o1 = Transformer2D.DecoderLayer(emb_dim, num_heads, hidden_dim, dropout)
        self.o1 = nn.Linear(emb_dim, 1)

        # Net 2
        self.trans_i2 = Transformer2D.DecoderLayer(emb_dim, num_heads, hidden_dim, dropout)
        self.lstm2 = nn.ModuleList([nn.ModuleList([FullLSTM(emb_dim, emb_dim, dropout), nn.LayerNorm(emb_dim)]) for i in range(num_layers)])
        self.trans_o2 = Transformer2D.DecoderLayer(emb_dim, num_heads, hidden_dim, dropout)
        self.o2 = nn.Linear(emb_dim, 1)

        self.out = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, xs_ids, mask=None):
        B, S = xs_ids.shape
        device = xs_ids.device
        self.symbol_embeddings = self.symbol_embeddings.to(device)
        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)

        # Generate ALiBi bias
        m = alibi_slope(self.num_heads, device)
        alibi_bias = (m * relative_positions(S, device)).unsqueeze(0).expand(B, -1, -1, -1)  # [B, NUM_HEAD, S, S]

        # Embedding
        xs = self.embeddings(xs_ids) * math.sqrt(self.emb_dim)

        # Net 1
        n1 = self.trans_i1(xs, xs, xs, mask=mask, attn_nonlin='softmax', alibi_bias=alibi_bias)
        for layer, norm in self.lstm1:
            # n1 = layer(n1) + n1  # add residual
            n1 = layer(n1)
            n1 = norm(n1)
        n1 = self.trans_o1(n1, n1, n1, mask=mask, attn_nonlin='softmax', alibi_bias=alibi_bias)
        n1 = self.o1(n1)

        # Net 2
        n2 = self.trans_i1(xs, xs, xs, mask=mask, attn_nonlin='softmax', alibi_bias=alibi_bias)
        for layer, norm in self.lstm2:
            # n2 = layer(n2) + n2  # add residual
            n2 = layer(n2)
            n2 = norm(n2)
        n2 = self.trans_o2(n2, n2, n2, mask=mask, attn_nonlin='softmax', alibi_bias=alibi_bias)
        n2 = self.o2(n2)

        # swap1 = torch.softmax(n1, dim=1)  # [B, S]
        # swap2 = torch.softmax(n2, dim=1)

        # swap1 = F.gumbel_softmax(n1, dim=1, hard=False)
        # swap2 = F.gumbel_softmax(n2, dim=1, hard=False)

        swap1 = F.gumbel_softmax(n1, dim=1, hard=True)
        swap2 = F.gumbel_softmax(n2, dim=1, hard=True)

        ys = permute_one(xs, swap1.squeeze(2), swap2.squeeze(2))
        logits = self.out(ys)

        assert logits.size(0) == xs_ids.size(0)
        assert logits.size(1) == xs_ids.size(1)
        assert logits.size(2) == self.vocab_size

        return logits


##################################################


class BindTransformer(nn.Module):
    '''

    Use 'Binding Attention'

    '''
    def __init__(self, tokenizer, emb_dim, hidden_dim, num_layers, num_heads=8, dropout=0.1):
        super(BindTransformer, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        self.max_symbols = 100
        self.symbol_embeddings = torch.randn(self.max_symbols, emb_dim)

        use_wq = True
        use_wk = True
        use_wv = True
        use_wout = True
        self.layers = nn.ModuleList([
            TransformerBind.DecoderLayer(emb_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout,
                                         use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
            for _ in range(num_layers)])

        self.out = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, xs_ids, mask=None):
        B, S = xs_ids.shape
        device = xs_ids.device

        self.symbol_embeddings = self.symbol_embeddings.to(device)
        sym = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)

        # # Generate ALiBi bias
        # m = alibi_slope(self.num_heads, device)
        # alibi_bias = (m * relative_positions(S, device)).unsqueeze(0).expand(B, -1, -1, -1)  # [B, NUM_HEAD, S, S]

        alibi_bias = None  # i don't think BindAttn needs it

        attn = 'softmax'

        # Embedding
        xs = self.embeddings(xs_ids) * math.sqrt(self.emb_dim)

        # Noise experiment
        # if model.training:
        #     xs = xs + torch.randn_like(xs) * 1e-0

        for i in range(self.num_layers):
            layer = self.layers[i]
            q = xs
            k = xs
            v = xs
            xs = layer(q, k, v, mask=None, attn_nonlin=attn, alibi_bias=alibi_bias)

        logits = self.out(xs)

        assert logits.size(0) == xs_ids.size(0)
        assert logits.size(1) == xs_ids.size(1)
        assert logits.size(2) == self.vocab_size

        return logits


##################################################


def relative_positions(seq_len: int, device) -> torch.tensor:
    ''' for ALiBi '''
    x = torch.arange(seq_len, device=device)[None, :]
    y = torch.arange(seq_len, device=device)[:, None]
    return x - y


def alibi_slope(num_heads, device):
    ''' for ALiBi '''
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

class ControlModel(nn.Module):
    def __init__(self, tokenizer, emb_dim, num_heads, num_layers, dim_feedforward, num_recurrence, attn_nonlin, dropout=0.1, model_type='transformer'):
        super(ControlModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_recurrence = num_recurrence
        self.attn_nonlin = attn_nonlin
        self.model_type = model_type
        assert model_type in {'transformer',
                              'abstractor_all', 'abstractor_first', 'abstractor_last',
                              'sorter',
                              'lstm', 'lstm_swap',
                              'bind_transformer'}

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)
        self.pos_encoding = Transformer.positional_encoding(emb_dim, max_len=20)

        if model_type == 'transformer':
            use_wq = True
            use_wk = True
            use_wl = True
            use_wv = True
            use_wout = True
            self.layers = nn.ModuleList([
                Transformer2D.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                                           use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
                for _ in range(num_layers)])

        elif model_type in {'abstractor_all', 'abstractor_first', 'abstractor_last'}:
            self.max_symbols = 100
            self.symbol_embeddings = torch.randn(self.max_symbols, emb_dim)

            use_wq = True
            use_wk = True
            use_wl = True
            use_wv = True
            use_wout = True

            self.layers = nn.ModuleList([

                Transformer2D.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                                           use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)

                # DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                #              use_wq=use_wq, use_wk=use_wk, use_wl=use_wl, use_wv=use_wv, use_wout=use_wout)

                for _ in range(num_layers)])


        elif model_type == 'sorter':
            self.max_symbols = 100
            self.symbol_embeddings = torch.randn(self.max_symbols, emb_dim)

            self.l_embeddings = nn.Embedding(self.vocab_size, emb_dim)

            use_wq = True
            use_wk = True
            use_wl = True
            use_wv = True
            use_wout = True


            # self.layers = nn.ModuleList([
            #     # DecoderLayer3D(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            #     #              use_wq=False, use_wk=False, use_wl=False, use_wv=False, use_wout=False)
            #     # if i == 0 else
            #     DecoderLayer3D(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            #                    use_wq=use_wq, use_wk=use_wk, use_wl=use_wl, use_wv=use_wv, use_wout=use_wout)
            #     for i in range(num_layers)])

            # self.layers = nn.ModuleList([
            #     # DecoderLayerDefeasible(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            #     #              use_wq=False, use_wk=False, use_wv=False, use_wout=False)
            #     # if i == 0 else
            #     DecoderLayerDefeasible(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            #                            use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
            #     for i in range(num_layers)])

            # self.layers = nn.ModuleList([
            #     # HRRLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            #     #              use_wq=False, use_wk=False, use_wv=False, use_wout=False)
            #     # if i == 0 else
            #     HRRLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            #              use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
            #     for i in range(num_layers)]
            #     )

            self.layers = nn.ModuleList([
                # Transformer2D.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                #              use_wq=False, use_wk=False, use_wv=False, use_wout=False)
                # if i == 0 else
                Transformer2D.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                                           use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
                for i in range(num_layers)]
                )

            self.probs = nn.Sequential(
                nn.Linear(emb_dim, 2, bias=False)

                # nn.Linear(emb_dim, 16),
                # nn.Tanh(),
                # nn.Linear(16, 2)
            )

            # self.probs2 = nn.Sequential(
            #     nn.Linear(emb_dim, 1, bias=False)

            #     # nn.Linear(emb_dim, 16),
            #     # nn.Tanh(),
            #     # nn.Linear(16, 2)
            # )

        elif model_type == 'lstm':
            self.lstm_model = LSTMModel(tokenizer, emb_dim, dim_feedforward, num_layers, num_heads, dropout)

        elif model_type == 'lstm_swap':
            self.lstm_swap_model = LSTMSwapModel(tokenizer, emb_dim, dim_feedforward, num_layers, num_heads, dropout)

        elif model_type == 'bind_transformer':
            self.bind_transformer_model = BindTransformer(tokenizer, emb_dim, dim_feedforward, num_layers, num_heads, dropout)

        if LOSS_FN in {'cross_entropy', 'cross_entropy_select_from_inputs'}:
            self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
        elif LOSS_FN == 'cosine_distance':
            self.fc_out = nn.Linear(emb_dim, emb_dim, bias=False)

        self.dropout = nn.Dropout(0.1)

    def forward(self, xs_ids):

        if self.model_type == 'lstm':
            return self.lstm_model(xs_ids)

        if self.model_type == 'lstm_swap':
            return self.lstm_swap_model(xs_ids)

        if self.model_type == 'bind_transformer':
            return self.bind_transformer_model(xs_ids)


        B, S = xs_ids.shape
        device = xs_ids.device
        if self.model_type in {'sorter', 'abstractor_all', 'abstractor_first', 'abstractor_last'}:
            self.symbol_embeddings = self.symbol_embeddings.to(device)

        xs = self.embeddings(xs_ids) * math.sqrt(self.emb_dim)
        orig_xs = xs  # save for use in sorter

        xs = self.dropout(xs)

        ##########
        # Sinusoidal Positions

        # Generate random start positions
        self.pos_encoding = self.pos_encoding.to(DEVICE)
        pos_len = self.pos_encoding.shape[0]
        p_start = torch.randint(0, pos_len, (B,), device=device)

        # Create indices for each sequence, allowing wraparound
        indices = (p_start.unsqueeze(1) + torch.arange(S, device=device).unsqueeze(0)) % pos_len

        # Select the positional encodings
        pos = self.pos_encoding[indices]
        # pos = self.pos_encoding[:S, :].to(DEVICE).expand(B, -1, -1)


        ##########
        # ALIBI

        alibi_bias_2 = -torch.arange(1, S + 1).to(device) * 0.2
        alibi_bias_2 = alibi_bias_2.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, S, -1)
        # alibi_bias_2 = alibi_bias_2.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, S)


        m = alibi_slope(self.num_heads, device)
        # m = 0.1
        alibi_bias = (m * relative_positions(S, device)).unsqueeze(0).expand(B, -1, -1, -1)  # [B, NUM_HEAD, S, S]


        alibi_bias = alibi_bias + alibi_bias_2


        for i in range(self.num_recurrence):

            for j in range(self.num_layers):
                layer = self.layers[j]
                # layer2 = self.layers2[j]

                if self.model_type == 'transformer':
                    q = xs
                    k = xs
                    l = xs
                    v = xs
                    if j == 0:  # add at first layer of each recurrence
                        q = q + pos
                        k = k + pos
                        l = l + pos
                        v = v + pos


                elif self.model_type == 'abstractor_all':
                    q = xs
                    k = xs
                    l = xs
                    v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                    if j == 0:  # add at first layer of each recurrence
                        q = q + pos
                        k = k + pos
                        l = l + pos
                        v = v + pos


                elif self.model_type == 'abstractor_first':
                    q = xs
                    k = xs
                    l = xs
                    if j == 0:
                        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                    else:
                        v = xs

                    if j == 0:  # add at first layer of each recurrence
                        q = q + pos
                        k = k + pos
                        l = l + pos
                        v = v + pos

                elif self.model_type == 'abstractor_last':
                    q = xs
                    k = xs
                    l = xs
                    if j == self.num_recurrence - 1:
                        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                    else:
                        v = xs

                    if j == 0:  # add at first layer of each recurrence
                        q = q + pos
                        k = k + pos
                        l = l + pos
                        v = v + pos

                elif self.model_type == 'sorter':
                    q = xs
                    k = xs

                    if j in {0, 2}:
                        l = self.l_embeddings(xs_ids) * math.sqrt(self.emb_dim)

                        # grab a random subsequence of symbols
                        start_indices = torch.randint(0, self.max_symbols, (B,), device=device)

                        # Create indices for each sequence, allowing wraparound
                        indices = (start_indices.unsqueeze(1) + torch.arange(S, device=device).unsqueeze(0)) % self.max_symbols
                        v = self.symbol_embeddings.unsqueeze(0).expand(B, self.max_symbols, self.emb_dim)[torch.arange(B).unsqueeze(1), indices]

                    else:
                        l = xs
                        v = xs

                    # # Sorter Positions
                    # # if j in {0, 1}:
                    # if True: # True:
                    #     q = q + pos
                    #     k = k + pos
                    #     l = l + pos
                    #     v = v # + pos

                if self.model_type == 'sorter' and j == 0:
                # if self.model_type == 'sorter' and j == self.num_recurrence - 1:
                    # attn = None
                    # attn = 'sigmoid'
                    attn = 'softmax'
                else:
                    attn = self.attn_nonlin

                if isinstance(layer, DecoderLayer3D):
                    xs = layer(q, k, l, v, mask=None, attn_nonlin=attn)
                else:
                    xs = layer(q, k, v, mask=None, attn_nonlin=attn, alibi_bias=alibi_bias)

                    # xs2 = layer2(q, k, v, mask=None, attn_nonlin=attn)

            # if False:
            if self.model_type == 'sorter':
                swap_ixs = self.probs(xs)

                s1 = swap_ixs[:, :, 0]
                s2 = swap_ixs[:, :, 1]

                swap1 = torch.softmax(s1, dim=1)  # [B, S]
                swap2 = torch.softmax(s2, dim=1)

                # swap1 = F.gumbel_softmax(swap_ixs[:, 0:S, 0], dim=1, hard=False)
                # swap2 = F.gumbel_softmax(swap_ixs[:, 0:S, 1], dim=1, hard=False)

                # swap1 = F.gumbel_softmax(swap_ixs[:, 0:S, 0], dim=1, hard=True)
                # swap2 = F.gumbel_softmax(swap_ixs[:, 0:S, 1], dim=1, hard=True)

                if DEBUG:
                    breakpoint()
                xs = permute_one(orig_xs, swap1, swap2)
                # xs = in_xs

        ##########
        # outputs

        if LOSS_FN == 'cosine_distance':
            assert xs.size(0) == xs_ids.size(0)
            assert xs.size(1) == xs_ids.size(1)
            return xs

        elif LOSS_FN == 'cross_entropy':
            xs = self.fc_out(xs)
            # xs = torch.einsum('vd, bsd -> bsv', self.embeddings.weight, xs)

            assert xs.size(0) == xs_ids.size(0)
            assert xs.size(1) == xs_ids.size(1)
            assert xs.size(2) == self.vocab_size
            return xs


##################################################


def show(attn_weights, batch_ix):
    head = 0
    with torch.no_grad():
        plt.imshow(attn_weights[batch_ix, head])
        plt.show()

def accuracy(y_pred, y, threshold=0.7):
    B, S, D = y_pred.size()

    # Normalize predictions and targets
    y_pred_norm = F.normalize(y_pred, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    # Compute cosine similarity between predictions and targets
    cosine_sim = torch.einsum('bsd, bsd -> bs', y_pred_norm, y_norm)

    # Check if cosine similarity is above the threshold
    correct = (cosine_sim > threshold).float()

    # Compute accuracy
    acc = torch.mean(correct)

    return acc




embedding_size = 256

LR = 3e-4
NUM_EPOCHS = 40
VAL_NUM_SAMPLES = 200

num_repetitions = 1  # number of times to repeat and average a result
# train_sizes = [400, 800, 1200, 1600, 2000, 2400, 2800] # , 2000, 4000, 8000]
train_sizes = [4000,]
architectures = [

    # {"name": "Transformer",
    #  "init_params": {"model_type": "transformer", "num_heads": 4, "num_layers": 3, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

    # {"name": "Neurallambda",
    #  "init_params": {"model_type": "sorter", "num_heads": 4, "num_layers": 3, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

    # {"name": "Abstractor_First",
    #  "init_params": {"model_type": "abstractor_first", "num_heads": 4, "num_layers": 3, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

    # {"name": "LSTM",
    #  "init_params": {"model_type": "lstm", "num_heads": 4, "num_layers": 3, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

    # {"name": "LSTMSwap",
    #  "init_params": {"model_type": "lstm_swap", "num_heads": 4, "num_layers": 3, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

    {"name": "BindTransformer",
     "init_params": {"model_type": "bind_transformer", "num_heads": 4, "num_layers": 3, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

]

# Collect results for each training size and architecture
results = {}
for train_size in train_sizes:
    train_raw = Data.insert_min_into_sorted_prefix(train_size, MIN_LENGTH, TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang, mask_type='all', sample_with_replacement=True)
    # train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))

    val_raw = Data.insert_min_into_sorted_prefix(VAL_NUM_SAMPLES, VAL_MAX_SEQUENCE_LENGTH, VAL_MAX_SEQUENCE_LENGTH, lang=val_lang, mask_type='all', sample_with_replacement=True)
    # val_raw = sorted(val_raw, key=lambda x: len(x['inputs']))

    tokenizer, create_dataloaders = Common.build_tokenizer_dataloader([train_raw, val_raw], data_keys=['inputs', 'outputs'])
    train_dl, val_dl = create_dataloaders(BATCH_SIZE)

    for arch in architectures:
        # per-arch results
        train_accs_list = []
        test_accs_list = []
        train_losses_list = []
        test_losses_list = []

        for _ in range(num_repetitions):
            name = arch['name']
            init_params = arch['init_params']
            print(f"Running test for {name} with {train_size} training samples")

            model = ControlModel(
                tokenizer,
                embedding_size,
                **init_params
            )
            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)

            for epoch in range(NUM_EPOCHS):
                train_loss, tacc, outs = Common.run_epoch(model, train_dl, optimizer, 'train', DEVICE, loss_fn=LOSS_FN, clip=GRAD_CLIP, check_accuracy=True)
                val_loss, vacc, _ = Common.run_epoch(model, val_dl, None, 'eval', DEVICE, loss_fn=LOSS_FN, clip=GRAD_CLIP, check_accuracy=True)
                print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')

            # Append the results for this repetition
            train_accs_list.append(tacc)
            test_accs_list.append(vacc)
            train_losses_list.append(train_loss)
            test_losses_list.append(val_loss)

        # Calculate the average results over all repetitions
        avg_train_acc = sum(train_accs_list) / num_repetitions
        avg_test_acc = sum(test_accs_list) / num_repetitions
        avg_train_loss = sum(train_losses_list) / num_repetitions
        avg_test_loss = sum(test_losses_list) / num_repetitions

        # Store the averaged results
        if name not in results:
            results[name] = {"train_sizes": [], "train_accs": [], "test_accs": [], "train_losses": [], "test_losses": []}
        results[name]["train_sizes"].append(train_size)
        results[name]["train_accs"].append(avg_train_acc)
        results[name]["test_accs"].append(avg_test_acc)
        results[name]["train_losses"].append(avg_train_loss)
        results[name]["test_losses"].append(avg_test_loss)



i = 0
for (src_ids, trg_ids, acc_mask) in val_dl:
    src_ids = src_ids.to(DEVICE)  # [batch, seq, vec_size]
    trg_ids = trg_ids.to(DEVICE)
    output = model(src_ids)  # [batch, seq, vec_size]
    if LOSS_FN == 'cosine_distance':
        # find the embedding closest to the output, consider
        # that the output_id
        output_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0), # [1, 1, VOCAB, EMB_DIM]
                                             output.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                             dim=3).argmax(dim=2)
    else:
        output_ids = output.argmax(dim=2)
    for _ in range(output.shape[0]):
        i += 1
        if i > 10:
            BRK
        print('----------')
        print(f'inp: {tokenizer.decode(src_ids[i].tolist())}')
        print(f'trg: {tokenizer.decode(trg_ids[i].tolist())}')
        print(f'out: {tokenizer.decode(output_ids[i].tolist())}')

    BRK




'''

# START_BLOCK_2
DEBUG=True
val_loss, vacc, _ = Common.run_epoch(model, val_dl, None, 'eval', DEVICE, loss_fn=LOSS_FN, clip=GRAD_CLIP, check_accuracy=True)
# END_BLOCK_2

'''

##########


if True:
    # Define nicer colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot accuracy comparison
    for i, name in enumerate(results):
        color = colors[i % len(colors)]
        # ax1.plot(results[name]["train_sizes"], results[name]["train_accs"], marker='o', linestyle='--', color=color)
        ax1.plot(results[name]["train_sizes"], results[name]["test_accs"], marker='o', linestyle='-', color=color, label=name)
    ax1.set_xlabel("Total Training Samples")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.set_title("Accuracy")

    # Plot loss comparison
    for i, name in enumerate(results):
        color = colors[i % len(colors)]
        # ax2.plot(results[name]["train_sizes"], results[name]["train_losses"], marker='o', linestyle='--', color=color)
        ax2.plot(results[name]["train_sizes"], results[name]["test_losses"], marker='o', linestyle='-', color=color, label=name)
    ax2.set_xlabel("Total Training Samples")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.set_title("Loss")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()



# tensor = torch.randn(2, 3, 4)
# e = tensor.unsqueeze(1).expand(2, 3, 3, 4).reshape(2, 9, 4)
# r = tensor.repeat(1, 3, 1)
# print(torch.allclose(e, r))


# Train BATCH_SIZE=1
if False:
    os.environ['DEBUG_MULTIHEAD_ATTENTION'] = 'False'

    os.environ['DEBUG_MULTIHEAD_ATTENTION'] = 'True'
    train_loss, tacc, outs = Common.run_epoch(model, train_dl, optimizer, 'train', DEVICE, loss_fn=LOSS_FN, clip=GRAD_CLIP, check_accuracy=True)


# Visualize Attention
if False:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    train_dl, val_dl = create_dataloaders(batch_size=1)
    for epoch in range(NUM_EPOCHS):
        train_loss, tacc, outs = Common.run_epoch(model, train_dl, optimizer, 'train', DEVICE, loss_fn=LOSS_FN, clip=GRAD_CLIP, check_accuracy=True)
        val_loss, vacc, _ = Common.run_epoch(model, val_dl, None, 'eval', DEVICE, loss_fn=LOSS_FN, clip=GRAD_CLIP, check_accuracy=True)
        print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')




# symbols = torch.arange(2 * 20 * 5).reshape(2, 20, 5)
# batch_size = 2
# n = 20
# s = 3
# start_indices = torch.randint(0, n - s + 1, (batch_size,))
# seq_range = torch.arange(s)
# indices = start_indices.unsqueeze(1) + seq_range.unsqueeze(0)
# subsequences = symbols[torch.arange(batch_size).unsqueeze(1), indices]


# START_BLOCK_3


# END_BLOCK_3
