'''

The first t10_swap found antiparallel vecs in a sequence and swapped em

In t10_swap_02, i want to do the same on an ordered set

RESULTS:

- it helped to randomly select symbols, instead of just `symbols[:S]`
- don't sample_with_replacement. this fixed everything


NOTES:

* responds well to dropout on embeddings. val acc increases, stability increases.
* weight decay hurts
* having a large vocabulary really helps
* Transformer3D has nebulous benefit
* TransformerDefeasible has nebulous benefit

TODO: [ ] sprinkle dropout throughout
TODO: [ ] visualize things
TODO: [ ] symmetry breaking, swap index (3, 8) or (8, 3)
TODO: [X] symmetry breaking, if there are multiple of equal values. SOLUTION: for now, sample dataset without replacement.


'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

import matplotlib.pyplot as plt
import math
import warnings

import neurallambda.model.transformer01 as Transformer2D
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
# Data

# lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
# lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' ')
lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9 aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt uu vv ww xx yy zz 00 11 22 33 44 55 66 77 88 99 aaa bbb'.split(' ')  # len=100
# lang = '0 1 2 3 4 5 6 7 8 9'.split(' ')
train_lang = lang
val_lang = lang

MIN_LENGTH = 3
TRAIN_MAX_SEQUENCE_LENGTH = 10
VAL_MAX_SEQUENCE_LENGTH = 100



##################################################
#

def swap(x, swap1, swap2):
    ''' swap1 and swap2 are softmax vectors (think onehot) of rows of x that will
    be swapped. '''
    # Combine swap1 and swap2 into a single matrix
    P = torch.einsum('bx,by->bxy', swap1, swap2)
    P = P + P.transpose(1, 2)  # swap both directions
    # identity matrix to keep non-swapped data
    Id = torch.diag_embed(torch.ones_like(swap1) - swap1 - swap2)
    x_swapped = torch.einsum('bij,bjd->bid', P + Id, x)
    return x_swapped




# # @@@@@@@@@@
# # Hand check generated data
# x_test, y_test, swap1_test, swap2_test, swap1_oh_test, swap2_oh_test = generate_data(num_samples=10, sequence_length=5, embedding_size=128)

# for i in range(10):
#     print('----------')
#     visualize_swap(x_test[i], y_test[i])
# print(x_test[0, :, :5])
# print(y_test[0, :, :5])

# # Test that the expected swaps took place
# for i in range(10):
#     assert torch.allclose(x_test[i, swap1_test[i]], y_test[i, swap2_test[i]]), f"Swap mismatch at sample {i}"
#     assert torch.allclose(x_test[i, swap2_test[i]], y_test[i, swap1_test[i]]), f"Swap mismatch at sample {i}"
#     cs1 = torch.cosine_similarity(x_test[i, swap1_test[i]], y_test[i, swap2_test[i]], dim=0) # orthogonal
#     cs2 = torch.cosine_similarity(x_test[i, 0], y_test[i, 1], dim=0) # random similarity

# # Test that swap1_oh_test and swap2_oh_test are one-hot encoded correctly
# assert torch.all(swap1_oh_test.sum(dim=1) == 1), "swap1_oh_test is not one-hot encoded"
# assert torch.all(swap2_oh_test.sum(dim=1) == 1), "swap2_oh_test is not one-hot encoded"

# # Test that swap1_oh_test and swap2_oh_test match swap1_test and swap2_test
# assert torch.all(swap1_oh_test.argmax(dim=1) == swap1_test), "swap1_oh_test does not match swap1_test"
# assert torch.all(swap2_oh_test.argmax(dim=1) == swap2_test), "swap2_oh_test does not match swap2_test"

# print("All assertions passed!")

# Q = x_test[0]
# K = x_test[0]
# Id = torch.eye(10, 10)
# swap1 = swap1_test[0]
# swap2 = swap2_test[0]
# # cs = torch.einsum('sd, td -> st', F.normalize(Q, dim=1), F.normalize(K, dim=1))
# cs = torch.einsum('sd, td -> st', Q, K)
# # plt.imshow(1 - (cs).abs())
# plt.imshow(cs)
# plt.show()

# BRK
# # @@@@@@@@@@



##################################################


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
                              'sorter'}

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)
        self.pos_encoding = Transformer.positional_encoding(emb_dim)

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


            self.layers = nn.ModuleList([
                # DecoderLayerDefeasible(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                #              use_wq=False, use_wk=False, use_wv=False, use_wout=False)
                # if i == 0 else
                Transformer2D.DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
                                       use_wq=use_wq, use_wk=use_wk, use_wv=use_wv, use_wout=use_wout)
                for i in range(num_layers)])



            self.swap = nn.Sequential(
                nn.Linear(emb_dim, 2, bias=False)
                # nn.Linear(emb_dim, 16),
                # nn.ReLU(),
                # nn.Linear(16, 2)
            )

        # self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
        # self.fc_out = nn.Linear(emb_dim, emb_dim, bias=False)
        if LOSS_FN in {'cross_entropy', 'cross_entropy_select_from_inputs'}:
            self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
        elif LOSS_FN == 'cosine_distance':
            self.fc_out = nn.Linear(emb_dim, emb_dim, bias=False)

        self.dropout = nn.Dropout(0.0)

    def forward(self, xs_ids):
        B, S = xs_ids.shape
        device = xs_ids.device
        if self.model_type in {'sorter', 'abstractor_all', 'abstractor_first', 'abstractor_last'}:
            self.symbol_embeddings = self.symbol_embeddings.to(device)

        xs = self.embeddings(xs_ids) * math.sqrt(self.emb_dim)
        xs = self.dropout(xs)

        pos = self.pos_encoding[:S, :].to('cuda')

        for i in range(self.num_recurrence):
            in_xs = xs  # save for use in sorter
            for j, layer in enumerate(self.layers):

                if self.model_type == 'transformer':
                    q = xs
                    k = xs
                    l = xs
                    v = xs
                elif self.model_type == 'abstractor_all':
                    q = xs
                    k = xs
                    l = xs
                    v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                elif self.model_type == 'abstractor_first':
                    q = xs
                    k = xs
                    l = xs
                    if j == 0:
                        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                    else:
                        v = xs
                elif self.model_type == 'abstractor_last':
                    q = xs
                    k = xs
                    l = xs
                    if j == self.num_recurrence - 1:
                        v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                    else:
                        v = xs
                elif self.model_type == 'sorter':
                    q = xs
                    k = xs

                    if j == 0:
                    # if j == self.num_recurrence - 1:
                    # if True:
                        l = self.l_embeddings(xs_ids) * math.sqrt(self.emb_dim)
                        # v = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                        # v = xs

                        start_indices = torch.randint(0, self.max_symbols - S + 1, (B,))
                        seq_range = torch.arange(S)
                        indices = start_indices.unsqueeze(1) + seq_range.unsqueeze(0)
                        # breakpoint()
                        v = self.symbol_embeddings.unsqueeze(0).expand(B, self.max_symbols, self.emb_dim)[torch.arange(B).unsqueeze(1), indices]

                    else:
                        l = xs
                        v = xs

                if j == 0:  # add at first layer of each recurrence
                # if True:
                    q = q + pos
                    k = k + pos
                    l = l + pos
                    v = v + pos

                if self.model_type == 'sorter' and j == 0:
                # if self.model_type == 'sorter' and j == self.num_recurrence - 1:
                    # attn = None
                    # attn = 'sigmoid'
                    attn = 'softmax'
                else:
                    attn = self.attn_nonlin

                # l = self.symbol_embeddings[:S].unsqueeze(0).expand(B, -1, -1)
                # l = xs

                if isinstance(layer, DecoderLayer3D):
                    xs = layer(q, k, l, v, mask=None, attn_nonlin=attn)
                else:
                    xs = layer(q, k, v, mask=None, attn_nonlin=attn)

            if self.model_type == 'sorter':
                swap_ixs = self.swap(xs)

                s1 = swap_ixs[:, :, 0]
                s2 = swap_ixs[:, :, 1]

                swap1 = torch.softmax(s1, dim=1)
                swap2 = torch.softmax(s2 ** 2, dim=1)

                # swap1 = F.gumbel_softmax(swap_ixs[:, 0:S, 0], dim=1, hard=False)
                # swap2 = F.gumbel_softmax(swap_ixs[:, 0:S, 1], dim=1, hard=False)

                # swap1 = F.gumbel_softmax(swap_ixs[:, 0:S, 0], dim=1, hard=True)
                # swap2 = F.gumbel_softmax(swap_ixs[:, 0:S, 1], dim=1, hard=True)

                xs = swap(in_xs, swap1, swap2)

        ##########
        # outputs

        if LOSS_FN == 'cosine_distance':
            assert xs.size(0) == xs_ids.size(0)
            assert xs.size(1) == xs_ids.size(1)
            return xs

        elif LOSS_FN == 'cross_entropy':
            xs = self.fc_out(xs)

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


num_repetitions = 2  # number of times to repeat and average a result

embedding_size = 256

LR = 3e-4
NUM_EPOCHS = 20
VAL_NUM_SAMPLES = 200

train_sizes = [400, 800, 1200, 1600, 2000, 2400, 2800] # , 2000, 4000, 8000]
architectures = [

    {"name": "Transformer",
     "init_params": {"model_type": "transformer", "num_heads": 4, "num_layers": 2, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

    {"name": "Neurallambda",
     "init_params": {"model_type": "sorter", "num_heads": 4, "num_layers": 2, "dim_feedforward": 128, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

    # {"name": "Abstractor_First",
    #  "init_params": {"model_type": "abstractor_first", "num_heads": 4, "num_layers": 2, "dim_feedforward": 64, 'num_recurrence': 1, 'attn_nonlin': 'softmax'}},

]

# Collect results for each training size and architecture
results = {}
for train_size in train_sizes:
    train_raw = Data.swap_max_and_min(train_size, MIN_LENGTH, TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang, mask_type='all', sample_with_replacement=False)
    # train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))

    val_raw = Data.swap_max_and_min(VAL_NUM_SAMPLES, MIN_LENGTH, VAL_MAX_SEQUENCE_LENGTH, lang=val_lang, mask_type='all', sample_with_replacement=False)
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
