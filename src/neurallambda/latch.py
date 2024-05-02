'''

A Neural Latch.

'''


from torch import einsum, tensor, allclose
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.util import transform_runs
import neurallambda.debug as D
from neurallambda.torch import cosine_similarity

class Latch(nn.Module):
    '''A Latch has 3 learned parameters, each vectors: predicate, true,
    false. It learns to emit false until the predicate has been matched, and
    then emits true for the rest of time.

    '''
    def __init__(self, vec_size):
        super(Latch, self).__init__()
        self.vec_size = vec_size
        self.predicate = nn.Parameter(torch.randn((vec_size,)) * 1e-2)
        self.true = nn.Parameter(torch.randn((vec_size,)) * 1e-2)
        self.false = nn.Parameter(torch.randn((vec_size,)) * 1e-2)

    def forward(self, inp):
        # inp    : [batch_size, vec_size]
        # output : [batch_size, vec_size]
        assert inp.ndim == 2
        predicate = self.predicate.unsqueeze(0)
        true = self.true
        false = self.false
        br_predicate, br_inp = torch.broadcast_tensors(predicate.real, inp.real)
        matched = cosine_similarity(br_predicate, br_inp, dim=1)
        return (
            torch.einsum('v, b -> bv', true, matched) +
            torch.einsum('v, b -> bv', false, 1 - matched)
        )

class DataLatch(nn.Module):
    def __init__(self, vec_size, init_scale, dropout=None):
        super(DataLatch, self).__init__()
        self.vec_size = vec_size
        self.init_scale = init_scale
        self.enable = nn.Parameter(torch.randn(vec_size, ) * init_scale)
        self.dropout = dropout
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)

    def forward(self, state, enable, data):
        # state  : [batch_size, vec_size]
        # enable : [batch_size, vec_size]
        # data   : [batch_size, vec_size]
        assert state.shape == enable.shape == data.shape
        assert state.ndim == 2

        br_self_enable, br_enable = torch.broadcast_tensors(self.enable.unsqueeze(0), enable)
        matched = cosine_similarity(br_self_enable, br_enable, dim=1)
        matched = torch.nn.functional.elu(matched)
        # matched = torch.nn.functional.selu(matched)
        # matched = torch.nn.functional.gelu(matched)
        # matched = torch.nn.functional.leaky_relu(matched)

        if_matched      = einsum('bv, b -> bv', data, matched)
        if_not_matched  = einsum('bv, b -> bv', state, 1-matched)

        out = if_matched + if_not_matched
        if self.dropout is not None:
            out = self.dropout(out)
        return out
