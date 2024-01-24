'''

A Neural Latch.

'''


from torch import einsum, tensor, allclose
import neurallambda.hypercomplex as H
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.util import transform_runs
import neurallambda.debug as D
import neurallambda.hypercomplex as H

class Latch(nn.Module):
    def __init__(self, vec_size, number_system):
        super(Latch, self).__init__()
        self.vec_size = vec_size
        self.number_system = number_system
        self.predicate = nn.Parameter(number_system.randn((vec_size,)) * 1e-2)
        self.true = nn.Parameter(number_system.randn((vec_size,)) * 1e-2)
        self.false = nn.Parameter(number_system.randn((vec_size,)) * 1e-2)

    def forward(self, inp):
        # inp    : [batch_size, vec_size]
        # output : [batch_size, vec_size]
        N = self.number_system
        predicate = N.to_mat(self.predicate).unsqueeze(0)
        true = N.to_mat(self.true)
        false = N.to_mat(self.false)
        matched = N.cosine_similarity(predicate.real, inp.real, dim=1)

        matched = matched.squeeze(-1).squeeze(-1) # squeeze hypercomplex dims
        return (
            torch.einsum('vqr, b -> bvqr', true, matched) +
            torch.einsum('vqr, b -> bvqr', false, 1 - matched)
        )

class DataLatch(nn.Module):
    def __init__(self, ):
        super(DataLatch, self).__init__()
        self.enable = nn.Parameter(torch.randn(VEC_SIZE, ) * INIT_SCALE)
        self.dropout = torch.nn.Dropout(0.01)

    def forward(self, state, enable, data):
        # state  : [batch_size, vec_size]
        # enable : [batch_size, vec_size]
        # data   : [batch_size, vec_size]

        matched = cosine_similarity(self.enable.unsqueeze(0), enable, dim=1)
        matched = torch.nn.functional.elu(matched)
        # matched = torch.nn.functional.selu(matched)
        # matched = torch.nn.functional.gelu(matched)
        # matched = torch.nn.functional.leaky_relu(matched)

        if_matched      = einsum('bv, b -> bv', data, matched)
        if_not_matched  = einsum('bv, b -> bv', state, 1-matched)

        out = if_matched + if_not_matched
        return self.dropout(out)
