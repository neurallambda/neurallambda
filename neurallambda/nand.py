'''.

CosSim things together, treat the resulting scalars as bool values, that can be
aggregated via AND (multiplied), and possibly NOTed first (1 - cos_sim). The
NOTing weights can be learned.

There are 2 versions:

1. NAND: Do cos_sim with internal weights, and combine using internal
     nand_weights.

2. FwdNAND: Cos_sim is done elsewhere in the network (ie cos_sim across latent
     vars, and not weight params), and passed into FwdNAND. FwdNAND just has the
     nand_weights for aggregating the cos_sims.


PROVENANCE:
  experiments/t05_hyperdimensional_nand_02_substitution.py

'''

from neurallambda.torch import cosine_similarity
from neurallambda.util import format_number
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class NAND(nn.Module):
    '''Given n_vecs in, return a bunch of similarities to internal weights.

    But it's not just the sim of input vecs to weight vecs. We will collect the
    similarities of input vecs to respective weight vecs, and then possibly NOT
    them before ANDing them all together.

    Have a set number of n_choices, and each sub-comparison can interpolate
    between the the not/not not'd version of the input before AND-aggregation.


    NOTE: Using redundant NAND computations which result in the same output
    vector seems to help. This module used to handle redundancy, but, the
    implementation was incorrect, and I think the correct way of accomplishing
    this is by multiplying `n_choice` by your desired redundancy, and then
    handling the aggregation outside of this module. Ex:

    VEC_SIZE = 256
    BATCH_SIZE = 5
    N_CHOICES = 13
    REDUNDANCY = 3

    vecs = torch.randn(VEC_SIZE, N_CHOICES)
    scale = torch.randn(BATCH_SIZE, N_CHOICES * REDUNDANCY)

    out1 = torch.einsum('vc, bcr -> bvr', vecs, scale.view(BATCH_SIZE, N_CHOICES, REDUNDANCY)).sum(dim=-1)
    out2 = torch.einsum('vc, bcr -> bv', vecs, scale.view(BATCH_SIZE, N_CHOICES, REDUNDANCY))
    out3 = torch.einsum('vr, br -> bv', vecs.repeat_interleave(REDUNDANCY, dim=1), scale)  # r_i copies data

    print(out1.shape)
    print(out2.shape)
    print(torch.allclose(out1, out2, rtol=1e-4))
    print(torch.allclose(out1, out3, rtol=1e-4))


    '''
    def __init__(self, vec_size, n_vecs, n_choices, clip='leaky_relu'):
        super(NAND, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.clip = clip

        assert clip in {'leaky_relu', 'abs', 'none'}

        self.weight = nn.Parameter(torch.randn(n_choices, vec_size * n_vecs))

        # Interpolation factor (gets sigmoided).
        #   If nw=1, interpolate toward cossim
        #   If nw=0, interpolate toward 1 - cossim
        self.nand_weight = nn.Parameter(torch.randn(n_choices, n_vecs))

        # # Normalize the main weights
        # with torch.no_grad():
        #     self.weight[:] = F.normalize(self.weight, dim=1)

        #     # init nand_weight to not contain NOTs
        #     NAND_BIAS = 3.0
        #     self.nand_weight[:] = torch.ones_like(self.nand_weight) + NAND_BIAS


    def forward(self, query: Union[List[torch.Tensor], torch.Tensor], eps=1e-6):
        # handle either lists or pre-hstacked inputs
        if isinstance(query, list):
            query = torch.hstack(query)

        # [1, n_choices, n_vecs, vec_size]
        weight_ = self.weight.view(-1, self.n_vecs, self.vec_size).unsqueeze(0)

        # [batch, 1, n_vecs, vec_size]
        query_ = query.view(-1, self.n_vecs, self.vec_size).unsqueeze(1)

        # [batch, n_choices, n_vecs]
        cos_sim = torch.cosine_similarity(query_, weight_, dim=3)

        # During interpolation, if nw=0 and cos_sim=-1, output goes to
        # +2.0. This is a weird behavior, and I think the proper remedy is to
        # clip negative similarities.
        if self.clip == 'leaky_relu':
            cos_sim = F.leaky_relu(cos_sim)
        elif self.clip == 'abs':
            cos_sim = cos_sim.abs()

        # interpolate between cos_sim and 1-cos_sim. This sends 1.0 (parallel) to 0.0
        # (orthogonal) and vice versa.
        nw = self.nand_weight.unsqueeze(0).sigmoid()
        interpolated = nw * cos_sim + (1 - nw) * (1 - cos_sim)  # [batch, n_choices, n_vecs]

        # Dont do this, it sends 1.0 to -1.0 and vice versa, and orthogonal
        # stays orthogonal. This isn't the sense of "NOT" that I want.
        #
        # interpolated = nw * cos_sim + (1 - nw) * (-cos_sim)  # [batch, n_choices, n_vecs]

        # product along n_vecs dimension to aggregate the NAND logic
        outs = interpolated.prod(dim=2)  # [batch, n_choices]

        return outs


class FwdNAND(nn.Module):
    '''NAND in the forward pass relies on cos_sims being passed in, and not
    being generated internally to this module. The NAND module can calculate
    cos_sims against internal weights, but sometimes you want cos_sims between
    inputs, or latents, and not parameters. If so, this module's for you.
    '''

    def __init__(self, n_cos_sim, n_choices):
        super(FwdNAND, self).__init__()

        self.n_cos_sim = n_cos_sim
        self.n_choices = n_choices

        # interpolation factors. 1 -> cossim. 0 -> 1-cossim
        self.nand_weight = nn.Parameter(torch.rand(n_choices, n_cos_sim))

    def forward(self, cos_sims):
        # handle either lists or pre-hstacked inputs

        if isinstance(cos_sims, list):
            cos_sims = torch.stack(cos_sims, dim=1)
        assert cos_sims.size(1) == self.n_cos_sim
        batch_size = cos_sims.size(0)

        cos_sims = cos_sims.unsqueeze(1).expand(-1, self.n_choices, -1)

        # interpolate between cos_sim and 1-cos_sim
        nw = self.nand_weight.sigmoid()
        interpolated = (
            einsum('cs, bcs -> bcs', nw, cos_sims) +
            einsum('cs, bcs -> bcs', (1 - nw), (1 - cos_sims))
        )  # [batch, n_choices, n_cos_sim]

        # product along n_cos_sim dimension to aggregate the NAND logic
        output = interpolated.prod(dim=2)  # [batch, n_choices]

        return output
