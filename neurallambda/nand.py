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
from typing import List
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

    '''
    def __init__(self, vec_size, n_vecs, n_choices, redundancy):
        super(NAND, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.method = method

        self.weight = nn.Parameter(torch.randn(n_choices * redundancy,
                                               vec_size * n_vecs))

        # interpolation factors. 1 -> cossim. 0 -> 1-cossim
        self.nand_weight = nn.Parameter(torch.rand(n_choices * redundancy,
                                                   n_vecs))

        # Normalize the main weights
        with torch.no_grad():
            self.weight[:] = F.normalize(self.weight, dim=1)

    def forward(self, query: torch.Tensor):
        # handle either lists or pre-hstacked inputs
        if isinstance(query, list):
            query = torch.hstack(query)

        # [1, n_choices * redundancy, n_vecs, vec_size]
        weight_ = self.weight.view(-1, self.n_vecs, self.vec_size).unsqueeze(0)

        # [batch, 1, n_vecs, vec_size]
        query_ = query.view(-1, self.n_vecs, self.vec_size).unsqueeze(1)

        # [batch, n_choices * redundancy, n_vecs]
        cos_sim = torch.cosine_similarity(query_, weight_, dim=3)

        # interpolate between cos_sim and 1-cos_sim
        nw = self.nand_weight.unsqueeze(0).sigmoid()  # Expand nand_weight for broadcasting
        interpolated = nw * cos_sim + (1 - nw) * (1 - cos_sim)  # [batch, n_choices * redundancy, n_vecs]

        # product along n_vecs dimension to aggregate the NAND logic
        output = interpolated.prod(dim=2)  # [batch, n_choices * redundancy]
        return output


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