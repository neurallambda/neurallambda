'''.

Avoiding Memorization.

Advance from previous file, do 3 things:

* Pair down the original work (previous file) to a minimal demonstration
  - instead of 2 streams of input, will variable-indirection work with 1 stream of input?

    RESULTS: actually, variable indirection not even needed. NAND stuff can
    learn mappings just fine. I previously thought they couldn't, and that var
    indirection helped, but I must have done something wrong, and no var
    indirection is needed.

  - instead of doing cos-sim with the `attend` symbol, can we just pass in a
    mask vector, like the one that occurs from the cossim?

* Can it combine with FFNN and prevent memorization / promote algorithmic generalization?

* Demonstrate a best-effort FFNN without these techniques

RESULTS:

- Variable indirection seems really interesting, IE, an input value refers to a
  variable name, referencing a value somewhere else. The NSymbolic stuff eats
  this for breakfast.

- Distractors Punish: running_sum is ideally calculated from prev sum + new
  number. Allowing a superfluous var to be part of that calc kills
  perf. Interpolating with True allows to ignore an input. This can be done at
  the entire input level, IE "always ignore", or per symbol match, IE, turn `A
  AND B AND Irrelevant` into `A AND B AND True`.

- Normalization matters with FFNN. I wasn't normalizing running_sum after adding
  summed redundant calculations to it, causing it to grow. This manifest as an
  interpolation block down weighting the running_sum, whereas I expected it to
  have full weight.

'''

import torch
import neurallambda.symbol as Sym
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Union, Tuple

import torch
import random
from datasets import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import neurallambda.stack as S
import neurallambda.latch as L
import neurallambda.queue as Q
from torch import einsum
from neurallambda.torch import cosine_similarity
import torch.nn.functional as F
from torch.nn.functional import elu, selu, gelu, leaky_relu
import neurallambda.symbol as Sym
import copy
from neurallambda.tensor import CosineSimilarity, Weight, ReverseCosineSimilarity
from neurallambda.torch import NormalizedLinear, Fn, Parallel, Cat, Stack, Diagnose, Id
import re
import numpy as np
import math
import torch.fft
import time
from torch import pi
from neurallambda.util import format_number



DEVICE = 'cuda'
torch.set_printoptions(precision=3, sci_mode=False)
torch.manual_seed(152)

DEVICE = 'cuda'
VEC_SIZE = 64
BATCH_SIZE = 100
NUM_EPOCHS = 30
LR = 1e-2
WD = 0.0

REDUNDANCY = 10
R = REDUNDANCY

NAND_CLIP = 'abs'
NAND_BIAS = 3.0

try:
    CONTINUE
except:
    CONTINUE = False  # on first pass, can't continue

'''
CONTINUE = True
CONTINUE = False
'''


##################################################
# NAND Lib

class Interpolate(nn.Module):
    def __init__(self, n_interpolands, shape):
        super(Interpolate, self).__init__()
        self.n_interpolands = n_interpolands
        self.shape = shape
        self.weights = nn.Parameter(torch.randn((n_interpolands,) + shape) * 1e-3)

    def forward(self, method='softmax', hard=False):
        # inps  = [batch, *shape]
        assert method in {'linear', 'softmax', 'gumbel_softmax'}
        if method == 'softmax':
            assert not hard, 'hard is not compatible with softmax. Use gumbel_softmax instead'
        if method == 'linear':
            mn = self.weights.min(dim=0).values
            mx = self.weights.max(dim=0).values
            gs = (self.weights - mn) / (mx - mn)
        elif method == 'softmax':
            gs = F.softmax(self.weights, dim=0)
        elif method == 'gumbel_softmax':
            gs = F.gumbel_softmax(self.weights, dim=0, hard=hard)
        return gs


# @@@@@@@@@@

if False:
    n_interpolands = 3
    shape = (1, 1)  # Simplified shape for easier verification
    module = Interpolate(n_interpolands, shape)

    # favor the first and the last interpoland more than the middle
    with torch.no_grad():
        module.weights.data = torch.tensor([[[1.0]], [[0.0]], [[1.0]]])

    inps = [torch.tensor([[[1.0]]]), torch.tensor([[[2.0]]]), torch.tensor([[[3.0]]])]

    favored_values_counts = [0, 0, 0]  # Count how often each interpoland's influence dominates
    n_runs = 1000

    for _ in range(n_runs):
        output = module(inps, method='gumbel_softmax', hard=True)
        # Find which of the input values the output is closest to
        closest_inp_idx = (torch.abs(output - torch.stack(inps)).squeeze()).argmin().item()
        favored_values_counts[closest_inp_idx] += 1
    print(favored_values_counts)

    # Assert that the output favored the 1st and 3rd interpolands significantly more often
    # than the 2nd one, given the parameter settings.
    assert favored_values_counts[0] > n_runs * 0.4 and favored_values_counts[2] > n_runs * 0.4, \
        "Interpolation did not favor the correct interpolands as expected."

# @@@@@@@@@@


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
    def __init__(self, vec_size, n_vecs, n_choices, clip='leaky_relu', nand_bias=3.0):
        super(NAND, self).__init__()
        assert clip in {'leaky_relu', 'abs', 'none'}

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.clip = clip

        self.weight = nn.Parameter(torch.randn(n_choices, vec_size * n_vecs))
        self.interp = Interpolate(3, (n_choices, n_vecs))

    def forward(self, query: Union[List[torch.Tensor], torch.Tensor], eps=1e-6):
        # handle either lists or pre-hstacked inputs
        if isinstance(query, list):
            # query = torch.hstack(query)
            query = torch.cat(query, dim=-1)

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

        interp = self.interp(method='softmax')
        sinp = torch.stack([
            cos_sim,                  # Params Match
            1 - cos_sim,              # NOT Params Match
            torch.ones_like(cos_sim), # Ignore (IE becomes: ... AND True)
        ], dim=1) # [batch, n_interpoland, n_choices, n_vecs]

        interpolated = einsum('inv, binv -> bnv', interp, sinp)

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


##################################################
# Run Training

def train_and_report(n_choices, redundancy, vec_size, Model, model=None, *args, **kwargs):
    print('------------------------------')
    print(f'Model = {str(Model)},  n_choices={n_choices}, redundancy={redundancy}',)

    # output choices
    if model is None:
        model = Model(vec_size, n_choices, redundancy)
        model.cuda()

    if Model == SymModel:
        set_weights(
            model.summer,
            model.syms,
            [
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==0], 0),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==1], 1),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==2], 2),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==3], 3),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==4], 4),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==5], 5),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==6], 6),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==7], 7),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==8], 8),
                ([((i, j), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==9], 9),

                # default
                # ([(('X', 'X'), (0, 0))]*10, 'X'),
            ], 10, confidence=5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Params: {format_number(n_params)}')

    #####
    # Train
    opt_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(opt_params, lr=LR, weight_decay=WD)
    train_losses = []
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for _, batch in enumerate(train_dl):
            # type(inps)=list, len(inps)=sequence len
            # type(inps[0])=list, len(inps[0])=input tuple len
            # type(inps[0][0])=tensor, inps[0][0].shape = [batch, vec_size]
            inps = batch['inps']
            trg = torch.stack(batch['outs'], dim=1) # list of batch of steps

            # TEST NOISE
            with torch.no_grad():
                NOISE_LVL = 1e-2
                for i in range(len(inps)): # iter over sequence
                    for ti in range(len(inps[i])): # iter over tuple
                        inps[i][ti] = inps[i][ti] + torch.randn_like(inps[i][ti]) * NOISE_LVL
                    # trg = trg + torch.randn_like(trg) * NOISE_LVL

            optimizer.zero_grad()
            output = model(inps)

            # LOSS
            loss = (1 - F.cosine_similarity(output, trg, dim=2)).mean()
            with torch.no_grad():
                train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_dl)
        print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')
    end = time.time()
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}  |  time={end - start:>.2f}')

    model.eval()
    correct = 0
    n = 0
    test_losses = []
    with torch.no_grad():
        for _, batch in enumerate(test_dl):
            inps = batch['inps'] # list of batch of steps
            trg = torch.stack(batch['outs'], dim=1)
            output = model(inps)
            # LOSS
            loss = (1 - F.cosine_similarity(output, trg, dim=2)).mean()
            test_losses.append(loss.item())

            batch_size = output.size(0)
            seq_len = output.size(1)
            for bi in range(batch_size):
                for si in range(seq_len):
                    trg = batch['uouts'][si][bi].item() # seq first, then batch
                    uout = unproject(output[bi][si])
                    n += 1
                    if trg == uout:
                        correct += 1

    print(f'acc: {correct / n:>.3f}')

    return model, train_losses, test_losses


##################################################
# Control Model

class NNModel(nn.Module):
    '''Control model, using standard FFNN. Hm, I'm actually adding
    unconventional stuff and it's helping.'''
    def __init__(self, vec_size, n_choices, redundancy, *args, **kwargs):
        super(NNModel, self).__init__()

        self.vec_size = vec_size
        self.n_choices = n_choices
        self.redundancy = redundancy

        H = 64
        n_vecs = 4

        self.interp = Interpolate(3, (n_vecs,))

        self.choice = nn.Sequential(
            nn.Linear(n_vecs * vec_size, H), # running_sum + all inputs
            nn.ReLU(),
            # nn.Linear(H, H),
            # nn.ReLU(),
            nn.Linear(H, n_choices * redundancy),
            nn.ReLU(),
        )

        # # use choice to select an output vec
        self.vecs = nn.Parameter(torch.randn(n_choices, vec_size))

    def forward(self, inps):
        # type(inps)=list, len(inps)=sequence len
        # type(inps[0])=list, len(inps[0])=input tuple len
        # type(inps[0][0])=tensor, inps[0][0].shape = [batch, vec_size]
        batch_size = inps[0][0].size(0)
        seq_len = len(inps)
        device = inps[0][0].device

        outputs = []
        running_sum = torch.zeros(batch_size, self.vec_size, device=device) + 1e-6
        for i in range(seq_len):
            attend_var, x_name, x = inps[i]
            inp = torch.stack([running_sum,
                               F.normalize(x, dim=-1),
                               F.normalize(x_name * attend_var, dim=-1), # necessary for indirection task
                               torch.randn_like(attend_var) # experiment, does interpolater ignore?
                               ], dim=1) # [batch, n_vecs, vec_size]

            interp = self.interp(method='softmax') # [n_interpolands, n_vecs]
            sinp = torch.stack([inp,
                                torch.zeros_like(inp), # interpolate to 0 to ignore
                                torch.randn_like(inp), # experiment, does interpolater ignore?
                                ], dim=1) # [batch, n_interpolands, n_vecs, vec_size]
            inp = einsum('in, binv -> bnv', interp, sinp)

            inp = inp.flatten(start_dim=1)
            choices = self.choice(inp)
            running_sum = torch.einsum('cv, bcr -> bv', self.vecs, choices.view(batch_size, self.n_choices, self.redundancy))
            running_sum = F.normalize(running_sum, dim=-1)
            outputs.append(running_sum)
        outputs = torch.stack(outputs, dim=1)
        return outputs


##################################################
# Symbolic Model

def set_weights(nand: nn.Module,
                out_vecs: torch.Tensor,
                mappings: List[Tuple[Tuple[str, str], Tuple[int, int], str]],
                redundancy,
                confidence=3):
    '''set weights in nand and out_vecs according to mappings. confidence scales
    the initial nand_weight, which gets sigmoided. When confidence=3,
    sigmoid(3)=0.953, which is confident but with reasonable gradients.

    Redundancy is handled by view-wrapping tensors, ie, redundant elems are not
    in adjacent indices.

    '''

    n_choices = nand.weight.size(0) // redundancy

    with torch.no_grad():

        nand.interp.weights[1, :] = torch.tensor([confidence])

        temp_weight = torch.zeros(n_choices, redundancy, nand.weight.size(1), device=nand.weight.device)
        temp_nand_weight = torch.zeros(n_choices, redundancy, nand.interp.weights[1].size(1), device=nand.interp.weights.device)

        for choice_i, (vals, out) in enumerate(mappings):
            out_vecs[:, choice_i] = project(out) # no redundancy

            # add all the vecs into the matcher vec
            for r_i_, (ins, nand_weight) in enumerate(vals):
                # TODO: modulo redundancy is buggy in nand_weight if it wraps around
                r_i = r_i_ % redundancy
                temp_weight[choice_i, r_i, :] += torch.concat([project(x) for x in ins])
                temp_nand_weight[choice_i, r_i, :] = torch.tensor([confidence if x==1 else -confidence for x in nand_weight]).to(DEVICE)

        upto = len(mappings) * redundancy  # don't overwrite rows for which no mappings were set
        nand.weight[:upto] = temp_weight.flatten(start_dim=0, end_dim=1)[:upto]
        nand.interp.weights[1, :upto] = temp_nand_weight.flatten(start_dim=0, end_dim=1)[:upto]


class SymModel(nn.Module):
    def __init__(self, vec_size, n_choices, redundancy):
        super(SymModel, self).__init__()

        self.vec_size = vec_size
        self.n_choices = n_choices
        self.redundancy = redundancy

        # self.syms = nn.Parameter(torch.randn(n_choices, vec_size))
        self.syms = torch.randn(vec_size, n_choices)

        self.summer = NAND(vec_size, 2, n_choices * redundancy, clip='abs')
        with torch.no_grad():
            self.summer.interp.weights[0, :] += 5 # bias toward not NOTed

        n_cos_sim = 2
        n_choices_var_choice = 4
        self.default = nn.Parameter(torch.randn(n_choices_var_choice - 2, vec_size))
        self.var_choice = FwdNAND(n_cos_sim, n_choices_var_choice)

        # Stacks
        N_STACKS = 2
        self.N_STACKS = N_STACKS
        stack_depth = 8
        init_sharpen = 8.0
        self.stacks = nn.ParameterList([])
        self.stack_init_vec = F.normalize(torch.randn(vec_size), dim=0)
        self.zero_offset = 1e-3
        for _ in range(N_STACKS):
            stack = S.Stack(stack_depth, vec_size)
            sharp = nn.Parameter(torch.tensor([init_sharpen]))
            pop_op  = NAND(vec_size, 2, 2 * redundancy, clip=NAND_CLIP, nand_bias=NAND_BIAS)
            push_op = NAND(vec_size, 2, 2 * redundancy, clip=NAND_CLIP, nand_bias=NAND_BIAS)
            push    = NAND(vec_size, 2, n_choices * redundancy, clip=NAND_CLIP, nand_bias=NAND_BIAS)
            params = nn.ParameterList([stack, sharp, pop_op, push_op, push])
            self.stacks.append(params)

        self.dropout = nn.Dropout(0.1)

    def forward(self, all_inps):
        # type(all_inps)=list, len(all_inps)=sequence len
        # type(all_inps[0])=list, len(all_inps[0])=input tuple len
        # type(all_inps[0][0])=tensor, all_inps[0][0].shape = [batch, vec_size]
        batch_size = all_inps[0][0].size(0)
        seq_len = len(all_inps)
        device = all_inps[0][0].device
        self.syms = self.syms.to(device=device)

        # Stacks prep
        self.stack_init_vec = self.stack_init_vec.to(device=device)
        for (stack, _, _, _, _) in self.stacks:
            stack.init(batch_size, self.zero_offset, device)
            stack.push(self.stack_init_vec.expand(batch_size, -1))

        outputs = []
        for i in range(seq_len):
            attend, opcode, xvar, x, yvar, y = all_inps[i]

            var_choice = self.var_choice(torch.stack([torch.cosine_similarity(attend, xvar, dim=1),
                                                      torch.cosine_similarity(attend, yvar, dim=1),], dim=1))

            # Handle Stacks
            all_pops = []

            for (stack, sharp, pop_op, push_op, push_fn) in self.stacks:

                push_val = torch.einsum('bc, bcv -> bv',
                                        var_choice,
                                        torch.cat([x.unsqueeze(1),
                                                   y.unsqueeze(1),
                                                   self.dropout(self.default.expand(batch_size, -1, -1))
                                                   ], dim=1))
                push_val = F.normalize(push_val, dim=-1)

                opinp = [opcode, push_val]
                pops   = pop_op(opinp)
                pushes = push_op(opinp)
                p = stack.pop_or_null_op(sharp,
                                         pops[:,:self.redundancy].max(dim=1).values,
                                         pops[:,self.redundancy:].max(dim=1).values)
                stack.push_or_null_op(sharp,
                                      pushes[:,:self.redundancy].max(dim=1).values,
                                      pushes[:,self.redundancy:].max(dim=1).values,
                                      push_val)

            # Peek at stacks
            peeks = []
            for (stack, _, _, _, _) in self.stacks:
                peeks.append(stack.read())

            summer_choices = self.summer(torch.hstack(peeks))
            out = torch.einsum('vc, bcr -> bv',
                                       self.syms,
                                       self.dropout(summer_choices.view(batch_size,
                                                                        self.n_choices,
                                                                        self.redundancy)))
            out = F.normalize(out, dim=-1)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs


####################
# Dataset

all_symbols =  list(range(10)) + ['a', 'b', 'c', 'd', 'e', 'f', 'push', 'pop', 'nop']
sym_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject

def build_dataset(data_len, seq_len, var_names):
    '''2 streams of inputs, plus an operation on possibly one of them, based on if
    the var name matches. Can be {push, pop, nop}.

    '''
    dataset = []
    for _ in range(data_len):
        uinps = []
        inps = []
        uouts = []
        outs = []

        # hidden, not given as inputs or outputs
        uxstack = []
        uystack = []

        for _ in range(seq_len):
            # switch
            uattend = random.choice(var_names)
            attend = project(uattend)

            uop = random.choice(
                ['push'] +
                ['pop']
                # ['nop']
            )
            op = project(uop)

            # var 1
            uxvar = random.choice(var_names)
            xvar = project(uxvar)
            ux = random.randint(0, 9)
            x = project(ux)

            # var 2
            uyvar = random.choice(list(set(var_names) - {uxvar}))  # enforce no reuse of x var
            yvar = project(uyvar)
            uy = random.randint(0, 9)
            y = project(uy)

            # stacks
            match (uattend, uop):
                case (ua, "push") if ua == uxvar:
                    uxstack.append(ux)
                case (ua, "pop") if ua == uxvar:
                    if len(uxstack)>0: uxstack.pop()
                case (ua, "push") if ua == uyvar:
                    uystack.append(uy)
                case (ua, "pop") if ua == uyvar:
                    if len(uystack)>0: uystack.pop()

            uxpeek = uxstack[-1] if len(uxstack) > 0 else 0
            uypeek = uystack[-1] if len(uystack) > 0 else 0

            uout = (uxpeek + uypeek) %10
            out = project(uout)

            uinp = (uattend, uop, uxvar, ux, uyvar, uy)
            inp = (attend, op, xvar, x, yvar, y)

            uinps.append(uinp)
            inps.append(inp)
            uouts.append(uout)
            outs.append(out)
        dataset.append(dict(uinps = uinps, inps = inps,
                            uouts = uouts, outs = outs))
    return dataset

train_dataset = build_dataset(1000, 12, ['a', 'b'])
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = build_dataset(200, 12, ['d', 'e', 'f'])
test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

'''

# Hand check data
for i in range(10):
    print()
    print(train_dataset[i]['uinps'])
    print(train_dataset[i]['uouts'])

'''

#####
# GO

experiments = [
    {'Model': SymModel, 'redundancy': R, 'n_choices': 10, 'vec_size':VEC_SIZE, 'name': 'SymModel'},
    # {'Model': NNModel,  'redundancy': R, 'n_choices': 10, 'vec_size':VEC_SIZE, 'name': 'FFNN'},
]

'''
CONTINUE = True
'''

if not CONTINUE:
    models = []
    all_train_losses = [[] for _ in range(len(experiments))]
    all_test_losses  = [[] for _ in range(len(experiments))]

for i, e in enumerate(experiments):
    if CONTINUE:
        print('CONTINUING TRAINING MODEL')
        model = models[i]
    else:
        print('INITIALIZING NEW MODEL')
        model = None
    model, train_losses, test_losses = train_and_report(model=model, **e)
    models.append(model)
    all_train_losses[i] += train_losses
    all_test_losses[i] += test_losses


# Default to continuing after a run
if not CONTINUE:
    CONTINUE = True


##########
# Viz

colors = [
    ("#e5e10d", "#edea47"), # Gold
    ("#e10de5", "#ea47ed"), # Pink
    ("#0de5e1", "#47edea"), # Teal
    ("#A77508", "#F9E79F"),  # Deep Gold
    ("#AA4499", "#D7B5D8"),  # Purple
    ("#AA0A3C", "#FC9272"),  # Crimson
    ("#882255", "#D4B9DA"),  # Wine
    ("#117733", "#74C476"),  # Dark Green
    ("#332288", "#B3B3E8"),  # Indigo
]
plt.figure(figsize=(10, 6))  # Set the figure size for better readability

for index, losses in enumerate(all_train_losses):
    # Training Losses
    cs = colors[index % len(colors)] # cycle colors
    plt.plot(losses, color=cs[1], label=f"{experiments[index]['name']} Train")

    # Test Loss
    avg_loss = np.mean(losses)
    plt.hlines(avg_loss, 0, len(all_train_losses[index])-1, colors=cs[0], linestyles='dashed', label=f"{experiments[index]['name']} Test Avg")

plt.title("Training and Test Losses Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
