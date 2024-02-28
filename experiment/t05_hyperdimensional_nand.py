'''.

Learnable NANDing of input vecs

----------
DESCRIPTION:

* So you can concat 2(+) vectors with hstack.

* But what if you want A AND NOT B?

* I propose you can still concat both vectors. But you'll also need a value to
  do `(1 - cossim_B)`. Then you'll need to do cossim separately on both halves,
  and combine. Or more aptly, since cossim isn't linear, we'll normalize the
  input vectors first, and the comparison vector, so this way cossim is now dot
  product. Then we can do dotpdt on both halves separately. For the NOT B half,
  we'll do 1-cossim_B. Then add cossim_A and cossim_B, and boom, you've got a
  vector that has high similarity to `A AND NOT B`.

*  [0, 1] == Bool
*  relu(cossim(x, y)) == Bool
*  cs1 * cs2 == And
*  cs1 + cs2 == Or
*  1 - cs == Not


----------
RESULTS:

Works amazing. Trains fast. Dropout friendly. Noise in inputs is friendly.

With too-few output vecs, it can still learn to perfect accuracy, demonstrating
superposition.

vec_size of 16 is great.

You can MiTM softmax over the choices to, I suspect, disincentivize
superposition. It takes more params to learn to full accuracy, but works fine,
and I think gives cleaner separation in latent space.



'''

import torch
import neurallambda.symbol as Sym
import torch.nn.functional as F
import torch.nn as nn
from typing import List

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


##################################################
#

'''

* given N variables, there are 2^N combinations where all variables are used, combined with AND, and possibly NOTd.

'''

def combos(N):
    # Total number of combinations is 2^N
    total_combinations = 2**N
    tensor = torch.zeros((total_combinations, N))
    for i in range(total_combinations):
        for j in range(N):
            # Use bitwise shift and AND to determine if the j-th bit of i is set or not
            tensor[i, j] = 0 if (i >> j) & 1 else 1
    return tensor


class NAND_Exhaustive(nn.Module):
    '''Given n_vecs in, return a bunch of softmax'd similarities to internal weights.

    But it's not just the sim of input vecs to weight vecs. We will collect the
    similarities of input vecs to respective weight vecs, and then create every
    combo of NANDing those similarities together.

    2 flavors of this are possible:

    1. Exhaustive (this module): explicitly perform every NAND combo

    2. Dynamic: do not explicitly perform every nand. Have a set number of
                n_choices, and each sub-comparison can interpolate between the
                the not/not not'd version of the input.

    '''
    def __init__(self, vec_size, n_vecs, n_choices, redundancy, method='softmax'):
        super(NAND_Exhaustive, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.method = method
        self.weight = nn.Parameter(torch.randn(n_choices * redundancy, vec_size * n_vecs))

        # init and normalize
        with torch.no_grad():
            self.weight[:] = F.normalize(self.weight, dim=1)

        # [2 ** n_vecs, n_vecs]
        self.combos = combos(n_vecs)

    def forward(self, query: torch.Tensor):
        self.combos = self.combos.to(query.device)

        # shape=[batch, n_vecs, vec_size]
        query_reshaped = query.view(-1, self.n_vecs, self.vec_size)

        # shape=[n_choices * redundancy, n_vecs, vec_size]
        weight_reshaped = self.weight.view(-1, self.n_vecs, self.vec_size)

        query_expanded = query_reshaped.unsqueeze(1)   # [batch, 1, n_vecs, vec_size]
        weight_expanded = weight_reshaped.unsqueeze(0) # [1, n_choices * redundancy, n_vecs, vec_size]

        cos_sim = torch.cosine_similarity(query_expanded, weight_expanded, dim=3)  # [batch, n_choices * redundancy, n_vecs]
        cos_sim = cos_sim.relu()

        # Use broadcasting for NAND operation over all combinations
        # Transform cos_sim to match combos shape: [batch, 1, n_choices * redundancy, n_vecs]
        cos_sim_expanded = cos_sim.unsqueeze(2).expand(-1, -1, self.combos.shape[0], -1)
        combos_expanded = self.combos.unsqueeze(0).unsqueeze(0)  # [1, 1, 2**n_vecs, n_vecs]

        # Apply NAND logic: 1's in combos mean take the cosine similarity, 0's mean take 1-cosine similarity
        transformed = combos_expanded * cos_sim_expanded + (1 - combos_expanded) * (1 - cos_sim_expanded)

        # Result shape: [batch, n_choices * redundancy, 2**n_vecs]
        output = transformed.prod(dim=-1)

        return output


class NAND(nn.Module):
    '''Given n_vecs in, return a bunch of similarities to internal weights.

    But it's not just the sim of input vecs to weight vecs. We will collect the
    similarities of input vecs to respective weight vecs, and then possibly NOT
    them before ANDing them all together.

    2 flavors of this are possible:

    1. Exhaustive: explicitly perform every NAND combo

    2. Dynamic (this module): do not explicitly perform every nand. Have a set
                number of n_choices, and each sub-comparison can interpolate
                between the the not/not not'd version of the input.

    '''
    def __init__(self, vec_size, n_vecs, n_choices, redundancy, method='softmax'):
        super(NAND, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.method = method

        self.weight = nn.Parameter(torch.randn(n_choices * redundancy, vec_size * n_vecs))

        # interpolation factors. 1 -> cossim. 0 -> 1-cossim
        self.nand_weight = nn.Parameter(torch.rand(n_choices * redundancy, n_vecs))

        # Normalize the main weights
        with torch.no_grad():
            self.weight[:] = F.normalize(self.weight, dim=1)

    def forward(self, query: torch.Tensor):

        # handle either lists or pre-hstacked inputs
        if isinstance(query, list):
            query = torch.hstack(query)

        # [batch, 1, n_vecs, vec_size]
        query_ = query.view(-1, self.n_vecs, self.vec_size).unsqueeze(1)

        # [1, n_choices * redundancy, n_vecs, vec_size]
        weight_ = self.weight.view(-1, self.n_vecs, self.vec_size).unsqueeze(0)

        # [batch, n_choices * redundancy, n_vecs]
        cos_sim = torch.cosine_similarity(query_, weight_, dim=3)
        # cos_sim = cos_sim.relu()

        # interpolate between cos_sim and 1-cos_sim
        nand_weight_ = self.nand_weight.unsqueeze(0)  # Expand nand_weight for broadcasting
        interpolated = nand_weight_ * cos_sim + (1 - nand_weight_) * (1 - cos_sim)  # [batch, n_choices * redundancy, n_vecs]

        # product along n_vecs dimension to aggregate the NAND logic
        output = interpolated.prod(dim=2)  # [batch, n_choices * redundancy]
        return output


class SymModel(nn.Module):
    def __init__(self, vec_size, n_vecs, n_choices, redundancy, method='softmax'):
        super(SymModel, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.method = method

        self.choice = NAND(vec_size, n_vecs, n_choices, redundancy, method='softmax')

        # use choice to select an output vec
        self.vecs = nn.Parameter(torch.randn(vec_size, n_choices))

        # Normalize the main weights
        with torch.no_grad():
            self.vecs[:] = F.normalize(self.vecs, dim=0)

        self.dropout = nn.Dropout(0.2)

    def forward(self, query: torch.Tensor):
        choices = self.choice(query)

        # choices = self.dropout(choices)

        # # Experiment with softmax
        # eps = 1e-6
        # choices = (choices).clip(eps, 1-eps)  # note: clips neg similarities
        # choices = torch.log((choices) / (1 - choices))  # maps [0,1] -> [-inf, inf]
        # choices = choices.softmax(dim=1)

        out = torch.einsum('vc, bc -> bv', self.vecs, choices)

        # out = self.dropout(out)

        return out


class NNModel(nn.Module):
    ''' Control model, using standard FFNN '''
    def __init__(self, vec_size, n_vecs, n_choices, *args, **kwargs):
        super(NNModel, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices

        H = 8

        self.ffnn = nn.Sequential(
            nn.Linear(n_vecs * vec_size, H),
            nn.ReLU(),
            nn.Linear(H, n_choices),
            nn.Sigmoid()
        )

        # use choice to select an output vec
        self.vecs = nn.Parameter(torch.randn(vec_size, n_choices))


    def forward(self, query: torch.Tensor):
        choices = self.ffnn(torch.hstack(query))
        return torch.einsum('vc, bc -> bv', self.vecs, choices)


# @@@@@@@@@@

if False:

    VEC_SIZE = 128
    sym_map = Sym.SymbolMapper(VEC_SIZE, Sym.chars + Sym.nums, device=DEVICE)
    project = sym_map.project
    unproject = sym_map.unproject

    a = project('A')
    b = project('B')
    c = project('C')

    query = torch.stack([
        torch.cat([a, b]),
        torch.cat([a, c]),
        torch.cat([c, b]),
    ])

    model = NAND(vec_size=VEC_SIZE, n_vecs=2, n_choices=4, redundancy=1)
    model = model.to(DEVICE)

    with torch.no_grad():
        model.weight[0] = torch.cat([a, b])
        model.weight[1] = torch.cat([a, b])
        model.weight[2] = torch.cat([a, b])
        model.weight[3] = torch.cat([a, b])

        model.nand_weight[0] = torch.tensor([0, 0])
        model.nand_weight[1] = torch.tensor([0, 1])
        model.nand_weight[2] = torch.tensor([1, 0])
        model.nand_weight[3] = torch.tensor([1, 1])

    out = model(query)
    print(out)

    '''
    #       -a^-b   -a^b   a^-b   a^b
    tensor([[0.000, 0.000, 0.000, 1.000],  # [a, b]
            [0.000, 0.000, 0.938, 0.062],  # [a, c]
            [0.000, 0.919, 0.000, 0.081]]) # [c, b]
    '''
    BRK
# @@@@@@@@@@


##################################################
#

def train_and_report(n_choices, redundancy, vec_size, method):
    print('------------------------------')
    print(f'method = {method},  n_choices={n_choices}, redundancy={redundancy}',)

    # output choices
    model = Model(vec_size, 3, n_choices, redundancy, method)
    model.cuda()


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Params: {format_number(n_params)}')

    #####
    # Train
    opt_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(opt_params, lr=LR, weight_decay=0.0)
    train_losses = []
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for _, batch in enumerate(train_dl):
            src = batch['inp']
            trg = batch['out']

            # TEST NOISE
            NOISE_LVL = 1e-3
            src[0] = src[0] + torch.randn_like(src[0]) * NOISE_LVL
            src[1] = src[1] + torch.randn_like(src[1]) * NOISE_LVL
            src[2] = src[2] + torch.randn_like(src[1]) * NOISE_LVL
            trg = trg + torch.randn_like(trg) * NOISE_LVL

            optimizer.zero_grad()
            output = model(src)

            # LOSS
            loss = (1 - F.cosine_similarity(output, trg)).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_dl)
        # print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')
    end = time.time()
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}  |  time={end - start:>.2f}')

    model.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for fn_name, fn in binary_functions.items():
            for x in [True, False]:
                for y in [True, False]:
                    n+=1
                    trg = fn(x, y)
                    out = model([
                        project(fn_name).unsqueeze(0),
                        project(x).unsqueeze(0),
                        project(y).unsqueeze(0),
                    ])

                    uout = unproject(out.squeeze(0))
                    if uout == trg:
                        correct += 1

    print(f'acc: {correct / n:>.3f}')

    return model


####################
# Dataset

binary_functions = {
    "xor"         : lambda x, y: True if (x, y) in {(False, True), (True, False)} else False,
    "and"         : lambda x, y: True if (x, y) in {(True, True)} else False,
    "or"          : lambda x, y: True if (x, y) in {(False, True), (True, False), (True, True)} else False,
    "nand"        : lambda x, y: False if (x, y) in {(True, True)} else True,
    "nor"         : lambda x, y: False if (x, y) in {(False, True), (True, False), (True, True)} else True,
    "not_left"    : lambda x, y: True if x == False else False,
    "not_right"   : lambda x, y: True if y == False else False,
    "const_False" : lambda x, y: False,
    "const_True"  : lambda x, y: True,
    "implies"     : lambda x, y: False if (x, y) == (True, False) else True,
    "rev_implies" : lambda x, y: False if (x, y) == (False, True) else True,
    "equiv"       : lambda x, y: True if x == y else False,
    "const_left"  : lambda x, y: x,
    "const_right" : lambda x, y: y,
}

####################


# PICK WHICH MODEL TO USE
# Model = NNModel
Model = SymModel

DEVICE = 'cuda'
VEC_SIZE = 64
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-2
all_symbols = [True, False] + list(binary_functions.keys())
sym_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject

dataset = []

for fn_name, fn in binary_functions.items():
    for x in [True, False]:
        for y in [True, False]:
            dataset.append(dict(
                uinp = (fn_name, x, y),
                inp = (project(fn_name), project(x), project(y)),
                uout = fn(x, y),
                out = project(fn(x, y)),
            ))

train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


#####
# GO

R = 1

experiments = [

    {'method': 'softmax', 'redundancy': R, 'n_choices': 4, 'vec_size':VEC_SIZE},
    {'method': 'softmax', 'redundancy': R, 'n_choices': 8, 'vec_size':VEC_SIZE},
    {'method': 'softmax', 'redundancy': R, 'n_choices': 16, 'vec_size':VEC_SIZE},

]

for e in experiments:
    model = train_and_report(**e)



##########
# Hand Testing

for i in range(model.n_choices):
    print(f'{i} {unproject(model.vecs[:, i], return_sim=True)}')


def f(nm, x, y):
    trg = binary_functions[nm](x, y)
    out = model([
        project(fn_name).unsqueeze(0),
        project(x).unsqueeze(0),
        project(y).unsqueeze(0),
    ])

    uout = unproject(out.squeeze(0))
    print(f'{nm.upper()} {x} {y} == {uout}.  trg={trg}')

'''

for fn_name in binary_functions.keys():
    print()
    for x in [True, False]:
        for y in [True, False]:
            f(fn_name, x, y)

'''

print('done')


##################################################
# Sandbox

N = 256
sym_map = Sym.SymbolMapper(N, Sym.chars + Sym.nums, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject

A = project('A')
B = project('B')
Q = project('Q')
R = project('R')
A_CAT_B = torch.concat([A, B])
A_CAT_Q = torch.concat([A, Q])
Q_CAT_B = torch.concat([Q, B])
R_CAT_Q = torch.concat([R, Q])

def sim_A_AND_NOT_B(query):
    Q1_, Q2_ = torch.split(F.normalize(query, dim=0), N)
    A_,  B_  = torch.split(F.normalize(A_CAT_B, dim=0), N)
    return ((A_ @ Q1_) * 2) *  ((0.5 - B_ @ Q2_) * 2)

print('Should have low sim: ', sim_A_AND_NOT_B(A_CAT_B))
print('Should have high sim:', sim_A_AND_NOT_B(A_CAT_Q))

print('Should have low sim: ',  sim_A_AND_NOT_B(A_CAT_B + torch.randn(N*2, device=DEVICE) * 1e-1))
print('Should have high sim:', sim_A_AND_NOT_B(A_CAT_Q + torch.randn(N*2, device=DEVICE) * 1e-1))

print('\n'*3)

def sim_A_OR_NOT_B(query):
    # convert A OR NOT B
    #   to
    # NOT (NOT A AND B)
    Q1_, Q2_ = torch.split(F.normalize(query, dim=0), N)
    A_,  B_  = torch.split(F.normalize(A_CAT_B, dim=0), N)
    AQ = (0.5 - A_ @ Q1_) * 2 # NOT A
    BQ = (B_ @ Q2_) * 2 # B
    return 1 - AQ * BQ # NOT (NOT A AND B) ~aka~ A OR NOT B

print('Should have high sim:', sim_A_OR_NOT_B(R_CAT_Q)) # high bc "not B"
print('Should have high sim:', sim_A_OR_NOT_B(A_CAT_Q)) # high bc both
print('Should have high sim:', sim_A_OR_NOT_B(A_CAT_B)) # high bc A, even though B should be not'd
print('Should have low sim :', sim_A_OR_NOT_B(Q_CAT_B))
