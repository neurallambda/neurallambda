'''.

Create a way to learn substitution based on the NAND work.

See `fn` for the test dataset to understand what it's needing to learn, it's a
tough problem.

----------
RESULTS:

A.M.A.Z.I.N.G.  I went with an approach like NAND, but instead of learning
weights to cos_sim against, I created FwdNAND where you can cos_sim against
other latent vars/inputs. These cos_sims pass into FwdNAND where it does have
some nand_weights to learn how the cos_sims should be NAND'ed together.

It can learn to incredibly high accuracy, far out of domain. It can operate on
symbols it's never seen, flawlessly, even after learning on an incredibly
impoverished training dataset.

----------
FUTURE WORK:

It may be the case that these positive results stem from merely introducing
latent*latent or input*input dynamics, instead of merely input*weight dynamics.

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
N_VECS = 3


##################################################
#

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


class SymModel(nn.Module):
    def __init__(self, vec_size, n_vecs, n_choices, redundancy):
        super(SymModel, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy

        self.choice = NAND(vec_size, n_vecs, n_choices, redundancy, method='softmax')

        self.n_fwd_cos_sim = 3
        self.n_fwd_choices = 4
        self.fwd_choice = FwdNAND(self.n_fwd_cos_sim, n_choices=self.n_fwd_choices)

        # use choice to select an output vec
        self.vecs = nn.Parameter(torch.randn(n_choices - 2 # + n_fwd_cos_sim
                                             , vec_size))

        # Normalize the main weights
        with torch.no_grad():
            self.vecs[:] = F.normalize(self.vecs, dim=0)


    def forward(self, inp: torch.Tensor):
        query, x_name, x_val, y_name, y_val = inp
        batch_size = query.size(0)

        # FwdNAND
        fwd_choices = self.fwd_choice(torch.stack([
            torch.cosine_similarity(query, x_name, dim=1),
            torch.cosine_similarity(query, y_name, dim=1),
            torch.cosine_similarity(x_name, y_name, dim=1),
        ], dim=1))

        # NAND
        choices = self.choice(torch.hstack([
            query,
            x_name,
            y_name,
        ]))

        choices = torch.concat([fwd_choices, choices], dim=1)

        # TODO:Experiment
        eps = 1e-6
        choices = (choices).clip(eps, 1-eps)  # note: clips neg similarities
        choices = torch.log((choices) / (1 - choices))  # maps [0,1] -> [-inf, inf]
        choices = torch.sum(choices.softmax(dim=1).view(batch_size,
                                                        self.n_choices + self.n_fwd_choices,
                                                        self.redundancy), dim=2)

        vecs = torch.concat([x_val.unsqueeze(1),
                             y_val.unsqueeze(1),
                             self.vecs.expand(batch_size, -1, -1),
                             x_val.unsqueeze(1), # fwd_choice
                             y_val.unsqueeze(1), # fwd_choice
                             x_val.unsqueeze(1), # fwd_choice redundancy
                             y_val.unsqueeze(1), # fwd_choice redundancy
                             ], dim=1)
        return torch.einsum('bcv, bc -> bv', vecs, choices)

class NNModel(nn.Module):
    ''' Control model, using standard FFNN '''
    def __init__(self, vec_size, n_vecs, n_choices, *args, **kwargs):
        super(NNModel, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices

        H = 32

        self.choice = nn.Sequential(
            nn.Linear(5 * vec_size, H),
            nn.Tanh(),
            nn.Linear(H, H),
            nn.Tanh(),
            # nn.Linear(H, H),
            # nn.Tanh(),
            nn.Linear(H, n_choices),
            nn.Sigmoid()
        )

        # # use choice to select an output vec
        self.vecs = nn.Parameter(torch.randn(n_choices - 2, vec_size))

    def forward(self, inp: torch.Tensor):
        query, x_name, x_val, y_name, y_val = inp
        batch_size = query.size(0)
        choices = self.choice(torch.hstack([query, x_name, x_val, y_name, y_val]))
        vecs = torch.concat([x_val.unsqueeze(1),
                             y_val.unsqueeze(1),
                             self.vecs.expand(batch_size, -1, -1)
                             ], dim=1)
        return torch.einsum('bcv, bc -> bv', vecs, choices)


##################################################
#

def train_and_report(n_choices, redundancy, vec_size, model, *args, **kwargs):
    print('------------------------------')
    print(f'model = {str(model)},  n_choices={n_choices}, redundancy={redundancy}',)

    # output choices
    model = model(vec_size, N_VECS, n_choices, redundancy)
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
            NOISE_LVL = 1e-1
            src[0] = src[0] + torch.randn_like(src[0]) * NOISE_LVL
            src[1] = src[1] + torch.randn_like(src[1]) * NOISE_LVL
            src[2] = src[2] + torch.randn_like(src[1]) * NOISE_LVL
            trg = trg + torch.randn_like(trg) * NOISE_LVL

            optimizer.zero_grad()
            output = model(src)

            # LOSS
            loss = (1 - F.cosine_similarity(output, trg)).mean()
            with torch.no_grad():
                train_losses.append(loss.item())

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
    test_losses = []
    with torch.no_grad():
        for _, batch in enumerate(test_dl):
            src = batch['inp']
            trg = batch['out']
            output = model(src)
            # LOSS
            loss = (1 - F.cosine_similarity(output, trg)).mean()
            test_losses.append(loss.item())

            for t, o in zip(batch['uout'], output):
                n += 1
                if t == unproject(o):
                    correct += 1

    print(f'acc: {correct / n:>.3f}')

    return model, train_losses, test_losses


####################
# Dataset

DEVICE = 'cuda'
VEC_SIZE = 64
BATCH_SIZE = 125
NUM_EPOCHS = 200
LR = 1e-2
all_symbols = Sym.nums + Sym.chars + ['Default', 'All', 'Collide']
sym_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject

def fn(query, x_name, x_val, y_name, y_val):
    if query == x_name == y_name:
        return 'All'
    if x_name == y_name:
        return 'Collide'
    if query == x_name:
        return x_val
    if query == y_name:
        return y_val
    return 'Default'

# if p > 0 return x else y
def build_dataset(var_names, vals):
    dataset = []
    for query in var_names:
        for x_name in var_names:
            for y_name in var_names:
                for x_val in vals:
                    for y_val in vals:
                        uinp = (query, x_name, x_val, y_name, y_val)
                        inp = tuple(project(q) for q in uinp)
                        uout = fn(*uinp)
                        out = project(uout)
                        dataset.append(dict( uinp = uinp, inp = inp, uout = uout, out = out))
    return dataset

train_dataset = build_dataset(range(0, 4), ['a', 'b'])
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = build_dataset(range(100, 105), ['d', 'e', 'f'])
test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


#####
# GO

R = 1

experiments = [

    {'model': SymModel, 'redundancy': R, 'n_choices': 5, 'vec_size':VEC_SIZE, 'name': 'SymModel'},
    {'model': NNModel,  'redundancy': R, 'n_choices': 5, 'vec_size':VEC_SIZE, 'name': 'FFNN'},

]


all_train_losses = []
all_test_losses = []
for e in experiments:
    model, train_losses, test_losses = train_and_report(**e)
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)


##########
# Viz


import numpy as np
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
for index, losses in enumerate(all_train_losses):
    plt.plot(losses, label=f"{experiments[index]['name']} Train")
for index, losses in enumerate(all_test_losses):
    # Calculate the average test loss for each experiment (assuming losses are a list of end-of-training values)
    avg_loss = np.mean(losses)
    # Plot a horizontal line for the average test loss
    plt.hlines(avg_loss, 0, len(all_train_losses[index])-1, colors=['r', 'g', 'b', 'c', 'm', 'y', 'k'][index % 7], linestyles='dashed', label=f"{experiments[index]['name']} Test Avg")
plt.title("Training and Test Losses Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ##########
# # Hand Testing

# for i in range(model.n_choices):
#     print(f'{i} {unproject(model.vecs[:, i], return_sim=True)}')


# def f(nm, x, y):
#     trg = binary_functions[nm](x, y)
#     out = model([
#         project(fn_name).unsqueeze(0),
#         project(x).unsqueeze(0),
#         project(y).unsqueeze(0),
#     ])

#     uout = unproject(out.squeeze(0))
#     print(f'{nm.upper()} {x} {y} == {uout}.  trg={trg}')

# '''

# for fn_name in binary_functions.keys():
#     print()
#     for x in [True, False]:
#         for y in [True, False]:
#             f(fn_name, x, y)

# '''

# print('done')
