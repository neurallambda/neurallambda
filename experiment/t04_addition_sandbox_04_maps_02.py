'''

I need to solve ideal ways of mapping:

Sym -> Float
Sym -> Sym

Sym * Sym -> Float
Sym * Sym -> Sym

Sym * Sym * Sym -> Float
Sym * Sym * Sym -> Sym


----------
OPTIONS:

* Outer pdt via VSA techniques (add all combinations of pdts into 1 vec)

* ID input symbols, IE map vector symbol to onehot vec of (of n_symbols dim)

* ID input symbols -> outer product -> map -> output symbols

* Don't ID input symbols -> map -> output symbols

--------------------------------------------------
NOTES:

* setting output symbols to known vector values helps initial training

'''


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

torch.manual_seed(152)

print('\n'*2)

DEBUG = False

import warnings
warnings.filterwarnings("always")
import traceback
import warnings
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = warn_with_traceback


##################################################
# Training

def format_number(num):
    """
    Formats a number with suffixes 'k', 'M', or 'B' for thousands, millions, and billions respectively.

    Parameters:
    - num (int): The number to format.

    Returns:
    - str: The formatted number as a string.
    """
    if abs(num) >= 1_000_000_000:  # Billion
        formatted_num = f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:  # Million
        formatted_num = f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:  # Thousand
        formatted_num = f"{num / 1_000:.1f}k"
    else:
        formatted_num = str(num)

    return formatted_num

def train_and_report(n_choices, redundancy, vec_size, has_guard, method):
    print('------------------------------')
    print(f'method = {method},  has_guard = {has_guard},  redundancy={redundancy}',)
    symbols = torch.stack([project(x) for x in range(N_CHOICES)])

    model = ChooseVec(vec_size, n_choices, redundancy, has_guard, method, symbols)

    # model = nn.Sequential(
    #     Choice(vec_size,
    #            2,
    #            n_choices=n_choices,
    #            redundancy=redundancy,
    #            has_guard=has_guard,
    #            method=method),
    #     Fn(lambda x: einsum('cv, bc -> bv', out_symbols, x), parameters=[out_symbols]),
    # )

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
        for _, (src, trg) in enumerate(train_odd_dl):

            # TEST NOISE
            NOISE_LVL = 1e-2
            src[0] = src[0] + torch.randn_like(src[0]) * NOISE_LVL
            src[1] = src[1] + torch.randn_like(src[1]) * NOISE_LVL
            trg = trg + torch.randn_like(trg) * NOISE_LVL

            optimizer.zero_grad()
            output = model(src)

            # LOSS
            loss = (1 - F.cosine_similarity(output, trg)).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_odd_dl)
        # print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')
    end = time.time()
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}  |  time={end - start:>.2f}')

    model.eval()
    correct = 0
    n = 0
    for i in range(-10, 10):
        for j in range(-10, 10):
            prj_i = project(i).unsqueeze(0)
            prj_j = project(j).unsqueeze(0)
            with torch.no_grad():
                out = model([prj_i, prj_j])
            expected = (i+j) % n_choices
            actual = torch.cosine_similarity(out, symbols, dim=1).argmax().item()
            if actual == expected:
                correct += 1
            n += 1
            # print(f'{i} + {j} = {out.item() :>.3f}')
    print(f'acc: {correct / n:>.3f}')

    return model


##################################################
# ChooseVec

class NuLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 init_extra_weight=None,
                 fwd_extra_dim=0,
                 bias=False,
                 normalize_input=True,
                 normalize_weight=True):
        super(NuLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_extra_weight = init_extra_weight
        self.fwd_extra_dim = fwd_extra_dim
        self.normalize_input = normalize_input
        self.normalize_weight = normalize_weight

        # Bias
        if bias:
            # Bias shape: [out_features]
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Weight
        if in_features > 0 and out_features > 0:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()
        else:
            self.weight = None

        if init_extra_weight is not None:
            assert init_extra_weight.dim() == 2 and init_extra_weight.size(1) == in_features, f"init_extra_weight must have shape [init_extra_dim, in_features={in_features}], but has shape={init_extra_weight.shape}"
            # Shape: [init_extra_dim, in_features]
            self.init_extra_weight = nn.Parameter(init_extra_weight)
            # Adjust total output features to include init_extra_weight
            self.total_out_features = out_features + init_extra_weight.size(0)
        else:
            self.init_extra_weight = None
            self.total_out_features = out_features

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.init_extra_weight is not None:
            nn.init.kaiming_uniform_(self.init_extra_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, extra_weight=None):
        # Input shape: [batch_size, in_features]
        if self.normalize_input:
            input = F.normalize(input, p=2, dim=1)

        weight = self.weight
        if self.init_extra_weight is not None:
            # Concatenated weight shape: [out_features + init_extra_dim, in_features]
            if weight is not None:
                weight = torch.cat([self.weight, self.init_extra_weight], dim=0)
            else:
                weight = self.init_extra_weight

        if extra_weight is not None:
            assert extra_weight.shape[1] == self.fwd_extra_dim and extra_weight.shape[2] == self.in_features, f"extra_weight must have shape [batch, fwd_extra_dim, in_features], but has shape={extra_weight.shape}"
            # Repeat and concatenate for shape: [batch, out_features + init_extra_dim + fwd_extra_dim, in_features]
            weight = torch.cat([weight.unsqueeze(0).repeat(extra_weight.size(0), 1, 1), extra_weight], dim=1)

        if self.normalize_weight:
            # Normalize across the appropriate dimension
            weight = F.normalize(weight, p=2, dim=-1)

        if extra_weight is not None:
            # Corrected output calculation for batched inputs
            output = torch.bmm(weight, input.unsqueeze(2)).squeeze(2)
        else:
            output = input.matmul(weight.t())

        if self.bias is not None:
            # Ensure bias is correctly expanded and added to output
            # Adjust bias shape based on actual output features
            bias = self.bias if self.init_extra_weight is None else torch.cat([self.bias, torch.zeros(self.init_extra_weight.size(0), device=self.bias.device)], 0)
            if extra_weight is not None:
                bias = torch.cat([bias, torch.zeros(self.fwd_extra_dim, device=bias.device)], 0)  # Extend bias for fwd_extra_dim
            output += bias.unsqueeze(0)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, normalize_input={}, normalize_weight={}, init_extra_dim={}, fwd_extra_dim={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.normalize_input, self.normalize_weight, self.init_extra_weight.size(0) if self.init_extra_weight is not None else 0, self.fwd_extra_dim
        )


# @@@@@@@@@@

# Initialize parameters
in_features = 10
out_features = 5
batch_size = 3
init_extra_dim = 2
fwd_extra_dim = 4

# Create dummy inputs
input = torch.randn(batch_size, in_features)
init_extra_weight = torch.randn(init_extra_dim, in_features)
fwd_extra_weight = torch.randn(batch_size, fwd_extra_dim, in_features)

# Create NuLinear instances and perform assertions
# Test without any extra weight
model = NuLinear(in_features, out_features, bias=True)
assert model(input).shape == (batch_size, out_features), "Output shape mismatch without extra weight"

# Test with initial extra weight
model_with_init_extra = NuLinear(in_features, out_features, init_extra_weight=init_extra_weight, bias=True)
assert model_with_init_extra(input).shape == (batch_size, out_features + init_extra_dim), "Output shape mismatch with init extra weight"

# Test with forward extra weight
model_with_fwd_extra = NuLinear(in_features, out_features, fwd_extra_dim=fwd_extra_dim, bias=True)
output_with_fwd_extra = model_with_fwd_extra(input, extra_weight=fwd_extra_weight)
assert output_with_fwd_extra.shape == (batch_size, out_features + fwd_extra_dim), "Output shape mismatch with forward extra weight"

# Test with both init and forward extra weight
model_with_both_extra = NuLinear(in_features, out_features, init_extra_weight=init_extra_weight, fwd_extra_dim=fwd_extra_dim, bias=True)
output_with_both_extra = model_with_both_extra(input, extra_weight=fwd_extra_weight)
assert output_with_both_extra.shape == (batch_size, out_features + init_extra_dim + fwd_extra_dim), "Output shape mismatch with both types of extra weight"

print("All assertions passed!")

# @@@@@@@@@@


##################################################
#

class Choice(nn.Module):
    ''' N-vectors -> [0, 1] '''
    def __init__(self,
                 vec_size,
                 n_vecs,
                 n_choices,
                 redundancy,
                 has_guard=False,
                 method='softmax',
                 init_extra_weight=None,
                 fwd_extra_weight_dim=0,
                 ):
        super(Choice, self).__init__()
        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.has_guard = has_guard
        self.method = method
        self.fwd_extra_weight_dim=fwd_extra_weight_dim
        assert method in {'max', 'softmax', 'gumbel_softmax', 'sum', 'mean'}

        self.ff = NuLinear(
            vec_size * n_vecs,
            0,
            # redundancy * n_choices,
            bias=False,
            fwd_extra_dim=vec_size * n_vecs,
            normalize_input=True,
            normalize_weight=True,
            init_extra_weight = init_extra_weight
        )

        if has_guard:
            self.guard = NuLinear(
                    vec_size * n_vecs,
                    redundancy * n_choices,
                    bias=False,
                    normalize_input=True,
                    normalize_weight=True,
                )
            self.guard_scale = nn.Parameter(torch.tensor([redundancy * 1.0]))

        self.scale = nn.Parameter(torch.tensor([redundancy * 0.1]))


        self.dropout = nn.Dropout(0.0)

    def forward(self, inp, extra_weights=None, eps=1e-6):
        # inp arity=2
        batch_size = inp[0].size(0)
        sinp = torch.hstack(inp)

        # inp arity=1
        # batch_size = inp.size(0)
        # sinp = inp

        outs = self.ff(sinp, extra_weights)
        outs = self.dropout(outs)

        hg = self.has_guard
        if hg:
            g = self.guard(sinp)
            g = self.dropout(g)


        sz = self.n_choices + self.fwd_extra_weight_dim // self.redundancy
        if self.method == 'max':
            outs = torch.max(outs.view(batch_size, sz, self.redundancy), dim=2).values
            if hg: g = torch.max(g.view(batch_size, sz, self.redundancy), dim=2).values


        elif self.method in {'softmax', 'gumbel_softmax'}:
            # softmax over the whole redundant vec, then sum each redundant chunk
            # clip because of singularities in tan and log(p/(1-p))
            outs = (outs).clip(eps, 1-eps)  # note: clips neg similarities
            outs = torch.log((outs) / (1 - outs))  # maps [0,1] -> [-inf, inf]
            # outs = torch.tan((outs - 0.5) * pi)  # maps [0,1] -> [-inf, inf]

            # outs = (outs).clip(-1+eps, 1-eps)
            # outs = torch.tan(outs * pi / 2)  # maps [-1,1] -> [-inf, inf]

            if self.method == 'softmax':
                outs = torch.sum(outs.softmax(dim=1).view(batch_size, sz, self.redundancy), dim=2)
            elif self.method == 'gumbel_softmax':
                outs = torch.sum(F.gumbel_softmax(outs, dim=1).view(batch_size, sz, self.redundancy), dim=2)

            if hg:
                g = (g).clip(eps, 1-eps)
                g = torch.log((g) / (1 - g))
                g = torch.sum(g.softmax(dim=1).view(batch_size, sz, self.redundancy), dim=2)


        # elif self.method == 'softmax':
        #     # softmax over the whole redundant vec, then sum each redundant chunk
        #     if DEBUG:
        #         breakpoint()
        #     eps = 1e-6

        #     outs = (outs).clip(eps, 1-eps)  # note: clips neg similarities
        #     outs = torch.log((outs) / (1 - outs))  # maps [0,1] -> [-inf, inf]
        #     # outs = torch.tan((outs - 0.5) * pi)  # maps [0,1] -> [-inf, inf]

        #     # outs = (outs).clip(-1+eps, 1-eps)
        #     # outs = torch.tan(outs * pi / 2)  # maps [-1,1] -> [-inf, inf]

        #     outs = torch.sum(outs.softmax(dim=1).view(batch_size, sz, self.redundancy), dim=2)
        #     if hg:
        #         g = (g).clip(eps, 1-eps)
        #         g = torch.log((g) / (1 - g))
        #         g = torch.sum(g.softmax(dim=1).view(batch_size, sz, self.redundancy), dim=2)

        elif self.method == 'sum':
            outs = torch.sum(outs.view(batch_size, sz, self.redundancy), dim=2)
            if hg: g = torch.sum(g.view(batch_size, sz, self.redundancy), dim=2)

        elif self.method == 'mean':
            outs = torch.mean(outs.view(batch_size, sz, self.redundancy), dim=2)
            if hg: g = torch.mean(g.view(batch_size, sz, self.redundancy), dim=2)

        if hg:
            outs = outs * g

        if self.method in {'sum', 'mean'}:
            # outs = outs * self.scale
            outs = torch.sigmoid(outs * self.scale)

        return outs


class ChooseVec(nn.Module):
    ''' N-vectors -> Vector '''
    def __init__(self, vec_size, n_choices, redundancy, has_guard, method, symbols):
        super(ChooseVec, self).__init__()

        # self.symbols = symbols

        self.program = nn.Parameter(torch.randn(redundancy * n_choices, vec_size * 2))

        self.choice = Choice(
            vec_size,
            2,
            n_choices=n_choices,
            redundancy=redundancy,
            has_guard=has_guard,
            method=method,
            init_extra_weight=self.program,
        )

        self.out_symbols = nn.Parameter(torch.randn((n_choices, vec_size)))

    def forward(self, input):

        # inp arity = 2
        x1 = self.choice(input)

        # # inp arity = 1
        # x1 = self.choice(input[0] * input[1])

        x2 = einsum('cv, bc -> bv', self.out_symbols, x1)
        return x2


##################################################
# DATA

DEVICE = 'cuda'
VEC_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-1
all_symbols = Sym.nums
int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = int_map.project
unproject = int_map.unproject

N_CHOICES = 5

odd_dataset = []

nums = list(range(-10, 11))

for i in nums:
    for j in nums:

        # Remove training data, it still does great!
        if i < 5 and j < 5:
            continue
        if (i, j) in {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)}:
            continue
        # Change bias of dataset
        if (i+j)%2==0 and i < 0:
            continue

        x0 = project(i)
        x1 = project(j)

        x = (x0, x1)
        y = project((i + j) % N_CHOICES).to(DEVICE)
        odd_dataset.append((x, y))

print(f'% possible data trained: {len(odd_dataset) / len(nums) ** 2 :> .3f}')

train_odd_dl = DataLoader(odd_dataset, batch_size=BATCH_SIZE, shuffle=True)


#####
# GO

R = 4

experiments = [

    # {'has_guard': True,  'method': 'max', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False, 'method': 'max', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},

    {'has_guard': True,  'method': 'softmax', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    {'has_guard': False, 'method': 'softmax', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},


    # {'has_guard': True,  'method': 'gumbel_softmax', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False, 'method': 'gumbel_softmax', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},


    # {'has_guard': True,  'method': 'sum', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False, 'method': 'sum', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},

    # {'has_guard': True,  'method': 'mean', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False, 'method': 'mean', 'redundancy': R, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},

    # {'has_guard': False,  'method': 'softmax', 'redundancy': 1, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False,  'method': 'softmax', 'redundancy': 2, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False,  'method': 'softmax', 'redundancy': 4, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False,  'method': 'softmax', 'redundancy': 8, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False,  'method': 'softmax', 'redundancy': 16, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False,  'method': 'softmax', 'redundancy': 32, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},
    # {'has_guard': False,  'method': 'softmax', 'redundancy': 64, 'n_choices': N_CHOICES, 'vec_size':VEC_SIZE},

    ]

for e in experiments:
    model = train_and_report(**e)
    print(f'sm_model = {model.choice.scale.item():>.3f}')

DEBUG=False
a = project(1).unsqueeze(0)
b, sim = unproject(model([a, a]).squeeze(0), return_sim=True)
print(f'sim={sim}, b={b}')
