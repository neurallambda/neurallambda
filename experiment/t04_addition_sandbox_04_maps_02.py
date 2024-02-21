'''

I need to solve ideal ways of mapping:

Sym -> Float
Sym -> Sym

Sym * Sym -> Float
Sym * Sym -> Sym

Sym * Sym * Sym -> Float
Sym * Sym * Sym -> Sym

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

import torch.fft

torch.manual_seed(152)

class Choice(nn.Module):
    ''' N-vectors -> [0, 1] '''
    def __init__(self, vec_size, n_vecs, n_choices, redundancy, has_guard=False, method='softmax'):
        super(Choice, self).__init__()
        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.has_guard = has_guard
        self.method = method
        assert method in {'max', 'softmax', 'outer_projection', 'sum'}

        # Stack inputs
        if n_vecs == 1:
            Vecs = Id()
        elif n_vecs == 2:
            Vecs = Fn(lambda a, b: torch.hstack([a, b]), nargs=2)
        elif n_vecs == 3:
            Vecs = Fn(lambda a, b, c: torch.hstack([a, b, c]), nargs=3)
        elif n_vecs == 4:
            Vecs = Fn(lambda a, b, c, d: torch.hstack([a, b, c, d]), nargs=4)
        elif n_vecs == 5:
            Vecs = Fn(lambda a, b, c, d, e: torch.hstack([a, b, c, d, e]), nargs=5)
        elif n_vecs == 6:
            Vecs = Fn(lambda a, b, c, d, e, f: torch.hstack([a, b, c, d, e, f]), nargs=6)

        self.ff = nn.Sequential(
            Vecs,
            NormalizedLinear(vec_size * n_vecs, redundancy * n_choices, bias=False),
            nn.GELU(),

            # nn.Linear(vec_size * n_vecs, redundancy * n_choices, bias=False),
            # nn.Sigmoid()
        )

        if has_guard:
            self.guard = nn.Sequential(
                Vecs,
                NormalizedLinear(vec_size * n_vecs, redundancy * n_choices, bias=False),
                nn.GELU(),

                # nn.Linear(vec_size * n_vecs, redundancy * n_choices, bias=False),
                # nn.Sigmoid()
            )

        if method == 'outer_projection':
            if n_choices == 2:
                self.outer = lambda xs: torch.einsum('za, zb -> zab', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices == 3:
                self.outer = lambda xs: torch.einsum('za, zb, zc -> zabc', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices == 4:
                self.outer = lambda xs: torch.einsum('za, zb, zc, zd -> zabcd', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices == 5:
                self.outer = lambda xs: torch.einsum('za, zb, zc, zd, ze -> zabcde', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices > 5:
                raise ValueError(f'The outer_projection method scales as O(redundancy ** n_choices). You probably dont want n_choices>5, but you picked n_choices={n_choices}')

            self.proj = nn.Sequential(
                NormalizedLinear(redundancy ** n_choices, n_choices, bias=False),
                nn.GELU(),

                # nn.Linear(redundancy ** n_choices, n_choices, bias=True),

                # nn.Sigmoid(),
                # nn.Softmax(dim=1)
            )

    def forward(self, inp):
        batch_size = inp[0].size(0)
        if self.has_guard:
            f = self.ff(inp)
            g = self.guard(inp)
            print(
                f'{f.min().item():>.1f}',
                f'{f.max().item():>.1f}',
                f'{f.mean().item():>.1f}',
                '|',
                f'{g.min().item():>.1f}',
                f'{g.max().item():>.1f}',
                f'{g.mean().item():>.1f}')
            outs = f * g
        else:
            outs = self.ff(inp)

        if self.method == 'outer_projection':
            chunks = torch.chunk(outs, self.n_choices, dim=1)
            return self.proj(self.outer(chunks))
        elif self.method == 'max':
            # avg over each chunk via reshaping first
            return torch.max(outs.view(batch_size, self.n_choices, self.redundancy), dim=2).values
        elif self.method == 'softmax':
            # softmax over the whole redundant vec, then sum each chunk
            return torch.sum(10 * outs.softmax(dim=1).view(batch_size, self.n_choices, self.redundancy), dim=2)
        elif self.method == 'sum':
            # avg over each chunk via reshaping first
            return torch.sum(outs.view(batch_size, self.n_choices, self.redundancy), dim=2)
        elif self.method == 'mean':
            # avg over each chunk via reshaping first
            return torch.mean(outs.view(batch_size, self.n_choices, self.redundancy), dim=2)

DEVICE = 'cuda'
VEC_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 40
all_symbols = Sym.nums
int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = int_map.project
unproject = int_map.unproject

N_CHOICES = 5
# METHOD = 'softmax'
# METHOD = 'max'
# METHOD = 'sum'
METHOD = 'outer_projection'
HAS_GUARD = False

odd_dataset = []

odd_avg = []
even_avg = []
for i in range(-10, 11):
    for j in range(-10, 11):

        # Remove training data, it still does great!
        if i < 5 and j > 5:
            continue
        if (i, j) in {(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)}:
            continue

        # # Change bias of dataset
        # if (i+j)%2==0 and i < 0:
        #     continue

        x0 = project(i)
        x1 = project(j)

        x = (x0, x1)
        y = project((i + j) % N_CHOICES).to(DEVICE)
        odd_dataset.append((x, y))

        hv = x0 * x1
        if (i+j) % 2 == 1:
            odd_avg.append(hv)
        else:
            even_avg.append(hv)

train_odd_dl = DataLoader(odd_dataset, batch_size=BATCH_SIZE, shuffle=True)

symbols = torch.stack([project(x) for x in range(N_CHOICES)])

#####

out_symbols = nn.Parameter(torch.randn((N_CHOICES, VEC_SIZE)))
odd_model = nn.Sequential(
    Choice(VEC_SIZE, 2, n_choices=N_CHOICES, redundancy=8, has_guard=HAS_GUARD, method=METHOD),
    Fn(lambda x: einsum('cv, bc -> bv', out_symbols, x), parameters=[out_symbols]),
)

odd_model.cuda()

n_params = sum(p.numel() for p in odd_model.parameters() if p.requires_grad)
print(f'Total Params: {n_params}')

#####
# Test Hack
odd_avg = torch.stack(odd_avg).mean(dim=0)
even_avg = torch.stack(even_avg).mean(dim=0)
hack_vec = (odd_avg - even_avg) / 2

# # # # hack_vec -= hack_vec.mean()
# # # # hack_vec /= hack_vec.max()
# # # hack_vec /= len(odd_dataset)
# # # # hack_vec = F.normalize(hack_vec, dim=0)
# with torch.no_grad():
#     odd_model.ff[1].weight[:] = hack_vec

#####
# Train
opt_params = list(filter(lambda p: p.requires_grad, odd_model.parameters()))
optimizer = optim.Adam(opt_params, lr=1e-2)
train_losses = []
for epoch in range(NUM_EPOCHS):
    odd_model.train()
    epoch_loss = 0
    for _, (src, trg) in enumerate(train_odd_dl):
        optimizer.zero_grad()
        output = odd_model(src)
        loss = (1 - F.cosine_similarity(output, trg)).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_odd_dl)
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')

correct = 0
n = 0
for i in range(-10, 10):
    for j in range(-10, 10):
        pi = project(i).unsqueeze(0)
        pj = project(j).unsqueeze(0)
        out = odd_model([pi, pj])
        expected = (i+j) % N_CHOICES
        actual = torch.cosine_similarity(out, symbols, dim=1).argmax().item()
        if actual == expected:
            correct += 1


        n += 1
        # print(f'{i} + {j} = {out.item() :>.3f}')
print(f'acc: {correct / n:>.3f}')
print('\n' * 3)


# # Viz
# model_w = odd_model.ff[1].weight.squeeze(0)
# print(f'sim to odd avg : {cosine_similarity(odd_avg, model_w, dim=0).item() :> .3f}')
# print(f'sim to even avg: {cosine_similarity(even_avg, model_w, dim=0).item() :> .3f}')

# sim_odd = []
# sim_even = []
# for (x0, x1), y in odd_dataset:
#     sim = cosine_similarity(x0 * x1, model_w, dim=0).item()
#     if y > 0.5:
#         sim_odd.append(sim)
#     else:
#         sim_even.append(sim)

# #plt.figure(figsize=(1, 2))
# plt.subplot(1, 2, 1)
# plt.plot(sim_odd)
# plt.subplot(1, 2, 2)
# plt.plot(sim_even)

# plt.tight_layout()
# plt.show()
