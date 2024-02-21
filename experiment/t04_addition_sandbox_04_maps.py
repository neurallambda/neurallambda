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

def convolve(x1, x2):
    """Performs circular convolution on two vectors using FFT and IFFT."""
    fft_x1 = torch.fft.fft(x1)
    fft_x2 = torch.fft.fft(x2)
    fft_product = fft_x1 * fft_x2
    output = torch.fft.ifft(fft_product)
    return output.real

def deconvolve(bound_vec, one_original_vec):
    """Performs the inverse operation of circular convolution to retrieve the original vector."""
    fft_bound_vec = torch.fft.fft(bound_vec)
    fft_one_original_vec = torch.fft.fft(one_original_vec)
    fft_other_original_vec = fft_bound_vec / fft_one_original_vec
    other_original_vec = torch.fft.ifft(fft_other_original_vec)
    return other_original_vec.real


##################################################
# Even/odd, but has to output symbol
#
# Practice for: Symbol ** 2 -> Symbol

# class IsOddSym(nn.Module):
#     def __init__(self, vec_size, n_in, n_out):
#         super(IsOddSym, self).__init__()
#         self.ff = nn.Sequential(

#             Fn(lambda x, y: x * y, nargs=2),
#             nn.Linear(vec_size, n_in, bias=False),
#             nn.Softmax(dim=1),
#             nn.Linear(n_in, n_out, bias=False),
#             nn.Softmax(dim=1),
#             nn.Linear(n_out, vec_size, bias=False),

#             # Fn(lambda x, y: x * y, nargs=2),
#             # nn.Linear(vec_size, n_in, bias=False),
#             # nn.Softmax(dim=1),
#             # nn.Linear(n_in, vec_size, bias=False),

#         )

#     def forward(self, inp):
#         return self.ff(inp)

# DEVICE = 'cuda'
# VEC_SIZE = 1024
# BATCH_SIZE = 16
# NUM_EPOCHS = 40
# all_symbols = Sym.nums + ['T', 'F']
# int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
# project = int_map.project
# unproject = int_map.unproject

# true_vec = project('T')
# false_vec = project('F')

# odd_dataset = []

# odd_avg = []
# even_avg = []
# for i in range(-10, 11):
#     for j in range(-10, 11):

#         # # Change bias of dataset
#         # if (i+j)%2==0 and i < 0:
#         #     continue

#         x0 = project(i)
#         x1 = project(j)

#         x = (x0, x1)
#         is_odd = (i + j) % 2 == 1
#         y = true_vec if is_odd else false_vec
#         odd_dataset.append((x, y))

#         hv = x0 * x1
#         if (i+j) % 2 == 1:
#             odd_avg.append(hv)
#         else:
#             even_avg.append(hv)

# train_odd_dl = DataLoader(odd_dataset, batch_size=BATCH_SIZE, shuffle=True)

# #####
# # Model
# odd_model = IsOddSym(VEC_SIZE, n_in=2, n_out=2)
# odd_model.cuda()

# #####
# # Test Hack
# odd_avg = torch.stack(odd_avg).mean(dim=0)
# even_avg = torch.stack(even_avg).mean(dim=0)

# # with torch.no_grad():

# #     odd_model.ff[1].weight[0, :] = odd_avg
# #     odd_model.ff[1].weight[1, :] = odd_avg
# #     odd_model.ff[1].weight[2, :] = even_avg
# #     odd_model.ff[1].weight[3, :] = even_avg

# #     # odd_model.ff[3].weight *= 0

# #     # # odd to odd
# #     # odd_model.ff[3].weight[0, 0] = 2.0
# #     # odd_model.ff[3].weight[0, 1] = 2.0
# #     # odd_model.ff[3].weight[1, 0] = 2.0
# #     # odd_model.ff[3].weight[1, 1] = 2.0

# #     # # even to even
# #     # odd_model.ff[3].weight[2, 2] = 2.0
# #     # odd_model.ff[3].weight[2, 3] = 2.0
# #     # odd_model.ff[3].weight[3, 2] = 2.0
# #     # odd_model.ff[3].weight[3, 3] = 2.0

# #     # # off diagonal blocks
# #     # odd_model.ff[3].weight[0, 2] = -2.0
# #     # odd_model.ff[3].weight[0, 3] = -2.0
# #     # odd_model.ff[3].weight[1, 2] = -2.0
# #     # odd_model.ff[3].weight[1, 3] = -2.0

# #     # odd_model.ff[3].weight[2, 0] = -2.0
# #     # odd_model.ff[3].weight[2, 1] = -2.0
# #     # odd_model.ff[3].weight[3, 0] = -2.0
# #     # odd_model.ff[3].weight[3, 1] = -2.0

# #     odd_model.ff[5].weight[:, 0] = true_vec
# #     odd_model.ff[5].weight[:, 1] = true_vec
# #     odd_model.ff[5].weight[:, 2] = false_vec
# #     odd_model.ff[5].weight[:, 3] = false_vec


# #####
# # Train
# opt_params = list(filter(lambda p: p.requires_grad, odd_model.parameters()))
# optimizer = optim.Adam(opt_params, lr=1e-2)
# train_losses = []
# for epoch in range(NUM_EPOCHS):
#     odd_model.train()
#     epoch_loss = 0
#     for _, (src, trg) in enumerate(train_odd_dl):
#         optimizer.zero_grad()
#         output = odd_model(src)
#         loss = (1 - cosine_similarity(output, trg, dim=1)).mean()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         train_loss = epoch_loss / len(train_odd_dl)
#     print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')


# true_odd_n = 0
# true_even_n = 0

# true_odd = 0
# false_odd = 0
# true_even = 0
# false_even = 0
# THRESH = 0.8
# for i in range(-10, 10):
#     for j in range(-10, 10):
#         pi = project(i).unsqueeze(0)
#         pj = project(j).unsqueeze(0)
#         out = odd_model([pi, pj])

#         sim_true = cosine_similarity(out, true_vec.unsqueeze(0), dim=1)
#         sim_false = cosine_similarity(out, false_vec.unsqueeze(0), dim=1)
#         expected = (i+j)%2==1
#         if i in {0, 1} and j in {0, 1, 2}:
#             print(f'{i}+{j}={i+j}. oddp={sim_true.item():>.1f}, evenp={sim_false.item():>.1f}')
#         if expected:
#             true_odd_n += 1
#             if sim_true > THRESH:
#                 true_odd += 1
#             else:
#                 false_even += 1
#         else:
#             true_even_n += 1
#             if sim_false > THRESH:
#                 true_even += 1
#             else:
#                 false_odd += 1

# print(f'true odd  : {true_odd / true_odd_n}')
# print(f'true even : {true_even / true_even_n}')
# print(f'false odd : {false_odd / true_odd_n}')
# print(f'false even: {false_even / true_even_n}')


# # Viz
# model_wi = odd_model.ff[1].weight.squeeze(0)
# model_wo = odd_model.ff[5].weight.squeeze(0)
# print(f'sim0 to odd avg : {cosine_similarity(odd_avg, model_wi[0], dim=0).item() :> .3f}')
# print(f'sim0 to even avg: {cosine_similarity(even_avg, model_wi[0], dim=0).item() :> .3f}')
# print(f'sim1 to odd avg : {cosine_similarity(odd_avg, model_wi[1], dim=0).item() :> .3f}')
# print(f'sim1 to even avg: {cosine_similarity(even_avg, model_wi[1], dim=0).item() :> .3f}')
# print()
# print(f'sim0 to true_vec: {cosine_similarity(true_vec, model_wo[:, 0], dim=0).item() :> .3f}')
# print(f'sim0 to false_vec: {cosine_similarity(false_vec, model_wo[:, 0], dim=0).item() :> .3f}')
# print(f'sim1 to true_vec: {cosine_similarity(true_vec, model_wo[:, 1], dim=0).item() :> .3f}')
# print(f'sim1 to false_vec: {cosine_similarity(false_vec, model_wo[:, 1], dim=0).item() :> .3f}')
# print(f'sim0 to odd_avg: {cosine_similarity(odd_avg, model_wo[:, 0], dim=0).item() :> .3f}')
# print(f'sim0 to even_avg: {cosine_similarity(even_avg, model_wo[:, 0], dim=0).item() :> .3f}')
# print(f'sim1 to odd_avg: {cosine_similarity(odd_avg, model_wo[:, 1], dim=0).item() :> .3f}')
# print(f'sim1 to even_avg: {cosine_similarity(even_avg, model_wo[:, 1], dim=0).item() :> .3f}')


# # sim_odd = []
# # sim_even = []
# # for (x0, x1), y in odd_dataset:
# #     sim = cosine_similarity(x0 * x1, model_w, dim=0).item()
# #     if y > 0.5:
# #         sim_odd.append(sim)
# #     else:
# #         sim_even.append(sim)

# # #plt.figure(figsize=(1, 2))
# # plt.subplot(1, 2, 1)
# # plt.plot(sim_odd)
# # plt.subplot(1, 2, 2)
# # plt.plot(sim_even)

# # plt.tight_layout()
# # plt.show()

# BRK


##################################################
# Even/odd
#
# Practice for: Symbol ** 2 -> Float

class IsOdd(nn.Module):
    ''' 1 dimensional match '''
    def __init__(self, vec_size):
        super(IsOdd, self).__init__()
        self.ff = nn.Sequential(
            Fn(lambda x, y: x * y, nargs=2),
            nn.Linear(vec_size, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, inp):
        return self.ff(inp)

class IsOdd(nn.Module):
    ''' N dimensional match '''
    def __init__(self, vec_size, n_symbols):
        super(IsOdd, self).__init__()
        self.n_symbols = n_symbols
        self.ffs = nn.ModuleList([
            nn.Sequential(

                Fn(lambda x, y: torch.hstack([x, y]), nargs=2),
                nn.Linear(vec_size * 2, 1, bias=False),
                nn.Sigmoid(),

            ) for _ in range(n_symbols)]
        )
        self.guards = nn.ModuleList([
            nn.Sequential(
                Fn(lambda x, y: torch.hstack([x, y]), nargs=2),
                nn.Linear(vec_size * 2, 1, bias=False),
                nn.Sigmoid(),
            ) for _ in range(n_symbols)]
        )

        # self.proj = nn.Linear((n_symbols//2) ** 2, 1, bias=False)

        HD = 16
        self.proj = nn.Sequential(
            nn.Linear((n_symbols//2) ** 2, HD, bias=True),
            nn.ReLU(),
            nn.Linear(HD, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inp):
        outs = []
        for ff, guard in zip(self.ffs, self.guards):
            outs.append(ff(inp) * guard(inp))
        h = self.n_symbols // 2

        # y0 = torch.stack(outs[:h], dim=1).max(dim=1).values
        # y1 = torch.stack(outs[h:], dim=1).max(dim=1).values

        # y0 = torch.stack(outs[:h], dim=1).sum(dim=1)
        # y1 = torch.stack(outs[h:], dim=1).sum(dim=1)
        # return (y1 - y0)

        y0 = torch.stack(outs[:h], dim=1).squeeze(-1)
        y1 = torch.stack(outs[h:], dim=1).squeeze(-1)
        return self.proj(einsum('bx, by -> bxy', y0, y1).flatten(start_dim=1, end_dim=2))


class Choice(nn.Module):
    ''' N-vectors -> [0, 1] '''
    def __init__(self, vec_size, n_vecs, n_hidden, has_guard=False):
        super(Choice, self).__init__()
        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_hidden = n_hidden
        self.has_guard = has_guard

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

        self.ffs = nn.ModuleList([
            nn.Sequential(
                Vecs,
                nn.Linear(vec_size * n_vecs, 1, bias=False),
                nn.Sigmoid(),

            ) for _ in range(n_hidden)]
        )

        if has_guard:
            self.guards = nn.ModuleList([
                nn.Sequential(
                    Vecs,
                    nn.Linear(vec_size * n_vecs, 1, bias=False),
                    nn.Sigmoid(),
                ) for _ in range(n_hidden)]
            )

        self.proj = nn.Sequential(
            nn.Linear((n_hidden//2) ** 2, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inp):
        outs = []
        if self.has_guard:
            for ff, guard in zip(self.ffs, self.guards):
                outs.append(ff(inp) * guard(inp))
        else:
            for ff in self.ffs:
                outs.append(ff(inp))

        h = self.n_hidden // 2

        # y0 = torch.stack(outs[:h], dim=1).max(dim=1).values
        # y1 = torch.stack(outs[h:], dim=1).max(dim=1).values

        # y0 = torch.stack(outs[:h], dim=1).sum(dim=1)
        # y1 = torch.stack(outs[h:], dim=1).sum(dim=1)
        # return (y1 - y0)

        y0 = torch.stack(outs[:h], dim=1).squeeze(-1)
        y1 = torch.stack(outs[h:], dim=1).squeeze(-1)
        return self.proj(einsum('bx, by -> bxy', y0, y1).flatten(start_dim=1, end_dim=2))#.squeeze(1)



class Choice(nn.Module):
    ''' N-vectors -> [0, 1] '''
    def __init__(self, vec_size, n_vecs, n_hidden, has_guard=False):
        super(Choice, self).__init__()
        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_hidden = n_hidden
        self.has_guard = has_guard

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

        self.ff = nn.Sequential(Vecs, nn.Linear(vec_size * n_vecs, n_hidden, bias=False), nn.Sigmoid())
        if has_guard:
            self.guard = nn.Sequential(Vecs, nn.Linear(vec_size * n_vecs, n_hidden, bias=False), nn.Sigmoid())

        self.proj = nn.Sequential(
            nn.Linear((n_hidden//2) ** 2, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inp):
        if self.has_guard:
            f = self.ff(inp)
            g = self.guard(inp)
            outs = f * g
        else:
            outs = self.ff(inp)
        h = self.n_hidden // 2

        y0 = outs[:, :h].max(dim=1).values.unsqueeze(1)
        y1 = outs[:, h:].max(dim=1).values.unsqueeze(1)
        return (y1 - y0 + 1) / 2

        # y0 = outs[:, :h].mean(dim=1).unsqueeze(1)
        # y1 = outs[:, h:].mean(dim=1).unsqueeze(1)
        # return (y1 - y0 + 1) / 2

        # y0 = outs[:, :h]# .squeeze(-1)
        # y1 = outs[:, h:]# .squeeze(-1)
        # return self.proj(einsum('bx, by -> bxy', y0, y1).flatten(start_dim=1, end_dim=2))#.squeeze(1)

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
        assert method in {'max', 'softmax', 'outer_projection'}

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

            # nn.Linear(vec_size * n_vecs, redundancy * n_choices, bias=False),
            # nn.Sigmoid()
        )

        if has_guard:
            self.guard = nn.Sequential(
                Vecs,

                NormalizedLinear(vec_size * n_vecs, redundancy * n_choices, bias=False),

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


            # self.proj = nn.Sequential(
            #     nn.Linear(redundancy ** n_choices, redundancy, bias=True),
            #     nn.ReLU(),
            #     nn.Linear(redundancy, n_choices, bias=True),
            #     nn.Sigmoid(),
            #     # nn.Softmax(dim=1)
            # )

            self.proj = nn.Sequential(
                NormalizedLinear(redundancy ** n_choices, n_choices, bias=False),

                # nn.Linear(redundancy ** n_choices, n_choices, bias=True),

                # nn.Sigmoid(),
                # nn.Softmax(dim=1)
            )

    def forward(self, inp):
        if self.has_guard:
            f = self.ff(inp)
            g = self.guard(inp)
            outs = f * g
        else:
            outs = self.ff(inp)

        if self.method == 'outer_projection':
            chunks = torch.chunk(outs, self.n_choices, dim=1)
            return self.proj(self.outer(chunks))

        # elif self.method == 'mean':
        #     # avg over each chunk via reshaping first
        #     return torch.mean(outs.view(-1, self.n_choices, self.redundancy), dim=2)
        elif self.method == 'max':
            # avg over each chunk via reshaping first
            return torch.max(outs.view(-1, self.n_choices, self.redundancy), dim=2).values
        elif self.method == 'softmax':
            # softmax over the whole redundant vec, then sum each chunk
            return torch.sum(10 * outs.softmax(dim=1).view(-1, self.n_choices, self.redundancy), dim=2)


DEVICE = 'cuda'
VEC_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 40
all_symbols = Sym.nums
int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = int_map.project
unproject = int_map.unproject

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
        y = torch.tensor([(i + j) % 2 * 1.0], device=DEVICE) # determine is_odd
        odd_dataset.append((x, y))

        hv = x0 * x1
        if (i+j) % 2 == 1:
            odd_avg.append(hv)
        else:
            even_avg.append(hv)

train_odd_dl = DataLoader(odd_dataset, batch_size=BATCH_SIZE, shuffle=True)

#####
# odd_model = IsOdd(VEC_SIZE, n_symbols=16)
# METHOD = 'softmax'
# METHOD = 'max'
METHOD = 'outer_projection'
odd_model = nn.Sequential(
    Choice(VEC_SIZE, 2, n_choices=2, redundancy=3, has_guard=False, method=METHOD),
    Fn(lambda x: (x[:,0] - x[:,1] + 1) / 2, nargs=1), # convert softmax choice to scalar
    Fn(lambda x: x.unsqueeze(-1)),
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
        loss = F.mse_loss(output, trg)
        # loss = (1 - F.cosine_similarity(output, trg)).mean()
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
        if out.item() > 0.5 and (i + j) %2 == 1:
            correct += 1
        if out.item() < 0.5 and (i + j) %2 == 0:
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

BRK

##################################################
# Adder
#
# Practice for: (Symbol ** 2) -> Symbol

class Add(nn.Module):
    def __init__(self, vec_size, hidden_dim):
        super(Add, self).__init__()

        self.n_symbols = 200
        self.dropout_p = 0.05

        self.sym_w = nn.Parameter(torch.randn(vec_size, self.n_symbols))
        torch.nn.init.kaiming_normal_(self.sym_w, mode='fan_out', nonlinearity='relu')

        self.ff = nn.Sequential(
            Fn(lambda x, y: (x * y), nargs=2),
            nn.Linear(vec_size, self.n_symbols, bias=False),
            nn.Sigmoid(),
            Fn(lambda x: einsum('bj, ij -> bi', x, self.sym_w)),
        )

    def forward(self, inp):
        return self.ff(inp)


DEVICE = 'cuda'
VEC_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 10
all_symbols = Sym.nums
int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = int_map.project
unproject = int_map.unproject

add_dataset = []
for i in range(-10, 10):
    for j in range(-10, 10):
        x = (project(i), project(j))
        y = project(i + j)
        add_dataset.append((x, y))
train_add_dl = DataLoader(add_dataset, batch_size=BATCH_SIZE, shuffle=True)
add_model = Add(VEC_SIZE, 64)
add_model.cuda()
opt_params = list(filter(lambda p: p.requires_grad, add_model.parameters()))
optimizer = optim.Adam(opt_params, lr=1e-2)
train_losses = []
for epoch in range(NUM_EPOCHS):
    add_model.train()
    epoch_loss = 0
    for _, (src, trg) in enumerate(train_add_dl):
        optimizer.zero_grad()
        output = add_model(src)
        loss = (1 - torch.cosine_similarity(output, trg, dim=1)).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_add_dl)
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')

n = 2
for i in range(-n, n):
    for j in range(-n, n):
        pi = project(i).unsqueeze(0)
        pj = project(j).unsqueeze(0)
        out = add_model([pi, pj])
        uout = unproject(out.squeeze())
        print(f'{i} + {j} = {uout}, {torch.cosine_similarity(out.squeeze(0), project(uout), dim=0):.4f}')

BRK









##################################################
##################################################
##################################################

def circular_convolution(input1, input2):
    """
    Performs circular convolution on two batches of vectors.

    Args:
    - input1: A tensor of shape (batch_size, vector_length)
    - input2: A tensor of shape (batch_size, vector_length)

    Returns:
    - output: The result of circular convolution, of shape (batch_size, vector_length)
    """
    # Check if inputs are of the same shape
    assert input1.shape == input2.shape, "Input tensors must have the same shape"

    # Perform FFT on both inputs
    fft_input1 = torch.fft.fft(input1, dim=1)
    fft_input2 = torch.fft.fft(input2, dim=1)

    # Element-wise multiplication in the frequency domain
    fft_product = fft_input1 * fft_input2

    # Inverse FFT to transform back to the time domain
    output = torch.fft.ifft(fft_product, dim=1)

    # Since ifft returns complex numbers, we take the real part as the output
    return output.real

def sort(matrix):
    """
    Sort the rows of the matrix lexicographically.
    This function assumes -1 < 1 for the purpose of comparison.
    """
    # The idea here is to sort based on the representation of the rows,
    # treating each row as a tuple for comparison. However, because PyTorch
    # does not support direct tuple comparison or lexicographic sorting,
    # this operation is not straightforward in PyTorch. Therefore, we convert
    # to numpy, sort, then convert back if staying purely within PyTorch is not mandatory.

    # Convert to numpy for sorting
    numpy_matrix = matrix.numpy()
    # Lexicographically sort the numpy matrix rows
    sorted_indices = np.lexsort(numpy_matrix.T[::-1])
    sorted_matrix = numpy_matrix[sorted_indices[::-1]]

    # Convert back to PyTorch tensor if necessary
    sorted_matrix_tensor = torch.from_numpy(sorted_matrix)
    return sorted_matrix_tensor

def walsh_matrix(n):
    """Generate the Walsh-Hadamard matrix of size 2^n x 2^n."""
    # Base case: if n=0, return 1x1 matrix with the element 1
    if n == 0:
        return torch.tensor([[1.]])

    # Recursive case: build the matrix for 2^n from 2^(n-1)
    smaller_matrix = walsh_matrix(n-1)
    # Create the top row by concatenating the smaller matrix with itself
    top = torch.cat([smaller_matrix, smaller_matrix], dim=1)
    # Create the bottom row by concatenating the smaller matrix with its negation
    bottom = torch.cat([smaller_matrix, -smaller_matrix], dim=1)
    # Concatenate the top and bottom to form the full matrix
    return torch.cat([top, bottom], dim=0)




def walsh_hadamard_transform(input_vec, walsh_matrix):
    """
    Applies the Walsh-Hadamard transform to the input vector using the provided Walsh matrix.

    Args:
    - input_vec: A tensor of shape (vector_length,) or (batch_size, vector_length) for batches.
    - walsh_matrix: The Walsh matrix to be used for the transformation.

    Returns:
    - transformed_vec: The Walsh-Hadamard transformed vector.
    """
    return torch.matmul(input_vec, walsh_matrix)

def inverse_walsh_hadamard_transform(transformed_vec, walsh_matrix):
    """
    Applies the inverse Walsh-Hadamard transform using the provided Walsh matrix.

    The Walsh-Hadamard transform is its own inverse, but the result needs to be normalized
    by the vector length (which is equivalent to the size of the Walsh matrix).

    Args:
    - transformed_vec: The transformed vector or batch of vectors.
    - walsh_matrix: The Walsh matrix used for the forward transformation.

    Returns:
    - original_vec: The inverse-transformed vector, approximating the original vector.
    """
    size = walsh_matrix.size(0)
    return torch.matmul(transformed_vec, walsh_matrix) / size

def walsh_circular_convolution(input1, input2, walsh_matrix):
    """
    Performs circular convolution on two vectors using the Walsh-Hadamard transform.

    Args:
    - input1: A tensor of shape (vector_length,) or (batch_size, vector_length) for batches.
    - input2: A tensor of shape (vector_length,) or (batch_size, vector_length) for batches.
    - walsh_matrix: The Walsh matrix to be used for the transformation.

    Returns:
    - output: The result of circular convolution, back in the original domain.
    """
    # Transform both inputs to the Walsh domain
    transformed_input1 = walsh_hadamard_transform(input1, walsh_matrix)
    transformed_input2 = walsh_hadamard_transform(input2, walsh_matrix)

    # Element-wise multiplication in the Walsh domain
    product = transformed_input1 * transformed_input2

    # Inverse transform to return to the original domain
    output = inverse_walsh_hadamard_transform(product, walsh_matrix)

    return output


# m = walsh_matrix(4)
# m = sort(m)
# plt.imshow(m)
# plt.show()


EXP = 4
VEC_SIZE = 2 ** EXP
N_VECS = 16
int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = int_map.project
unproject = int_map.unproject

wm = sort(walsh_matrix(EXP))

vecs = F.normalize(torch.randn((N_VECS, VEC_SIZE)), dim=1)
# vecs = wm

test_vec = F.normalize(torch.randn(VEC_SIZE), dim=0) + 0.2

# test_vec = torch.zeros(VEC_SIZE) - 1
# test_vec[:8] = 1

convolutions = []
for v in vecs:
    # convolutions.append(circular_convolution(test_vec.unsqueeze(0), v.unsqueeze(0)))
    convolutions.append(walsh_circular_convolution(test_vec.unsqueeze(0), v.unsqueeze(0), wm))

convolutions = torch.stack(convolutions).squeeze(1)


# ##########
# # Vis

# # Plotting
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# # Plot sorted Walsh matrix
# axs[0].imshow(wm, cmap='viridis', interpolation='nearest')
# axs[0].set_title('Sorted Walsh Matrix')

# # Plot circular convolution results
# axs[1].imshow(convolutions, cmap='viridis', aspect='auto')
# axs[1].set_title('Circular Convolutions')

# plt.tight_layout()
# plt.show()



def unbind_bound_vector(bound_vec, one_original_vec, walsh_matrix):
    """
    Unbinds one vector from a bound vector using the Walsh-Hadamard transform,
    effectively reversing the binding operation to retrieve the other original vector.

    Args:
    - bound_vec: The result of the binding operation (circular convolution in Walsh domain).
    - one_original_vec: One of the original vectors used in the binding operation.
    - walsh_matrix: The Walsh matrix used for transformations.

    Returns:
    - The other original vector, retrieved by undoing the binding.
    """
    # Transform the bound vector and the known original vector to the Walsh domain
    transformed_bound_vec = walsh_hadamard_transform(bound_vec, walsh_matrix)
    transformed_one_original_vec = walsh_hadamard_transform(one_original_vec, walsh_matrix)

    # Element-wise division in the Walsh domain to reverse the binding
    # Note: In practice, ensure to handle division by zero or very small numbers appropriately
    unbound_transformed_vec = transformed_bound_vec / transformed_one_original_vec

    # Inverse transform to get the other original vector
    other_original_vec = inverse_walsh_hadamard_transform(unbound_transformed_vec, walsh_matrix)

    return other_original_vec



def circular_deconvolution(bound_vec, one_original_vec):
    """
    Performs the inverse of circular convolution (deconvolution) to retrieve the other original vector
    given a bound vector and one of the original vectors. This uses FFT/IFFT.

    Args:
    - bound_vec: A tensor of shape (vector_length,) or (batch_size, vector_length) representing the bound vector.
    - one_original_vec: A tensor of shape (vector_length,) or (batch_size, vector_length) representing one of the vectors used in the binding.

    Returns:
    - The other original vector, retrieved by undoing the binding operation.
    """
    # Ensure input is in batch form
    if bound_vec.dim() == 1:
        bound_vec = bound_vec.unsqueeze(0)
    if one_original_vec.dim() == 1:
        one_original_vec = one_original_vec.unsqueeze(0)

    # Perform FFT on both inputs
    fft_bound_vec = torch.fft.fft(bound_vec, dim=1)
    fft_one_original_vec = torch.fft.fft(one_original_vec, dim=1)

    # Element-wise division in the frequency domain to undo the binding
    # Note: Add a small epsilon to the denominator to avoid division by zero
    epsilon = 1e-10
    fft_other_original_vec = fft_bound_vec / (fft_one_original_vec + epsilon)

    # Inverse FFT to transform back to the time domain
    other_original_vec = torch.fft.ifft(fft_other_original_vec, dim=1)

    # Taking the real part assuming the original vectors are real
    return other_original_vec.real


a = circular_convolution(test_vec.unsqueeze(0), vecs[0].unsqueeze(0))
b = circular_deconvolution(a.unsqueeze(0), vecs[0].unsqueeze(0))
c = circular_deconvolution(a.unsqueeze(0), vecs[0].unsqueeze(0))
c[8:] *= 0

print('test_vec:', test_vec)
print('a:', a)
print('b:', b)
print('diff b:', test_vec - b)
print()
print('c:', c)
print('diff c:', test_vec - c)


##################################################


import torch

def convolve(x1, x2):
    """Performs circular convolution on two vectors using FFT and IFFT."""
    fft_x1 = torch.fft.fft(x1)
    fft_x2 = torch.fft.fft(x2)
    fft_product = fft_x1 * fft_x2
    output = torch.fft.ifft(fft_product)
    return output.real

def deconvolve(bound_vec, one_original_vec):
    """Performs the inverse operation of circular convolution to retrieve the original vector."""
    fft_bound_vec = torch.fft.fft(bound_vec)
    fft_one_original_vec = torch.fft.fft(one_original_vec)
    fft_other_original_vec = fft_bound_vec / fft_one_original_vec
    other_original_vec = torch.fft.ifft(fft_other_original_vec)
    return other_original_vec.real


# Generate two random test vectors
vector_length = 256
x1 = torch.randn(vector_length)
x2 = torch.randn(vector_length)

# Perform circular convolution
bound_vec = convolve(x1, x2)

# Deconvolve to retrieve x1
x1_recon = deconvolve(bound_vec, x2)
x2_recon = deconvolve(bound_vec, x1)

assert torch.allclose(x1 - x1_recon, torch.zeros_like(x1), atol=1e-5)
assert torch.allclose(x2 - x2_recon, torch.zeros_like(x2), atol=1e-5)
print('done.')
