'''

Test NAND stuff

'''

from neurallambda.torch import cosine_similarity
from neurallambda.util import format_number
from neurallambda.nand import NAND
from typing import List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import neurallambda.symbol as Sym

# torch.manual_seed(152)
torch.set_printoptions(precision=3, sci_mode=False)
t = lambda x: torch.tensor([x], dtype=torch.float)

VEC_SIZE = 4096
DEVICE = 'cuda'

all_symbols = Sym.nums + Sym.chars + ['_']
sym_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject
p = project
u = unproject


##################################################
# Tests

def set_weights(model, vals):
    confidence = 5
    with torch.no_grad():
        for i, (vs, nvs) in enumerate(vals):
            model.weight[i, :] = torch.concat([project(x) for x in vs])
            model.nand_weight[i, :] = torch.tensor([confidence if x==1 else -confidence for x in nvs]).to(DEVICE)

##########
# Test N_VEC=1

'''
def test_one_vec_simple():
    n_vecs = 1
    n_choices = 4
    model = NAND(VEC_SIZE, n_vecs, n_choices, clip='leaky_relu', nand_bias=10.0)
    model.to(DEVICE)
    set_weights(model, [
        (('a',), (1,)),
        (('b',), (1,)),
        (('c',), (1,)),
    ])

    # Positive matches
    inps = ['a', 'b', 'c']
    arg_maxes = [0, 1, 2]
    data = torch.stack([p(inp) for inp in inps]).to(DEVICE)
    outs = model(data)
    for out, a in zip(outs, arg_maxes):
        oi = out.argmax().item()
        assert oi == a, f'{oi} != {a}'
        assert out[oi] > 0.95

    # Non-matches
    inps = ['x', 'y', 'z']
    data = torch.stack([p(inp) for inp in inps]).to(DEVICE)
    outs = model(data)
    assert outs.max() < 0.1

def test_one_vec_nots():
    n_vecs = 1
    n_choices = 4
    model = NAND(VEC_SIZE, n_vecs, n_choices, clip='leaky_relu', nand_bias=10.0)
    model.to(DEVICE)
    set_weights(model, [
        (('a',), (1,)),
        (('a',), (0,)),
    ])

    inps = ['a', 'b']
    arg_maxes = [0, 1]
    data = torch.stack([p(inp) for inp in inps]).to(DEVICE)
    outs = model(data)
    for out, a in zip(outs, arg_maxes):
        oi = out.argmax().item()
        assert oi == a, f'{oi} != {a}'
        assert out[oi] > 0.95

##########
# Test N_VEC=2

def test_two_vec():
    n_vecs = 2
    n_choices = 4
    model = NAND(VEC_SIZE, n_vecs, n_choices, clip='leaky_relu', nand_bias=10.0)
    model.to(DEVICE)
    set_weights(model, [
        (('a','b'), (1,1)),
        (('a','b'), (1,0)),
        (('a','b'), (0,1)),
        (('a','b'), (0,0)),
    ])
    arg_maxes = [0, 1, 2, 3]
    data = torch.stack([
        torch.cat((p('a'), p('b')), dim=0),
        torch.cat((p('a'), p('x')), dim=0),
        torch.cat((p('x'), p('b')), dim=0),
        torch.cat((p('x'), p('y')), dim=0),
    ]).to(DEVICE)
    outs = model(data)
    for out, a in zip(outs, arg_maxes):
        oi = out.argmax().item()
        assert oi == a, f'{oi} != {a}'
        assert out[oi] > 0.95

##########
# Test N_VEC=3

def test_three_vec():
    n_vecs = 3
    n_choices = 10
    model = NAND(VEC_SIZE, n_vecs, n_choices, clip='leaky_relu', nand_bias=10.0)
    model.to(DEVICE)
    set_weights(model, [
        (('a','b','c'), (1,1,1)),
        (('a','b','c'), (1,1,0)),
        (('a','b','c'), (1,0,1)),
        (('a','b','c'), (0,1,1)),
        (('a','b','c'), (1,0,0)),
        (('a','b','c'), (0,0,1)),
        (('a','b','c'), (0,1,0)),
        (('a','b','c'), (0,0,0)),
    ])
    arg_maxes = [0, 1, 2, 3, 4, 5, 6, 7]
    data = torch.stack([
        torch.cat((p('a'), p('b'), p('c')), dim=0),
        torch.cat((p('a'), p('b'), p('_')), dim=0),
        torch.cat((p('a'), p('_'), p('c')), dim=0),
        torch.cat((p('_'), p('b'), p('c')), dim=0),
        torch.cat((p('a'), p('_'), p('_')), dim=0),
        torch.cat((p('_'), p('_'), p('c')), dim=0),
        torch.cat((p('_'), p('b'), p('_')), dim=0),
        torch.cat((p('_'), p('_'), p('_')), dim=0),
    ]).to(DEVICE)
    outs = model(data)
    for out, a in zip(outs, arg_maxes):
        oi = out.argmax().item()
        assert oi == a, f'{oi} != {a}'
        assert out[oi] > 0.90

'''

##################################################
# Sandbox

def set_weights_2(nand: nn.Module,
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
        nand.nand_weight[:, :] = torch.tensor([confidence])
        temp_weight = torch.zeros(n_choices, redundancy, nand.weight.size(1), device=nand.weight.device)
        temp_nand_weight = torch.zeros(n_choices, redundancy, nand.nand_weight.size(1), device=nand.nand_weight.device)
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
        nand.nand_weight[:upto] = temp_nand_weight.flatten(start_dim=0, end_dim=1)[:upto]


##########
# N_VEC = 1

'''
n_choices = 5
redundancy = 2

out_vecs = torch.randn(VEC_SIZE, n_choices).to(DEVICE)
nand = NAND(VEC_SIZE, 1, n_choices * redundancy).to(DEVICE)

set_weights(
    nand, out_vecs, [
        ([(('1',), (1,))] * redundancy, 'A'),
        ([(('1',), (0,))] * redundancy, 'B'),
    ], redundancy, confidence=100)

data = torch.stack([
    torch.concat([p('1'),]),
    torch.concat([p('0'),]),
    torch.concat([p('9')]),
])
batch_size = data.size(0)

choice = nand(data)
out = torch.einsum('vc, bcr -> bv', out_vecs, choice.view(batch_size, n_choices, redundancy))
for o in out:
    print(u(o, return_sim=True))
print(choice)

for ix in range(nand.weight.size(0)):
    print(u(nand.weight[ix], return_sim=True), nand.nand_weight[ix])

# for ix in range(out_vecs.size(1)):
#     print(u(out_vecs[:, ix], return_sim=True))

BRK
# '''

##########
# N_VEC = 2

'''
n_choices = 10
redundancy = 4

out_vecs = torch.randn(VEC_SIZE, n_choices).to(DEVICE)
nand = NAND(VEC_SIZE, 2, n_choices * redundancy, clip='leaky_relu').to(DEVICE)

set_weights(
    nand, out_vecs, [
        ([(('1', '2'), (1, 1))] * redundancy, 'A'),
        ([(('1', '2'), (0, 1))] * redundancy, 'B'),
        ([(('1', '2'), (1, 0))] * redundancy, 'C'),
        ([(('N', 'N'), (0, 0))], 'N'),
    ], redundancy)

data = torch.stack([
    torch.concat([p('1'), p('2')]),
    torch.concat([p('0'), p('2')]),
    torch.concat([p('1'), p('0')]),

    torch.concat([p('9'), p('9')]),
])
batch_size = data.size(0)

choice = nand(data)
out = torch.einsum('bcr, vc -> bv', choice.view(batch_size, n_choices, redundancy), out_vecs)
for o in out:
    v, sim = u(o, return_sim=True)
    print(f'{v}, {sim.item():>.3f}')

# for ix in range(nand.weight.size(0)):
#     for nv in [0, 1]:
#         v, sim = u(nand.weight[ix, nv*VEC_SIZE:(nv+1)*VEC_SIZE], return_sim=True)
#         print(f'{v}, {sim.item():>.3f}, {nand.nand_weight[ix].detach().cpu().tolist()}', end='  |  ')
#     print()

# for ix in range(out_vecs.size(1)):
#     print(u(out_vecs[:, ix], return_sim=True))

BRK

# '''

# ##########
# # Better way to do redundancy

# VEC_SIZE = 256
# BATCH_SIZE = 5
# N_CHOICES = 13
# REDUNDANCY = 3

# vecs = torch.randn(VEC_SIZE, N_CHOICES)
# scale = torch.randn(BATCH_SIZE, N_CHOICES * REDUNDANCY)

# out1 = torch.einsum('vc, bcr -> bvr', vecs, scale.view(BATCH_SIZE, N_CHOICES, REDUNDANCY)).sum(dim=-1)
# out2 = torch.einsum('vc, bcr -> bv', vecs, scale.view(BATCH_SIZE, N_CHOICES, REDUNDANCY))
# out3 = torch.einsum('vr, br -> bv', vecs.repeat_interleave(REDUNDANCY, dim=1), scale)  # r_i copies data

# print(out1.shape)
# print(out2.shape)
# print(torch.allclose(out1, out2, rtol=1e-4))
# print(torch.allclose(out1, out3, rtol=1e-4))



##########
# Visualize activation functions

'''
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the input range
x = torch.linspace(-1, 1, 100, dtype=torch.float)

def adjusted_elu(x, alpha=1.0, scale=1.0):
    return torch.where(x > 0, scale * x, alpha * (torch.exp(x) - 1))

# Activation functions
activations = {
    # 'Sigmoid': torch.sigmoid(x),
    'ReLU': torch.relu(x),
    'ELU': F.elu(x),
    'GELU': F.gelu(x),
    # 'Tanh': torch.tanh(x),
    'Softplus': torch.nn.functional.softplus(x),
    'LeakyReLU': torch.nn.functional.leaky_relu(x),
    'Adjusted ELU': adjusted_elu(x, alpha=0.1, scale=1.0),
}

# Plotting
plt.figure(figsize=(10, 8))
for name, activation in activations.items():
    plt.plot(x.numpy(), activation.numpy(), label=name)

plt.title('Activation Functions')
plt.xlabel('Input value')
plt.ylabel('Activation output')
plt.legend()
plt.grid(True)
plt.show()
'''
