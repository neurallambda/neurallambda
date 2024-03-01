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

all_symbols = Sym.nums + Sym.chars
sym_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject
p = project
u = unproject

def set_weights(nand: nn.Module,
                out_vecs: torch.Tensor,
                mappings: List[Tuple[Tuple[str, str], Tuple[int, int], str]],
                redundancy,
                confidence=3):
    '''set weights in nand and out_vecs according to mappings. confidence scales
    the initial nand_weight, which gets sigmoided. When confidence=3,
    sigmoid(3)=0.953, which is confident but with reasonable gradients.

    '''
    with torch.no_grad():
        nand.nand_weight[:, :] = torch.tensor([confidence])

        for idx, (ins, nand_weight, out) in enumerate(mappings):
            out_vecs[:, idx] = project(out) # no redundancy
            for r in range(idx*redundancy, (idx+1)*redundancy):
                nand.weight[r, :] = torch.concat([project(x) for x in ins])
                nand.nand_weight[r, :] = torch.tensor([confidence if x==1 else -confidence for x in nand_weight])

##########
# N_VEC = 1

'''
n_choices = 5
redundancy = 10

out_vecs = torch.randn(VEC_SIZE, n_choices).to(DEVICE)
nand = NAND(VEC_SIZE, 1, n_choices, redundancy, method='softmax').to(DEVICE)

set_weights(
    nand, out_vecs, [
        (('1',), (1,), 'A'),
        (('1',), (0,), 'B'),
        # (('N', 'N'), (0, 0), 'N'),
    ], redundancy)

data = torch.stack([
    torch.concat([p('1'),]),
    torch.concat([p('0'),]),
    torch.concat([p('9')]),
])

choice = nand(data)
out = torch.einsum('bc, vc -> bv', choice, out_vecs)
for o in out:
    print(u(o, return_sim=True))

# for ix in range(nand.weight.size(0)):
#     print(u(nand.weight[ix], return_sim=True), nand.nand_weight[ix])

# for ix in range(out_vecs.size(1)):
#     print(u(out_vecs[:, ix], return_sim=True))

BRK
'''

##########
# N_VEC = 2


n_choices = 10
redundancy = 1

out_vecs = torch.randn(VEC_SIZE, n_choices).to(DEVICE)
nand = NAND(VEC_SIZE, 2, n_choices, redundancy, method='max').to(DEVICE)

set_weights(
    nand, out_vecs, [
        (('1', '2'), (1, 1), 'A'),
        (('1', '2'), (0, 1), 'B'),
        (('1', '2'), (1, 0), 'C'),
        # (('N', 'N'), (0, 0), 'N'),
    ], redundancy)

data = torch.stack([
    torch.concat([p('1'), p('2')]),
    torch.concat([p('0'), p('2')]),
    torch.concat([p('1'), p('0')]),

    torch.concat([p('9'), p('9')]),
])

choice = nand(data)
# choice = choice - 0.02
out = torch.einsum('bc, vc -> bv', choice, out_vecs)
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


##########
