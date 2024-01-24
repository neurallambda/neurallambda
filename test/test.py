'''

Tests for e24_neurallambda.py

'''

from dataclasses import dataclass
from lark import Lark, Transformer, Token, Tree
from torch import einsum, tensor, allclose
from torch.nn import functional as F
from typing import Dict, Union, List, Tuple
from typing import Union, List, Any, Type
import numpy as np
import pprint
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import *

torch.set_printoptions(precision=3, sci_mode=False)

SEED = 42
DEVICE = 'cuda'
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
br = breakpoint
print('\n'*200)







# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Neurallambda Utils
#

# @@@@@@@@@@
# Test `replace`

TEST_N = H.Complex

TEST_N_BATCH = 2
TEST_N_ADDRESSES = 5
TEST_N_DIM = 1024

xs = TEST_N.to_mat(TEST_N.randn((TEST_N_BATCH, TEST_N_ADDRESSES, TEST_N_DIM)))
from_ix = 2
to_ix = 4

from_value = xs[:, from_ix, :]
to_value   = xs[:, to_ix,   :]

expected = xs.clone()
expected[:, from_ix] = to_value

result = replace(from_value, to_value, xs)

assert cosine_similarity(result[0, 0], xs[0, 0], dim=0) > 0.8
assert cosine_similarity(result[0, 1], xs[0, 1], dim=0) > 0.8
assert cosine_similarity(result[0, 2], xs[0, 2], dim=0) > 0.8
assert cosine_similarity(result[0, 3], xs[0, 3], dim=0) > 0.8
assert cosine_similarity(result[0, 4], xs[0, 4], dim=0) < 0.2  # dissimilar to original
assert cosine_similarity(result[0, 4], to_value[0], dim=0) < 0.2  # dissimilar to original
assert cosine_similarity(result[0, 4], to_value[1], dim=0) < 0.2  # dissimilar to original
assert cosine_similarity(result[0, 4], from_value[0], dim=0) > 0.8  # similar to new
assert cosine_similarity(result[0, 4], from_value[1], dim=0) < 0.2  # no batch leakage

assert cosine_similarity(result[1, 0], xs[1, 0], dim=0) > 0.8
assert cosine_similarity(result[1, 1], xs[1, 1], dim=0) > 0.8
assert cosine_similarity(result[1, 2], xs[1, 2], dim=0) > 0.8
assert cosine_similarity(result[1, 3], xs[1, 3], dim=0) > 0.8
assert cosine_similarity(result[1, 4], xs[1, 4], dim=0) < 0.2  # dissimilar to original
assert cosine_similarity(result[1, 4], to_value[0], dim=0) < 0.2  # dissimilar to original
assert cosine_similarity(result[1, 4], to_value[1], dim=0) < 0.2  # dissimilar to original
assert cosine_similarity(result[1, 4], from_value[0], dim=0) < 0.2  # no batch leakage
assert cosine_similarity(result[1, 4], from_value[1], dim=0) > 0.8  # similar to new

# @@@@@@@@@@


# @@@@@@@@@@@@@@@@@@@@
# Test `kv-insert`

# @@@@@@@@@@
# Simple
n_batches = 3
n_addresses = 13
key_size = 1024
value_size = 1
keys   = torch.randn((n_addresses, key_size))
values = torch.zeros((n_batches, n_addresses, value_size))
# value at key location should be nearly equal to the new v
for i in range(n_addresses):
    k = keys[i].expand(n_batches, -1, N.dim, N.dim)
    new_v = 17
    v = torch.ones((n_batches, value_size)) * 17
    values_2 = kv_insert(keys, values, k, v, eps=0.1)
    assert (values_2[0][i] - new_v).abs() < 0.001
# if no key matches, there should be no updates
for _ in range(5):
    k = torch.randn((key_size,)).expand(n_batches, -1, N.dim, N.dim)
    new_v = 17
    v = torch.ones((n_batches, value_size)) * 17
    values_2 = kv_insert(keys, values, k, v, eps=0.1)
    diff = (values_2[0][i] - new_v).abs()
    # diff should ~= new_v
    assert diff.item() > new_v - 0.0001
    assert diff.item() < new_v + 0.0001


# @@@@@@@@@@
# Test `kv_insert` so that across batches things that are supposed to update are, and aren't, aren't.

TEST_N_BATCHES = 5
TEST_N_ADDRESSES = 7
TEST_ADDRESS_DIM = 1024

ks     = nn.Parameter(torch.randn((TEST_N_ADDRESSES, TEST_ADDRESS_DIM)))  # .to('cuda'))
vs     = torch.randn((TEST_N_BATCHES, TEST_N_ADDRESSES, TEST_ADDRESS_DIM))  # .to('cuda')
new_vs = torch.randn((TEST_N_BATCHES, TEST_N_ADDRESSES, TEST_ADDRESS_DIM))  # .to('cuda')

for k_ix in [0, 1, 2]:
    batch_k = ks[k_ix].unsqueeze(0).repeat(TEST_N_BATCHES, 1, 1, 1)  # same key for all batches
    # batch_k = N.randn_like(batch_k)  # nonsense key experiment

    batch_v = new_vs[:, k_ix, :]  # get just 1 val from new vals

    # pre insert target
    zero_sim   = torch.zeros((TEST_N_BATCHES, TEST_N_ADDRESSES, TEST_N_ADDRESSES)) # .to('cuda')

    # post insert target
    target_sim = torch.zeros((TEST_N_BATCHES, TEST_N_ADDRESSES, TEST_N_ADDRESSES)) # .to('cuda')
    target_sim[:, k_ix, k_ix] = 1.0

    pre_sim = torch.cosine_similarity(
        vs.unsqueeze(1),      # [TEST_N_BATCHES, 1, TEST_N_ADDRESSES, TEST_ADDRESS_DIM]
        new_vs.unsqueeze(2),  # [TEST_N_BATCHES, TEST_N_ADDRESSES, 1, TEST_ADDRESS_DIM]
        dim=3
    )  # [TEST_N_BATCHES, TEST_N_ADDRESSES, TEST_N_ADDRESSES]

    assert torch.allclose(pre_sim, zero_sim, atol=0.2)
    ins_vs = kv_insert(
        ks,
        vs,
        batch_k,
        batch_v,
        eps=None
    )

    # compare against non-updated locations
    post_sim_vs = torch.cosine_similarity(
        ins_vs.unsqueeze(1),  # [TEST_N_BATCHES, 1, TEST_N_ADDRESSES, TEST_ADDRESS_DIM]
        vs.unsqueeze(2),      # [TEST_N_BATCHES, TEST_N_ADDRESSES, 1, TEST_ADDRESS_DIM]
        dim=3
    )  # [TEST_N_BATCHES, TEST_N_ADDRESSES, TEST_N_ADDRESSES]

    # compare against updated locations
    post_sim_new_vs = torch.cosine_similarity(
        ins_vs.unsqueeze(1),  # [TEST_N_BATCHES, 1, TEST_N_ADDRESSES, TEST_ADDRESS_DIM]
        new_vs.unsqueeze(2),  # [TEST_N_BATCHES, TEST_N_ADDRESSES, 1, TEST_ADDRESS_DIM]
        dim=3
    )  # [TEST_N_BATCHES, TEST_N_ADDRESSES, TEST_N_ADDRESSES]


    assert torch.allclose(post_sim_new_vs, target_sim, atol=0.2)

    # # Vizualization
    # import matplotlib.pyplot as plt
    # for batch_num in range(TEST_N_BATCHES):
    #     plt.subplot(TEST_N_BATCHES, 3, batch_num * 3 + 1)
    #     plt.imshow(target_sim.cpu().detach().numpy()[batch_num])

    #     plt.subplot(TEST_N_BATCHES, 3, batch_num * 3 + 2)
    #     plt.imshow(post_sim_vs.cpu().detach().numpy()[batch_num])

    #     plt.subplot(TEST_N_BATCHES, 3, batch_num * 3 + 3)
    #     plt.imshow(post_sim_new_vs.cpu().detach().numpy()[batch_num])

    # plt.show()


##########
#

for x in range(-10, 10):
    projected_vector = project_int(x)
    assert unproject_int(projected_vector) == x


##########
#

x = torch.stack([
    # base type
    var_tag_vec,
    intlit_tag_vec,

    # not base type
    app_tag_vec,
    fn_tag_vec,
    defn_tag_vec,
])
y = is_base_type(x)
assert y[0] > 0.8
assert y[1] > 0.8
assert y[2] < 0.2
assert y[3] < 0.2
assert y[4] < 0.2

print('ALL TESTS DONE')
