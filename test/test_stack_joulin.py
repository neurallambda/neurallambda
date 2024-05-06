'''

Tests for the Neuralstack, neurallambda.stack

'''

import torch
import neurallambda.stack_joulin as S
from torch import cosine_similarity, einsum
import torch.nn.functional as F
import neurallambda.symbol as Sym
from neurallambda.torch import Fn
from typing import Any, Iterable, List, Optional, Tuple

VEC_SIZE = 256
DEVICE = 'cuda'

tokens = [
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'x1',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'x2',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'x3',
]
all_symbols = Sym.nums + Sym.chars + tokens
sym_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = sym_map.project
p = project
unproject = sym_map.unproject
u = unproject
ps = lambda xs: torch.stack([p(x) for x in xs]) # several projections

batch_size = 3
n_stack = 8

one = torch.ones((batch_size,), device=DEVICE)
zero = torch.zeros((batch_size,), device=DEVICE)
sharp = torch.ones((batch_size, 1), device=DEVICE) * 20

def test_forward_simple():
    ss = S.initialize(VEC_SIZE, n_stack, batch_size, DEVICE)
    aa = ['a1', 'a2', 'a3']
    ss1, _ = S.push_pop_nop(ss, should_push=one, should_pop=zero, should_null_op=zero, value=ps(aa))
    _, pop_val = S.pop(ss1)
    for t, o in zip(aa, pop_val):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

def test_forward_multiple():
    ss = S.initialize(VEC_SIZE, n_stack, batch_size, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']

    nss1, _ = S.push_pop_nop(ss, should_push=one, should_pop=zero, should_null_op=zero, value=ps(aa)) # push
    nss2, _ = S.push_pop_nop(nss1, should_push=one, should_pop=zero, should_null_op=zero, value=ps(bb)) # push
    nss3, _ = S.push_pop_nop(nss2, should_push=one, should_pop=zero, should_null_op=zero, value=ps(cc)) # push

    nss4, out = S.pop(nss3)
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    nss5, out = S.pop(nss4)
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    nss6, out = S.pop(nss5)
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

def test_forward_complex():
    ss = S.initialize(VEC_SIZE, n_stack, batch_size, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']
    dd = ['d1', 'd2', 'd3']
    ee = ['e1', 'e2', 'e3']
    xx = ['x1', 'x2', 'x3']

    nss1, _ = S.push_pop_nop(ss, should_push=one, should_pop=zero, should_null_op=zero, value=ps(aa)) # push
    nss2, _ = S.push_pop_nop(nss1, should_push=one, should_pop=zero, should_null_op=zero, value=ps(bb)) # push

    # null-ops
    nss3, _ = S.push_pop_nop(nss2, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    nss4, _ = S.push_pop_nop(nss3, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    nss5, _ = S.push_pop_nop(nss4, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    nss6, _ = S.push_pop_nop(nss5, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    nss7, _ = S.push_pop_nop(nss6, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))

    # pop
    nss8, out = S.push_pop_nop(nss7, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    nss9, out = S.push_pop_nop(nss8, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    # push more
    nss10, _ = S.push_pop_nop(nss9, should_push=one, should_pop=zero, should_null_op=zero, value=ps(cc)) # push
    nss11, _ = S.push_pop_nop(nss10, should_push=one, should_pop=zero, should_null_op=zero, value=ps(dd)) # push

    # null-op more
    nss12, _ = S.push_pop_nop(nss11, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    nss13, _ = S.push_pop_nop(nss12, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))

    # pop more
    nss14, out = S.push_pop_nop(nss13, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(dd, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    nss15, out = S.push_pop_nop(nss14, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    nss16, _ = S.push_pop_nop(nss15, should_push=one, should_pop=zero, should_null_op=zero, value=ps(ee)) # push

    _, out = S.push_pop_nop(nss16, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(ee, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98


def test_forward_singles():
    ''' test non-forward functions: pop, push, read '''
    ss = S.initialize(VEC_SIZE, n_stack, batch_size, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']

    # push
    nss1 = S.push(ss, ps(aa))
    nss2 = S.push(nss1, ps(bb))
    nss3 = S.push(nss2, ps(cc))

    # pops
    out = S.read(nss3)
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98
    nss4, _ = S.pop(nss3)

    out = S.read(nss4)
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98
    nss5, _ = S.pop(nss4)

    out = S.read(nss5)
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98


def test_forward_doubles():
    ''' test non-forward functions: pop_or_null_op, push_or_null_op '''
    ss = S.initialize(VEC_SIZE, n_stack, batch_size, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']

    # push
    nss1 = S.push_or_null_op(ss, one, zero, ps(aa))
    nss2 = S.push_or_null_op(nss1, one, zero, ps(bb))
    nss3 = S.push_or_null_op(nss2, one, zero, ps(cc))

    # null_ops
    nss4 = S.push_or_null_op(nss3, zero, one, ps(aa))
    nss5 = S.push_or_null_op(nss4, zero, one, ps(aa))
    nss6 = S.push_or_null_op(nss5, zero, one, ps(aa))
    nss7, _ = S.pop_or_null_op(nss6, zero, one)
    nss8, _ = S.pop_or_null_op(nss7, zero, one)
    nss9, _ = S.pop_or_null_op(nss8, zero, one)

    # pops
    nss10, out = S.pop_or_null_op(nss9, one, zero)
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    nss11, out = S.pop_or_null_op(nss10, one, zero)
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    nss12, out = S.pop_or_null_op(nss11, one, zero)
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98
