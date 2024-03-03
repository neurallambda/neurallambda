'''

Tests for the Neuralstack, neurallambda.stack

'''

import torch
import neurallambda.stack as S
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
    stack = S.Stack(n_stack, VEC_SIZE)
    stack.init(batch_size, 1e-6, DEVICE)

    aa = ['a1', 'a2', 'a3']
    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(aa))
    out = stack.pop()
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

def test_forward_multiple():
    stack = S.Stack(n_stack, VEC_SIZE)
    stack.init(batch_size, 1e-6, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']

    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(aa)) # push
    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(bb)) # push
    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(cc)) # push

    out = stack.pop()
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    out = stack.pop()
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    out = stack.pop()
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

def test_forward_complex():
    stack = S.Stack(n_stack, VEC_SIZE)
    stack.init(batch_size, 1e-6, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']
    dd = ['d1', 'd2', 'd3']
    ee = ['e1', 'e2', 'e3']
    xx = ['x1', 'x2', 'x3']

    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(aa)) # push
    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(bb)) # push

    # null-ops
    stack(sharp, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    stack(sharp, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    stack(sharp, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    stack(sharp, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    stack(sharp, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))

    # pop
    out = stack(sharp, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    out = stack(sharp, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    # push more
    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(cc)) # push
    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(dd)) # push

    # null-op more
    stack(sharp, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))
    stack(sharp, should_push=zero, should_pop=zero, should_null_op=one, value=ps(xx))

    # pop more
    out = stack(sharp, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(dd, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    out = stack(sharp, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    stack(sharp, should_push=one, should_pop=zero, should_null_op=zero, value=ps(ee)) # push

    out = stack(sharp, should_push=zero, should_pop=one, should_null_op=zero, value=ps(xx)) # pop
    for t, o in zip(ee, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

def test_forward_singles():
    ''' test non-forward functions: pop, push, read '''
    stack = S.Stack(n_stack, VEC_SIZE)
    stack.init(batch_size, 1e-6, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']

    # push
    stack.push(ps(aa))
    stack.push(ps(bb))
    stack.push(ps(cc))

    # pops
    out = stack.read()
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98
    stack.pop()

    out = stack.read()
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98
    stack.pop()

    out = stack.read()
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98


def test_forward_doubles():
    ''' test non-forward functions: pop_or_null_op, push_or_null_op '''
    stack = S.Stack(n_stack, VEC_SIZE)
    stack.init(batch_size, 1e-6, DEVICE)

    aa = ['a1', 'a2', 'a3']
    bb = ['b1', 'b2', 'b3']
    cc = ['c1', 'c2', 'c3']

    # push
    stack.push_or_null_op(sharp, one, zero, ps(aa))
    stack.push_or_null_op(sharp, one, zero, ps(bb))
    stack.push_or_null_op(sharp, one, zero, ps(cc))

    # null_ops
    stack.push_or_null_op(sharp, zero, one, ps(aa))
    stack.push_or_null_op(sharp, zero, one, ps(aa))
    stack.push_or_null_op(sharp, zero, one, ps(aa))
    stack.pop_or_null_op(sharp, zero, one)
    stack.pop_or_null_op(sharp, zero, one)
    stack.pop_or_null_op(sharp, zero, one)

    # pops
    out = stack.pop_or_null_op(sharp, one, zero)
    for t, o in zip(cc, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    out = stack.pop_or_null_op(sharp, one, zero)
    for t, o in zip(bb, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98

    out = stack.pop_or_null_op(sharp, one, zero)
    for t, o in zip(aa, out):
        uo, uo_sim = u(o, return_sim=True)
        assert uo == t
        assert uo_sim.item() > 0.98
