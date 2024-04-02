'''

A Finite Differentiable Neuralstack datastructure.

'''

from torch import einsum, tensor, allclose
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.util import transform_runs
import neurallambda.debug as D
from neurallambda.torch import cosine_similarity
from dataclasses import dataclass

@dataclass
class StackState:
    stack: torch.Tensor    # [batch, n_stack, vec_size]
    pointer: torch.Tensor  # [batch, n_stack]
    zero_vec: torch.Tensor # [batch, vec_size]


def push_pop_nop(ss: StackState,
                 sharpen_pointer,
                 should_push,
                 should_pop,
                 should_null_op,
                 value) -> StackState:
    '''Apply all possible stack operations in superposition, and hopefully
    scaled appropriately to signify the *actual* operation you intended.

    Args:

      should_push, should_pop, should_null_op: ndarray([BATCH_SIZE]), values
        in (0, 1). 0 means "dont do this operation", 1 means "do this
        operation". Note: these are NOT constrained to sum to 1.0, the
        caller can choose to do this.

      value: value to push, if pushing. ndarray([BATCH_SIZE, VEC_SIZE])

    '''
    # Push value onto stack
    push_ss = push(ss, value)

    # Pop value off of stack
    pop_ss, popped_val = pop(ss)
    popped_val = einsum('bv, b -> bv', popped_val, should_pop) # interpolate if pop not intended

    new_stack = (
        einsum('bnv, b -> bnv', ss.stack, should_null_op) +
        einsum('bnv, b -> bnv', push_ss.stack, should_push) +
        einsum('bnv, b -> bnv', pop_ss.stack, should_pop)
    )
    new_pointer = (
        einsum('bn, b -> bn', ss.pointer, should_null_op) +
        einsum('bn, b -> bn', push_ss.pointer, should_push) +
        einsum('bn, b -> bn', pop_ss.pointer, should_pop)
    )

    ##########
    # Sharpen (softmax) pointer
    new_pointer = torch.softmax(new_pointer * sharpen_pointer, dim=1)
    psum = new_pointer.sum(dim=1).unsqueeze(1)
    new_pointer = new_pointer / torch.maximum(psum, torch.zeros_like(psum) + 1e-8)

    new_ss = StackState(stack=new_stack,
                        pointer=new_pointer,
                        zero_vec=ss.zero_vec)
    return new_ss, popped_val


####################
# Single Ops: read, push, pop

def read(ss: StackState) -> torch.Tensor:
    '''Read the top of the stack.

    Remember the pointer points at all locations simultaneously, so what
    you'll actually get returned from this call is a sum of all pointer
    locations scaled by confidence that the pointer is actually at that
    location. Recall as well that the pointer is softmaxed, so, the scaling
    of all locations sums to 1.

    '''
    return einsum('bnv, bn -> bv', ss.stack, ss.pointer)

def push(ss: StackState, val) -> StackState:
    ''' '''
    # shift pointer
    new_p = torch.roll(ss.pointer, shifts=1)

    # place val at new pointer
    old_stack = einsum('bnv, bn -> bnv', ss.stack, 1 - new_p)
    new_stack = einsum('bv, bn -> bnv', val, new_p)

    new_ss = StackState(stack=old_stack + new_stack,
                        pointer=new_p,
                        zero_vec=ss.zero_vec)
    return new_ss

def pop(ss: StackState) -> StackState:
    ''' '''
    # read off top
    out = read(ss)

    # zero out memory location @ pointer
    old_stack_ = einsum('bnv, bn -> bnv', ss.stack, 1 - ss.pointer)
    new_stack_ = einsum('bv, bn -> bnv', ss.zero_vec, ss.pointer)
    new_stack = old_stack_ + new_stack_

    # shift pointer back
    new_p = torch.roll(ss.pointer, shifts=-1)

    new_ss = StackState(stack=new_stack,
                        pointer=new_p,
                        zero_vec=ss.zero_vec)
    return new_ss, out


####################
# Double Ops: push_or_null_op, pop_or_null_op

def push_or_null_op(ss: StackState,
                    sharpen_pointer,
                    should_push,
                    should_null_op,
                    value):
    '''Apply push and null_op operations in superposition, and hopefully
    scaled appropriately to signify the *actual* operation you intended.

    Args:

      should_push, should_null_op: ndarray([BATCH_SIZE]), values
        in (0, 1). 0 means "dont do this operation", 1 means "do this
        operation".

      value: value to push, if pushing. ndarray([BATCH_SIZE, VEC_SIZE])

    '''
    push_ss = push(ss, value)
    new_stack = (
        einsum('bnv, b -> bnv', ss.stack, should_null_op) +
        einsum('bnv, b -> bnv', push_ss.stack, should_push)
    )
    new_pointer = (
        einsum('bn, b -> bn', ss.pointer, should_null_op) +
        einsum('bn, b -> bn', push_ss.pointer, should_push)
    )
    # Sharpen (softmax) pointer
    new_pointer = torch.softmax(new_pointer * sharpen_pointer, dim=1)
    psum = new_pointer.sum(dim=1).unsqueeze(1)
    new_pointer = new_pointer / torch.maximum(psum, torch.zeros_like(psum) + 1e-8)
    return StackState(new_stack, new_pointer, ss.zero_vec)

def pop_or_null_op(ss: StackState,
                   sharpen_pointer,
                   should_pop,
                   should_null_op):
    '''Apply pop and null_op stack operations in superposition, and
    hopefully scaled appropriately to signify the *actual* operation you
    intended.

    Args:

      should_pop, should_null_op: ndarray([BATCH_SIZE]), values
        in (0, 1). 0 means "dont do this operation", 1 means "do this
        operation".

      value: value to push, if pushing. ndarray([BATCH_SIZE, VEC_SIZE])

    '''
    pop_ss, popped_val = pop(ss)
    popped_val = einsum('bv, b -> bv', popped_val, should_pop)

    new_stack = (
        einsum('bnv, b -> bnv', ss.stack, should_null_op) +
        einsum('bnv, b -> bnv', pop_ss.stack, should_pop)
    )
    new_pointer = (
        einsum('bn, b -> bn', ss.pointer, should_null_op) +
        einsum('bn, b -> bn', pop_ss.pointer, should_pop)
    )

    ##########
    # Sharpen (softmax) pointer
    new_pointer = torch.softmax(new_pointer * sharpen_pointer, dim=1)
    psum = new_pointer.sum(dim=1).unsqueeze(1)
    new_pointer = new_pointer / torch.maximum(psum, torch.zeros_like(psum) + 1e-8)

    nss = StackState(new_stack, new_pointer, ss.zero_vec)
    return nss, popped_val


####################
#

def initialize(vec_size, n_stack, batch_size, zero_offset, device, dtype=torch.float32):
    '''Initialize the stack for a particular run. '''

    pointer = torch.zeros((batch_size, n_stack), device=device, dtype=dtype)
    pointer[:, 0] = 1 # start stack pointer at ix=0

    # TODO: this zero_offset is likely introducing strong noise when
    #       `forward` is called, and all the addresses get summed down into
    #       one value.
    stack = torch.zeros(
        (batch_size,
         n_stack,
         vec_size), device=device, dtype=dtype) + zero_offset
    zero_vec = torch.zeros(
        (batch_size,
         vec_size), device=device, dtype=dtype) + zero_offset

    return StackState(stack, pointer, zero_vec)


##################################################
# Pretty printer debugging tools

def pp_sim_addresses(nl, stack_ix, stack_val, zero_vec, addresses):
    txts = []
    br_stack_val, br_zero_vec = torch.broadcast_tensors(stack_val, nl.zero_vec[0])
    null_sim = cosine_similarity(br_stack_val, br_zero_vec, dim=0)
    txts.append(D.colorize(f'NULL', value=null_sim.item()))
    for i, a in enumerate(addresses):
        br_stack_val, br_a = torch.broadcast_tensors(stack_val, a)
        sim = cosine_similarity(br_stack_val, br_a, dim=0)
        txt = D.colorize(f'{i:> 2d} ', value=sim.item())
        txts.append(txt)
    return (stack_ix, null_sim, txts)


def pp_stack(ss: StackState, nl):
    ''' Color the indexes in the stack according to the pointer. Calculate a
    similarity between the address at the pointer and every other address in
    `addresses` (and Null). Color code the print out according to
    similarity.
    '''
    addresses = nl.addresses

    NULL_SIM = 0.5 # If stack_val is sim to zero_vec more than this, we'll
                   # collapse its display
    BATCH_I = 0
    print()
    print('STACK:')

    sb = ss.stack[BATCH_I]  # [batch, stack_size, vec_size]
    pb = ss.pointer[BATCH_I] # [batch, stack_size]
    BATCH_I = 0

    similarities = [] # [(pointer_p, stack_ix, null_sim, txt)]
    for stack_ix, (stack_val, p) in enumerate(zip(sb, pb)):
        similarities.append((p,) + pp_sim_addresses(nl, stack_ix, stack_val, nl.zero_vec, addresses[BATCH_I]))

    similarities = transform_runs(
        similarities,
        lambda x, y: x[2] > NULL_SIM and y[2] > NULL_SIM, # if they're very likely null, omit from print
        lambda run: run[0] if len(run) == 1 else (None, None, None, f'{run[0][1]} - {run[-1][1]} : NULL')
    )

    for p, stack_ix, n, t in similarities:
        if p is not None:
            ix_txt = D.colorize(f'{stack_ix:> 2d}', value=p.item())
            print(f'{ix_txt} = {"".join(t)}')
        else:
            print(t)
    print()
