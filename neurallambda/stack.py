'''

A Finite Differentiable Neuralstack.

'''

from torch import einsum, tensor, allclose
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.util import transform_runs
import neurallambda.debug as D
from torch import cosine_similarity

class Stack(nn.Module):
    '''A Neural stack. Push, pop, pointer to top of stack, read from top.

    '''

    def __init__(self, n_stack, vec_size):
        '''Initialize the Neuralstack.

        Args:

        '''
        super(Stack, self).__init__()
        self.n_stack = n_stack
        self.vec_size = vec_size

        # run init to populate
        self.stack = None
        self.pointer = None

    def forward(self,
                sharpen_pointer,
                should_push,
                should_pop,
                should_null_op,
                value):
        '''Apply all possible stack operations in superposition, and hopefully
        scaled appropriately to signify the *actual* operation you intended.

        Args:

          should_push, should_pop, should_null_op: ndarray([BATCH_SIZE]), values
            in (0, 1). 0 means "dont do this operation", 1 means "do this
            operation".

          value: value to push, if pushing. ndarray([BATCH_SIZE, VEC_SIZE])

        '''
        # Push value onto stack
        push_stack, push_pointers = self.push_(value)

        # Pop value off of stack
        pop_stack, pop_pointers, popped_val = self.pop_()

        self.stack = (
            einsum('bnv, b -> bnv', self.stack, should_null_op) +
            einsum('bnv, b -> bnv', push_stack, should_push) +
            einsum('bnv, b -> bnv', pop_stack, should_pop)
        )
        self.pointer = (
            einsum('bn, b -> bn', self.pointer, should_null_op) +
            einsum('bn, b -> bn', push_pointers, should_push) +
            einsum('bn, b -> bn', pop_pointers, should_pop)
        )

        ##########
        # Sharpen (softmax) pointers
        self.pointer = torch.softmax(self.pointer * sharpen_pointer, dim=1)
        psum = self.pointer.sum(dim=1).unsqueeze(1)
        self.pointer = self.pointer / torch.maximum(psum, torch.zeros_like(psum) + 1e-8)

        popped_val = einsum('bv, b -> bv', popped_val, should_pop)
        return popped_val

    def read(self):
        '''Read the top of the stack.

        Remember the pointer points at all locations simultaneously, so what
        you'll actually get returned from this call is a sum of all pointer
        locations scaled by confidence that the pointer is actually at that
        location. Recall as well that the pointer is softmaxed, so, the scaling
        of all locations sums to 1.

        '''
        return einsum('bnv, bn -> bv', self.stack, self.pointer)

    def push_(self, val):
        ''' Library consumers should NOT call this function. You must only call
        `forward`.

        NOTE: returns immutable stack and pointers, but does not mutate
        self.stack nor self.pointer '''
        # shift pointer
        new_p = torch.roll(self.pointer, shifts=1)

        # place val at new pointer
        old_stack = einsum('bnv, bn -> bnv', self.stack, 1 - new_p)
        new_stack = einsum('bv, bn -> bnv', val, new_p)
        return old_stack + new_stack, new_p

    def pop_(self):
        '''Library consumers should NOT call this function. You must only call
        `forward`.

        NOTE: returns immutable stack and pointers, but does not mutate
        self.stack nor self.pointer '''
        # read off top
        out = self.read()

        # zero out memory location @ pointer
        old_stack_ = einsum('bnv, bn -> bnv', self.stack, 1 - self.pointer)
        new_stack_ = einsum('bv, bn -> bnv', self.zero_vec, self.pointer)
        new_stack = old_stack_ + new_stack_

        # shift pointers back
        new_p = torch.roll(self.pointer, shifts=-1)
        return new_stack, new_p, out

    def init(self, batch_size, zero_offset, device, dtype=torch.float32):
        '''Initialize the stack for a particular run.

        Args:

          init_offset: float. This creates a "zero_vec" in all memory locations
             that isn't actually 0, because 0-only vecs don't play nicely with
             cos-sim.

          initial_sharpen: float. This is a scalar (0, inf) that scales the
            softmax that decides "where the pointer should be". The pointer
            lives in a scaled superposition of pointing at all locations in the
            stack (because it's end-to-end differentiable, like how "attention"
            works). The softmax sharpens this up. A large value (100+) means
            it'll *really* sharpen up the pointer and you'll have incredible
            fidelity. But I suspect it also means that gradients will be zeroed
            for all other values, so may inhibit training (but be good for
            inference).
        '''
        self.device = device

        self.pointer = torch.zeros((batch_size, self.n_stack), device=device, dtype=dtype)
        self.pointer[:, 0] = 1 # start stack pointer at ix=0
        self.stack = torch.zeros(
            (batch_size,
             self.n_stack,
             self.vec_size), device=device, dtype=dtype) + zero_offset
        self.zero_vec = torch.zeros(
            (batch_size,
             self.vec_size), device=device, dtype=dtype) + zero_offset


##################################################
# Pretty printer debugging tools

def pp_sim_addresses(nl, stack_ix, stack_val, zero_vec, addresses):
    txts = []
    null_sim = cosine_similarity(stack_val, nl.zero_vec[0], dim=0)
    txts.append(D.colorize(f'NULL', value=null_sim.item()))
    for i, a in enumerate(addresses):
        sim = cosine_similarity(stack_val, a, dim=0)
        txt = D.colorize(f'{i:> 2d} ', value=sim.item())
        txts.append(txt)
    return (stack_ix, null_sim, txts)


def pp_stack(stack, nl):
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

    ss = stack.stack[BATCH_I]  # [batch, stack_size, vec_size]
    pp = stack.pointer[BATCH_I] # [batch, stack_size]
    BATCH_I = 0

    similarities = [] # [(pointer_p, stack_ix, null_sim, txt)]
    for stack_ix, (stack_val, p) in enumerate(zip(ss, pp)):
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
