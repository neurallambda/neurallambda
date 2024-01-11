'''

A Finite Differentiable Neuralstack. Push, pop, pointer to top of stack, read from top.

'''


from torch import einsum, tensor, allclose
import neurallambda.hypercomplex as H
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.util import transform_runs
import neurallambda.debug as D

class Stack(nn.Module):
    ''' NOTE: einsum's have "qr" for making use of matrix-form complex numbers '''

    def __init__(self, n_stack, vec_size, number_system, device, init_offset=1e-3, initial_sharpen=10):
        super(Stack, self).__init__()
        self.init_offset = init_offset
        self.n_stack = n_stack
        self.vec_size = vec_size
        self.number_system = number_system
        self.N = number_system # convenient shorthand
        self.device = device
        self.sharpen_k = nn.Parameter(torch.tensor([initial_sharpen], dtype=torch.float32, device=self.device))
        self.stacks = None # run init to populate
        self.pointers = None # run init to populate

    def forward(self,
                should_push,
                should_pop,
                should_null_op,
                value):
        '''Push and Pop will both be performed no matter what, and `action` signifies
        whether the caller intended a pop or push. The action will scale the
        actual results.

        Args:
          action: ndarray([BATCH_SIZE]). Real value from (-1, 1). -1=pop, +1=push, or 0=null_op
          value: value to push, if pushing. ndarray([BATCH_SIZE, VEC_SIZE])

        '''
        # Push value onto stack
        push_stack, push_pointers = self.push_(value)

        # Pop value off of stack
        pop_stack, pop_pointers, popped_val = self.pop_()

        self.stacks = (
            einsum('bnvqr, b -> bnvqr', self.stacks, should_null_op) +
            einsum('bnvqr, b -> bnvqr', push_stack, should_push) +
            einsum('bnvqr, b -> bnvqr', pop_stack, should_pop)
        )
        self.pointers = (
            einsum('bn, b -> bn', self.pointers, should_null_op) +
            einsum('bn, b -> bn', push_pointers, should_push) +
            einsum('bn, b -> bn', pop_pointers, should_pop)
        )

        ##########
        # Sharpen (softmax) pointers
        self.pointers = torch.softmax(self.pointers * self.sharpen_k, dim=1)
        psum = self.pointers.sum(dim=1).unsqueeze(1)
        self.pointers = self.pointers / torch.maximum(psum, torch.zeros_like(psum) + 1e-8)

        popped_val = einsum('bvqr, b -> bvqr', popped_val, should_pop)
        return popped_val

    def read(self):
        return einsum('bnvqr, bn -> bvqr', self.stacks, self.pointers)

    def push_(self, val):
        ''' NOTE: returns immutable stack and pointers, but does not mutate
        self.stacks nor self.pointers '''
        # shift pointer
        new_p = torch.roll(self.pointers, shifts=1)

        # place val at new pointer
        old_stack = einsum('bnvqr, bn -> bnvqr', self.stacks, 1 - new_p)
        new_stack = einsum('bvqr, bn -> bnvqr', val, new_p)
        return old_stack + new_stack, new_p

    def pop_(self):
        ''' NOTE: returns immutable stack and pointers, but does not mutate
        self.stacks nor self.pointers '''
        # read off top
        out = self.read()

        # zero out memory location @ pointer
        old_stack_ = einsum('bnvqr, bn -> bnvqr', self.stacks, 1 - self.pointers)
        new_stack_ = einsum('bvqr, bn -> bnvqr', self.zero_vec, self.pointers)
        new_stack = old_stack_ + new_stack_

        # shift pointers back
        new_p = torch.roll(self.pointers, shifts=-1)
        return new_stack, new_p, out

    def init(self, batch_size):
        self.pointers = torch.zeros((batch_size, self.n_stack), device=self.device)
        self.pointers[:, 0] = 1 # start stack pointer at ix=0
        self.stacks = torch.zeros(
            (batch_size,
             self.n_stack,
             self.vec_size,
             self.N.dim,
             self.N.dim), device=self.device) + self.init_offset
        self.zero_vec = torch.zeros(
            (batch_size,
             self.vec_size,
             self.N.dim,
             self.N.dim), device=self.device) + self.init_offset



##################################################
# Pretty printer debugging tools

def pp_sim_addresses(nl, stack_ix, stack_val, zero_vec_mat, addresses):
    txts = []
    null_sim = H.cosine_similarity(nl.N.to_mat(stack_val), nl.zero_vec_mat[0], dim=0)
    txts.append(D.colorize(f'NULL', value=null_sim.item()))
    for i, a in enumerate(addresses):
        sim = H.cosine_similarity(nl.N.to_mat(stack_val), a, dim=0)
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

    ss = nl.N.from_mat(stack.stacks[BATCH_I])  # [batch, stack_size, vec_size, complex, complex]
    pp = stack.pointers[BATCH_I] # [batch, stack_size]
    BATCH_I = 0

    similarities = [] # [(pointer_p, stack_ix, null_sim, txt)]
    for stack_ix, (stack_val, p) in enumerate(zip(ss, pp)):
        similarities.append((p,) + pp_sim_addresses(nl, stack_ix, stack_val, nl.zero_vec_mat, addresses[BATCH_I]))

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
