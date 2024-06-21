'''

Modifying neurallambda's Stack to comport with the stack from: https://arxiv.org/pdf/1503.01007

The original paper conflates the stack datastructure with concerns like trainable params,
stack operations, etc. and frankly, I think makes some harmful choices. For
instance the push_val is passed through an MLP+nonlin before making it onto the
stack. So in this module I only represent the stack datastructure.

The missing pieces you might want are demonstrated in
`demo/d02_palindrome_comparison.py`, and include a linear+softmax for the stack
ops, linear+softmax for the final output, and that linear+sigmoid on the
push_val.

'''

from torch import einsum
import torch
from dataclasses import dataclass


@dataclass
class StackState:
    stack: torch.Tensor  # [batch, n_stack, vec_size]


def push_pop_nop(ss: StackState,
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
    popped_val = einsum('bv, b -> bv', popped_val, should_pop)  # interpolate if pop not intended

    new_stack = (
        einsum('bnv, b -> bnv', ss.stack, should_null_op) +
        einsum('bnv, b -> bnv', push_ss.stack, should_push) +
        einsum('bnv, b -> bnv', pop_ss.stack, should_pop)
    )

    new_ss = StackState(stack=new_stack)
    return new_ss, popped_val


####################
# Single Ops: read, push, pop

def read(ss: StackState) -> torch.Tensor:
    '''Read the top of the stack.'''
    return ss.stack[:, 0]  # [B, S, D] -> [B, D]


def push(ss: StackState, val) -> StackState:
    ''' A guaranteed push op'''
    return StackState(
        torch.cat([val.unsqueeze(1),
                   ss.stack[:, :-1]], dim=1))


def pop(ss: StackState) -> StackState:
    ''' '''
    # read off top
    B, D = ss.stack.size(0), ss.stack.size(2)
    device, dtype = ss.stack.device, ss.stack.dtype
    pop_val = read(ss)
    nss = StackState(
        torch.cat([ss.stack[:, 1:],
                   torch.zeros(B, 1, D, device=device, dtype=dtype)], dim=1))
    return nss, pop_val


####################
# Double Ops: push_or_null_op, pop_or_null_op

def push_or_null_op(ss: StackState,
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
    return StackState(new_stack)


def pop_or_null_op(ss: StackState,
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

    nss = StackState(
        einsum('bnv, b -> bnv', ss.stack, should_null_op) +
        einsum('bnv, b -> bnv', pop_ss.stack, should_pop)
    )
    return nss, popped_val


####################
#

def initialize(vec_size, n_stack, batch_size, device, dtype=torch.float32) -> StackState:
    '''Initialize the stack for a particular run. '''

    # TODO: this zero_offset is likely introducing strong noise when
    #       `forward` is called, and all the addresses get summed down into
    #       one value.
    stack = torch.zeros(
        (batch_size,
         n_stack,
         vec_size), device=device, dtype=dtype)

    return StackState(stack)
