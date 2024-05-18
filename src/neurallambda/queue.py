'''

A Finite Differentiable Neuralqueue.

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
class QueueState:
    queue: torch.Tensor    # [batch, n_queue, vec_size]
    head: torch.Tensor  # [batch, n_queue] - dequeue from head
    tail: torch.Tensor  # [batch, n_queue] - enqueue at tail
    zero_vec: torch.Tensor # [batch, vec_size]


def enqueue_dequeue_nop(
        qs,
        sharpen_head,
        sharpen_tail,
        should_put,
        should_get,
        should_null_op,
        value):
    '''Apply all possible queue operations in superposition, and hopefully
    scaled appropriately to signify the *actual* operation you intended.

    Args:

      should_put, should_get, should_null_op: ndarray([BATCH_SIZE]), values
        in (0, 1). 0 means "dont do this operation", 1 means "do this
        operation".

      value: value to put, if puting. ndarray([BATCH_SIZE, VEC_SIZE])

    '''
    # Put value onto queue
    put_queue, put_tail = put_(qs, value)

    # Get value off of queue
    get_queue, get_head, get_val = get_(qs)

    new_queue = (
        einsum('bnv, b -> bnv', qs.queue, should_null_op) +
        einsum('bnv, b -> bnv', put_queue, should_put) +
        einsum('bnv, b -> bnv', get_queue, should_get)
    )

    new_head = (
        einsum('bn, b -> bn', qs.head, should_null_op) +
        einsum('bn, b -> bn', qs.head, should_put) + # put can't affect head
        einsum('bn, b -> bn', get_head, should_get)
    )

    new_tail = (
        einsum('bn, b -> bn', qs.tail, should_null_op) +
        einsum('bn, b -> bn', put_tail, should_put) +
        einsum('bn, b -> bn', qs.tail, should_get) # get can't affect tail
    )

    ##########
    # Sharpen (softmax) pointers

    new_head = torch.softmax(new_head * sharpen_head, dim=1)
    hsum = new_head.sum(dim=1).unsqueeze(1)
    new_head = new_head / torch.maximum(hsum, torch.zeros_like(hsum) + 1e-8)

    new_tail = torch.softmax(new_tail * sharpen_tail, dim=1)
    tsum = new_tail.sum(dim=1).unsqueeze(1)
    new_tail = new_tail / torch.maximum(tsum, torch.zeros_like(tsum) + 1e-8)

    get_val = einsum('bv, b -> bv', get_val, should_get)

    new_qs = QueueState(new_queue, new_head, new_tail, qs.zero_vec)

    return new_qs, get_val

def read(qs):
    '''Read the top of the queue.

    Remember the pointer points at all locations simultaneously, so what
    you'll actually get returned from this call is a sum of all pointer
    locations scaled by confidence that the pointer is actually at that
    location. Recall as well that the pointer is softmaxed, so, the scaling
    of all locations sums to 1.

    '''
    return einsum('bnv, bn -> bv', qs.queue, qs.head)

def read_n(qs, n):
    '''Read the top of the queue.

    Remember the pointer points at all locations simultaneously, so what
    you'll actually get returned from this call is a sum of all pointer
    locations scaled by confidence that the pointer is actually at that
    location. Recall as well that the pointer is softmaxed, so, the scaling
    of all locations sums to 1.

    '''
    outs = []
    for i in range(n):
        new_q, new_h, out = get_(qs)
        outs.append(out)
        qs = QueueState(new_q, new_h, qs.tail, qs.zero_vec)
    return torch.stack(outs, dim=1)  # [B, n, D]


# def put(qs, val):
#     '''Use this if you're sure you want to `put` only, and this will be
#     baked into the architecture (IE it can't learn to *not* `put`, wherever
#     you use this.'''
#     qs.queue, qs.tail = qs.put_(val)

def put_(qs, val):
    ''' Library consumers should NOT call this function. You must only call
    `forward`.

    NOTE: returns immutable queue and pointers, but does not mutate
    qs.queue nor qs.pointer '''
    # shift pointer
    new_tail = torch.roll(qs.tail, shifts=1)

    # place val at new pointer
    old_queue = einsum('bnv, bn -> bnv', qs.queue, 1 - new_tail)
    new_queue = einsum('bv, bn -> bnv', val, new_tail)
    return old_queue + new_queue, new_tail

def get_(qs):
    '''Library consumers should NOT call this function. You must only call
    `forward`.

    NOTE: returns immutable queue and pointers, but does not mutate
    qs.queue nor qs.pointer '''
    # read off top
    out = read(qs)

    # zero out memory location @ pointer
    old_queue_ = einsum('bnv, bn -> bnv', qs.queue, 1 - qs.head)
    new_queue_ = einsum('bv, bn -> bnv', qs.zero_vec, qs.head)
    new_queue = old_queue_ + new_queue_

    # shift pointers back
    new_head = torch.roll(qs.head, shifts=1)
    return new_queue, new_head, out


def initialize(vec_size, n_queue, batch_size, zero_offset, device, init_head_ix=1, init_tail_ix=0, dtype=torch.float32):
    '''Initialize the queue for a particular run.

    Args:

      init_offset: float. This creates a "zero_vec" in all memory locations
         that isn't actually 0, because 0-only vecs don't play nicely with
         cos-sim.

      initial_sharpen: float. This is a scalar (0, inf) that scales the
        softmax that decides "where the pointer should be". The pointer
        lives in a scaled superposition of pointing at all locations in the
        queue (because it's end-to-end differentiable, like how "attention"
        works). The softmax sharpens this up. A large value (100+) means
        it'll *really* sharpen up the pointer and you'll have incredible
        fidelity. But I suspect it also means that gradients will be zeroed
        for all other values, so may inhibit training (but be good for
        inference).
    '''
    head = torch.zeros((batch_size, n_queue), device=device, dtype=dtype)
    head[:, init_head_ix] = 1  # start queue head at ix=1

    tail = torch.zeros((batch_size, n_queue), device=device, dtype=dtype)
    tail[:, init_tail_ix] = 1  # start queue tail at ix=0

    # TODO: this zero_offset is likely introducing strong noise when
    #       `forward` is called, and all the addresses get summed down into
    #       one value.
    queue = torch.zeros(
        (batch_size,
         n_queue,
         vec_size), device=device, dtype=dtype) + zero_offset

    zero_vec = torch.zeros(
        (batch_size,
         vec_size), device=device, dtype=dtype) + zero_offset
    return QueueState(queue, head, tail, zero_vec)



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Tests

if False:
    BATCH_SIZE = 32
    DEVICE = 'cuda'
    VEC_SIZE = 2048
    N_QUEUE = 16

    ##########
    # Project Ints to Vectors

    int_range_start = -200
    int_range_end   =  200

    # A matrix where each row ix represents the int, projected to VEC_SIZE
    int_vecs = torch.queue([
        torch.randn((VEC_SIZE,))
        for _ in range(int_range_start, int_range_end + 1)
    ]).to(DEVICE)

    def project_int(integer):
        """Projects an integer to a vector space."""
        index = integer - int_range_start
        return int_vecs[index]

    def unproject_int(vector):
        """Unprojects a vector from the vector space back to an integer.

        Assumes matrix formatted `vector`.
        """
        cs = torch.cosine_similarity(vector.unsqueeze(0), int_vecs, dim=1)
        max_index = torch.argmax(cs).item()
        return max_index + int_range_start


    # Test round tripping
    for i in range(int_range_start, int_range_end):
        assert i == unproject_int(project_int(i))

    # @@@@@@@@@@
    #

    SHARP = 100
    put = torch.ones((BATCH_SIZE, )).to(DEVICE)
    no_put = torch.zeros((BATCH_SIZE, )).to(DEVICE)

    get = torch.ones((BATCH_SIZE, )).to(DEVICE)
    no_get = torch.zeros((BATCH_SIZE, )).to(DEVICE)

    null_op = torch.ones((BATCH_SIZE, )).to(DEVICE)
    no_null_op = torch.zeros((BATCH_SIZE, )).to(DEVICE)

    p = lambda x: project_int(x).unsqueeze(0).to(DEVICE) # batch friendly
    u = lambda x: unproject_int(x[0])

    q = Queue(N_QUEUE, VEC_SIZE)
    q.to(DEVICE)
    q.init(BATCH_SIZE, zero_offset=1e-3, device=DEVICE)
    p1 = q(SHARP, SHARP, put, no_get, no_null_op, p(1))
    p2 = q(SHARP, SHARP, put, no_get, no_null_op, p(2))
    p3 = q(SHARP, SHARP, put, no_get, no_null_op, p(3))

    g1 = q(SHARP, SHARP, no_put, get, no_null_op, p(42))
    g2 = q(SHARP, SHARP, no_put, get, no_null_op, p(42))

    p4 = q(SHARP, SHARP, put, no_get, no_null_op, p(4))
    p5 = q(SHARP, SHARP, put, no_get, no_null_op, p(5))

    g3 = q(SHARP, SHARP, no_put, get, no_null_op, p(42))
    g4 = q(SHARP, SHARP, no_put, get, no_null_op, p(42))
    g5 = q(SHARP, SHARP, no_put, get, no_null_op, p(42))

    print()
    print(u(g1))
    print(u(g2))
    print(u(g3))
    print(u(g4))
    print(u(g5))
