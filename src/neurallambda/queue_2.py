'''

A more Joulin approach to a queue

'''

INCOMPLETE, consider push-only queue


from torch import einsum
import torch
from dataclasses import dataclass


@dataclass
class QueueState:
    queue: torch.Tensor  # [batch, n_queue, vec_size]


def enqueue_dequeue_nop(
    qs: QueueState,
    should_enq,
    should_deq,
    should_null_op,
    value
) -> QueueState:
    '''Apply all possible queue operations in superposition, and hopefully
    scaled appropriately to signify the *actual* operation you intended.

    Args:

      should_enq, should_deq, should_null_op: ndarray([BATCH_SIZE]), values
        in (0, 1). 0 means "dont do this operation", 1 means "do this
        operation". Note: these are NOT constrained to sum to 1.0, the
        caller can choose to do this.

      value: value to enq, if enqing. ndarray([BATCH_SIZE, VEC_SIZE])

    '''
    # Enqueue
    enq_ss = enq(qs, value)

    # Dequeue
    deq_ss, deq_val = deq(qs)
    deq_val = einsum('bv, b -> bv', deq_val, should_deq)  # interpolate if deq not intended

    new_queue = (
        einsum('bnv, b -> bnv', qs.queue, should_null_op) +
        einsum('bnv, b -> bnv', enq_ss.queue, should_enq) +
        einsum('bnv, b -> bnv', deq_ss.queue, should_deq)
    )

    new_ss = QueueState(queue=new_queue)
    return new_ss, deq_val


####################
# Single Ops: read, enq, deq

def read(qs: QueueState) -> torch.Tensor:
    '''Read the top of the queue.'''
    return qs.queue[:, 0]  # [B, S, D] -> [B, D]


def enq(qs: QueueState, val) -> QueueState:
    ''' A guaranteed enq op'''
    return QueueState(
        torch.cat([val.unsqueeze(1),
                   qs.queue[:, :-1]], dim=1))


def deq(qs: QueueState) -> QueueState:
    ''' '''
    # read off top
    B, D = qs.queue.size(0), qs.queue.size(2)
    device, dtype = qs.queue.device, qs.queue.dtype
    deq_val = read(qs)
    nss = QueueState(
        torch.cat([qs.queue[:, 1:],
                   torch.zeros(B, 1, D, device=device, dtype=dtype)], dim=1))
    return nss, deq_val


####################
# Double Ops: enq_or_null_op, deq_or_null_op

def enq_or_null_op(qs: QueueState,
                    should_enq,
                    should_null_op,
                    value):
    '''Apply enq and null_op operations in superposition, and hopefully
    scaled appropriately to signify the *actual* operation you intended.

    Args:

      should_enq, should_null_op: ndarray([BATCH_SIZE]), values
        in (0, 1). 0 means "dont do this operation", 1 means "do this
        operation".

      value: value to enq, if enqing. ndarray([BATCH_SIZE, VEC_SIZE])

    '''
    enq_ss = enq(qs, value)
    new_queue = (
        einsum('bnv, b -> bnv', qs.queue, should_null_op) +
        einsum('bnv, b -> bnv', enq_ss.queue, should_enq)
    )
    return QueueState(new_queue)


def deq_or_null_op(qs: QueueState,
                   should_deq,
                   should_null_op):
    '''Apply deq and null_op queue operations in superposition, and
    hopefully scaled appropriately to signify the *actual* operation you
    intended.

    Args:

      should_deq, should_null_op: ndarray([BATCH_SIZE]), values
        in (0, 1). 0 means "dont do this operation", 1 means "do this
        operation".

      value: value to enq, if enqing. ndarray([BATCH_SIZE, VEC_SIZE])

    '''
    deq_ss, deq_val = deq(qs)
    deq_val = einsum('bv, b -> bv', deq_val, should_deq)

    nss = QueueState(
        einsum('bnv, b -> bnv', qs.queue, should_null_op) +
        einsum('bnv, b -> bnv', deq_ss.queue, should_deq)
    )
    return nss, deq_val


####################
#

def initialize(vec_size, n_queue, batch_size, device, dtype=torch.float32):
    '''Initialize the queue for a particular run. '''

    # TODO: this zero_offset is likely introducing strong noise when
    #       `forward` is called, and all the addresses get summed down into
    #       one value.
    queue = torch.zeros(
        (batch_size,
         n_queue,
         vec_size), device=device, dtype=dtype)

    return QueueState(queue)
