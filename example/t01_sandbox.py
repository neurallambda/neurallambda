'''.

NEURALLAMBDA

BIG CLEANUP

RESULTS: <in progress>

ISSUES:

  - [ ] Move towards making this integratable with RNNs
    - Class to hold Params
    - Reset `is_reduced` / ir?
    - Reset stack?
    - Read programs into memory? We could read in from large embedding-type model.

  - [ ] Turn the sandbox run steps into bonafide function

  - [ ] Make separate tests as these things get fixed

  - [ ] Fix errors when reading out from neurallambda

  - [ ] Trace `is_reduced` values, at the end of reduction, they're all around
        ~0.5. It appears that `ir` values stay ok until the thing is fully
        reduced, then after a couple more steps (once the stack depletes?) they
        jump in a single step to ~0.5. The sudden jump appears only with
        Complex; apparently not with Quat.

  - [ ] Check/test the stack. It seems to not null_op as expected.

  - [ ] `is_base_type` could cause rescaling of values, because it sums
        similarities

  - [ ] The stack doesn't clear memory. Should it? This means that after a
        location pops, the original value remains in memory, but hopefully the
        pointer isn't pointing at it. In the future, it can be overwritten by
        interpolating with a new value, but the orig sticks around
        somewhat. Also consider the balance between inference accuracy and
        information that backprop will need, IE, don't force important info to
        vanish when backprop needed it.

  - [ ] Is this harmful to grads?: `torch.where(alpha > eps, alpha, 0)`

  - [ ] We have a lot of `x.clip(0, 1)`. That seems hacky.

  - [ ] Recursive/Self-referential functions are broken. If GC is turned on, it
        incorrectly cleans up memory locations. Could we avoid parameterized GC,
        and add some mechanism to decide? We could maybe have a recursion
        primitive? Mu-calculus?

  - [ ] Vars are all currently globally scoped. Need local scoping.
    - We can generate unique var names at the time the memory is filled, such
      that it is effectively locally scoped vars.
    - But that doesn't work for recursive functions

  - [X] Visualize Stack
  - [X] Split out tests
  - [X] Colorized debugging of stack
  - [X] Fix Identity Fn Reduction Case: ((fn [x] x) 42)


  - OPTIM:
    - kv_insert needs to find an address location multiple times in
      reduce_app_fn, and this can be reused.
    - handling identity fn in `reduce_app_fn` is expensive, for infrequent payout.
    - cosine_sim vs normalize all vecs and use dot product?
    - Garbage Collection in `reduce_app_fn` is expensive


TODO:

  - Just get it into the world?

  - Debugging tools:
    - distance measure: when printing recon_mem, track inaccuracies in reconstructed term (all multiplied together?)

  - Profiling

  - Slight Inaccuracies? Something's corrupting beta reduction faster than I'd expect.

  - How to integrate within a NN

  - SVD literal as a binary function. So some vector gets blown up into a matrix
    through which vectors can be matmul'd. This function vector may come from a
    library. Unary fns are easy. How about binary... maybe the tag *is* the
    function for these? Or we need to allow partial application of the function,
    to allow unary to consume multiple params.

  - Error correction, by using with.no_grad, and doing lookups in code, and
    replacing with "correct" version?

  - Syntax Update, refactor out: ArithOp, ListP, Car, Cdr. These should be symbols, not primitives.


ABLATION STUDY:

  - Garbage Collection
  - Syntax pieces


COMPLEXITIES, ABSENT HERE
  - memory allocation
  - var scope (user must not reuse var names. a separate renaming procedure could be written)
  - lifetime
  - optimization (inlining, dead code elimination)
  - currying is uncommon. instead call stack and frame pointers.
  - type-checking
  - Reading a program into memory? And guarding that process.


PROVENANCE:
- sandbox/t02_lambda_calc_05.py
- e23_neurallambda_2
- e23_neuralstack

DEPS:
  pip install torch lark numpy


--------------------------------------------------
NOTES:


Lambda Calculus, Fully Differentiable.


Simple Demonstration:


TL;DR:

1. Implement an extended Lambda Calculus, wholly within a (pytorch) tensor
   representation. This Neurallambda is stored in 4 tensors.

2. Implement beta reduction using only e2e differentiable tensor operations.

3. Some QoL tools for writing programs into a Neurallambda, eg a human readable
   string like

```lisp
((fn [x y z] '(x y z)) 1 2 3)
```

turns into 4 tensors aka the Neurallambda.

4. Given a Neurallambda, which is 4 tensors, read back out a human-readable
   string representation of the program.

   Those 4 tensors include: 1) Memory Addresses, 2) A Type tag, 3+4) References
   to other addresses, or literal values.

'''

from dataclasses import dataclass
from lark import Lark, Transformer, Token, Tree
from torch import einsum, tensor, allclose
from typing import Dict, Union, List, Tuple
from typing import Union, List, Any, Type
import neurallambda.hypercomplex as H
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import neurallambda.stack as ns
import neurallambda.tensor as nl

torch.set_printoptions(precision=3, sci_mode=False)

SEED = 42
DEVICE = 'cpu'
torch.manual_seed(SEED)
random.seed(SEED)
br = breakpoint
print('\n'*200)


##################################################
# Params

N_STACK     = 16  # Stack size
VEC_SIZE    = 2048  # Size of addresses and values
N_ADDRESSES = 24  # Memory size
BATCH_SIZE  = 1
N = H.Complex # number system

# Garbage collection (overwrites memory locations with `zero_vec`)
GC_STEPS = 1


##################################################
# Sample Programs
#
# A curious person may enjoy playing with each different small program, and
# watching how it gets reduced step by step.

# # Trivial
# x, total_steps = "((fn [x] x) 42)", 7


# Simple Fn Application: -> '(1 13)
x, total_steps = "((fn [x] '(1 x)) 13)", 13


# # Multi application: -> '(1 2 3)
# x, total_steps = "((fn [x y z] '(x y z)) 1 2 3)", 38


# # Double Function Application: -> '(1 100 10)
# x, total_steps = "((fn [x y] '(1 y x)) 10 100)", 28


# # Function passing [x f]: -> '(0 100)
# x, total_steps = "((fn [x f] (f x)) 42 (fn [y] '(0 y y 100 y)))", 42


# # Composition: -> (fn [z] '('(z z) '(z z) '(z z)))
# #
# # NOTE: this level of complexity is enough that successful reduction depends on
# #       the starting RNG seed. I think that noise issue can be solved tho.
# x, total_steps = """
# (
#  (fn [g f z] (g (f z)))
#  (fn [y] '(y y y))
#  (fn [x] '(x x))
# )
# """, 53


# # Y Combinator: An interesting case, has issues.
# #   -> corrupts if GC is turned on because referenced memory cells get zeroed out
# #   -> expands and corrupts if GC is off
# x, total_steps = '''
# (fn [f] (
#   (fn [x] (x x))
#   (fn [y] (f (fn [z] ((y y) z))))
# ))
# ''', 30, "NOTE: Y-Combinator doesn't work yet"


##################################################
# Go!

def step_neurallambda(total_steps, start_address, addresses, tags, col1, col2, gc):
    '''Given a neurallambda, step its betareduction `total_steps` forward. This
    function will create some a helper stack, and some other helper
    functions.'''

    # `is_reduced` for columns 1 and 2
    ir1 = torch.zeros((BATCH_SIZE, N_ADDRESSES)).to(DEVICE)
    ir2 = torch.zeros((BATCH_SIZE, N_ADDRESSES)).to(DEVICE)

    # stack
    stack = ns.Stack(N_STACK, VEC_SIZE, initial_sharpen=100)
    stack.init(BATCH_SIZE, DEVICE)

    # Convert
    addresses = N.to_mat(addresses)
    tags = N.to_mat(tags)
    col1 = N.to_mat(col1)
    col2 = N.to_mat(col2)

    #####
    # Setup Stack. Initialize by pushing ix=0
    should_push    = torch.ones((BATCH_SIZE,), device=DEVICE)
    should_pop     = torch.zeros((BATCH_SIZE,), device=DEVICE)
    should_null_op = torch.zeros((BATCH_SIZE,), device=DEVICE)

    stack(should_push,
          should_pop,
          should_null_op,
          start_address,
    )

    debug_ixs = []

    for step in range(total_steps):

        # The address chosen to be reduced next
        at_addr = stack.read()

        # Perform one step of reduction.
        tags, col1, col2, ir1, ir2 = reduce(at_addr, addresses, tags, col1, col2, ir1, ir2, gc=GC_STEPS, stack=stack)

        ##########
        # Debug
        ix = vec_to_address(N.from_mat(at_addr), N.from_mat(addresses))
        print()
        print(f'STEP {step} @ix={ix} ----------')
        debug_ixs.append(ix)
        recon_mem = neurallambda_to_mem(
            N.from_mat(addresses),
            N.from_mat(tags),
            N.from_mat(col1),
            N.from_mat(col2),
            n_ixs=N_ADDRESSES,
        )

        # Print human-readable memory
        pm(recon_mem[0], ir1[0], ir2[0])

        ##########
        # Debug Stack
        # pp_stack(stack, addresses)

    print()
    print('ixs visited: ', debug_ixs)
    print('FINAL REDUCTION: ', pretty_print(memory_to_terms(recon_mem[0], A(0),
                                             resugar_app=False,
                                             resugar_fn=False,
                                             )))

##########

addresses, tags, col1, col2 = nl.string_to_neurallambda(
    x,
    number_system=N,
    n_addresses=N_ADDRESSES,
    vec_size=VEC_SIZE,
    device=DEVICE,
    batch_size=BATCH_SIZE,
)
start_ix = 0
start_address = N.to_mat(addresses[:, start_ix])

with torch.no_grad():
# with torch.enable_grad():
    step_neurallambda(total_steps, start_address, addresses, tags, col1, col2, gc=GC_STEPS)
