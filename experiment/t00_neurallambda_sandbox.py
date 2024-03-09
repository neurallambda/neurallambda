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
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import neurallambda.language as L
import neurallambda.stack as S
import neurallambda.tensor as T
import neurallambda.memory as M


torch.set_printoptions(precision=3, sci_mode=False)

SEED = 42
DEVICE = 'cpu'
torch.manual_seed(SEED)
random.seed(SEED)
br = breakpoint  # convenience
print('\n'*200)


##################################################
# Params

N_STACK     = 16   # Stack size
VEC_SIZE    = 4096 # Size of addresses and values
N_ADDRESSES = 24   # Memory size
BATCH_SIZE  = 1

# Garbage collection (overwrites memory locations with `zero_vec`)
GC_STEPS = 2


##################################################
# Sample Programs
#
# A curious person may enjoy playing with each different small program, and
# watching how it gets reduced step by step.

# Trivial
x, total_steps = "((fn [x] x) 42)", 7

# # Simple Fn Application: -> '(1 13)
# x, total_steps = "((fn [x] '(1 x)) 13)", 13


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

def step_neurallambda(nl, n_stack, start_address, total_steps, gc_steps):
    '''Given a neurallambda, step its betareduction `total_steps` forward. This
    function will create some a helper stack, and some other helper
    functions.'''

    nb = T.Neuralbeta(nl, n_stack, initial_sharpen_pointer=20)
    nb.push_address(start_address)

    debug_ixs = []

    for step in range(total_steps):
        # The address chosen to be reduced next
        at_addr = nb.stack.read()

        # Perform one step of reduction.
        # tags, col1, col2, ir1, ir2 = nb.reduce_step(at_addr, gc_steps)
        nb.reduce_step(at_addr, gc_steps)
        tags = nb.nl.tags
        col1 = nb.nl.col1
        col2 = nb.nl.col2
        ir1 = nb.ir1
        ir2 = nb.ir2

        ##########
        # Debug
        ix = nl.vec_to_address(at_addr, nl.addresses[0])
        print()
        print(f'STEP {step} @ix={ix} ----------')
        debug_ixs.append(ix)
        recon_mem = T.neurallambda_to_mem(
            nl,
            nl.addresses,
            nl.tags,
            nl.col1,
            nl.col2,
            n_ixs=N_ADDRESSES,
        )

        # Print human-readable memory
        M.print_mem(recon_mem[0], nb.ir1[0], nb.ir2[0])

        ##########
        # Debug Stack
        # pp_stack(stack, addresses)

    print()
    print('ixs visited: ', debug_ixs)
    print('FINAL REDUCTION: ', L.pretty_print(
        M.memory_to_terms(recon_mem[0], M.Address(0),
                          resugar_app=False,
                          resugar_fn=False,
                          )))
    return nb

##########

nl = T.string_to_neurallambda(
    x,
    batch_size=BATCH_SIZE,
    n_addresses=N_ADDRESSES,
    vec_size=VEC_SIZE,
    zero_vec_bias=1e-1,
    device=DEVICE,
)
start_ix = 0
start_address = nl.addresses[:, start_ix]

with torch.no_grad():
# with torch.enable_grad():
    nb = step_neurallambda(
        nl,
        N_STACK,
        start_address,
        total_steps,
        gc_steps=GC_STEPS
    )

S.pp_stack(nb.stack, nl)
