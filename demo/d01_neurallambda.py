'''.

Demo of NeuralLambda.

USAGE:
  # Default Trivial Program
  PYTHONPATH=. python demo/d01_neurallambda.py

  # Demo Programs
  PYTHONPATH=. python demo/d01_neurallambda.py --device cuda --demo_ix 1

  # Custom Program
  PYTHONPATH=. python demo/d01_neurallambda.py --device cuda --n_steps 14 --program "((fn [x] '(x x)) 42)"


PROVENANCE:
  experiment/t00_neurallambda_sandbox.py

'''

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import neurallambda.language as L
import neurallambda.stack as S
import neurallambda.neurallambda as N
import neurallambda.memory as M

torch.set_printoptions(precision=3, sci_mode=False)

SEED = 42
torch.manual_seed(SEED)


##################################################
# Params

N_STACK     = 16   # Stack size
VEC_SIZE    = 4096 # Size of addresses and values
N_ADDRESSES = 24   # Memory size
BATCH_SIZE  = 1

# Garbage collection (overwrites neuralmemory locations with `zero_vec` this
# many times via interpolation)
GC_STEPS = 2


##################################################
# Functions

def reduce_neurallambda(nl, n_stack, start_address, total_steps, gc_steps, device):
    '''Given a neurallambda, step its betareduction `total_steps` forward. This
    function will create a helper stack, and some other helper
    functions.'''

    nb = N.Neuralbeta(nl, n_stack, initial_sharpen_pointer=20)
    nb.to(device)
    nb.push_address(start_address)

    debug_ixs = []

    for step in range(total_steps):
        # The address chosen to be reduced next
        # at_addr = nb.stack.read()
        at_addr = S.read(nb.ss)

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
        recon_mem = N.neurallambda_to_mem(
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


##################################################
# Sample Programs
#
# A curious person may enjoy playing with each different small program, and
# watching how it gets reduced step by step.

# Trivial
sample_programs = [

    # 0)  Trivial Function
    ("((fn [x] x) 42)", 7), # (program, n beta reduction steps)

    # 1)  Simple Fn Application: -> '(1 13)
    ("((fn [x] '(1 x)) 13)", 13),

    # 2)  Multi application: -> '(1 2 3)
    ("((fn [x y z] '(x y z)) 1 2 3)", 38),

    # 3)  Double Function Application: -> '(1 100 10)
    ("((fn [x y] '(1 y x)) 10 100)", 28),

    # 4)  Function passing [x f]: -> '(0 100)
    ("((fn [x f] (f x)) 42 (fn [y] '(0 y y 100 y)))", 42),

    # 5) Composition: -> (fn [z] '('(z z) '(z z) '(z z)))
    #
    # NOTE: this level of complexity is enough that successful reduction depends on
    #       the starting RNG seed. I think that noise issue can be solved tho.
    ("""
(
 (fn [g f z] (g (f z)))
 (fn [y] '(y y y))
 (fn [x] '(x x))
)
""", 53),

    # 6)  Y Combinator: An interesting case, HAS ISSUES.
    #
    #  * corrupts if GC is turned on because referenced memory
    #    cells get zeroed out
    #
    #  * expands and corrupts if GC is off
    ('''
(fn [f] (
  (fn [x] (x x))
  (fn [y] (f (fn [z] ((y y) z))))
))
''', 30)

]


##################################################
# Go!

def main(args):
    if args.program is not None and args.n_steps is not None:
        program = args.program
        total_steps = args.n_steps
    else:
        program, total_steps = sample_programs[args.demo_ix]

    nl = N.string_to_neurallambda(
        program,
        batch_size=BATCH_SIZE,
        n_addresses=N_ADDRESSES,
        vec_size=VEC_SIZE,
        zero_vec_bias=1e-1,
        device=args.device,
    )
    start_ix = 0
    start_address = nl.addresses[:, start_ix]

    with torch.no_grad():
        nb = reduce_neurallambda(
            nl,
            N_STACK,
            start_address,
            total_steps,
            gc_steps=GC_STEPS,
            device=args.device,
        )

    # pretty print results
    S.pp_stack(nb.ss, nl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a program from the sample_programs list.")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("-i", "--demo_ix", type=int, default=0, help="Index of the program to run (default: 0)")
    parser.add_argument("-p", "--program", type=str, help="Custom program to run (default: None)")
    parser.add_argument("-n", "--n_steps", type=int, default=10, help="Number of steps of betareduction to perform (default: 10)")
    args = parser.parse_args()

    main(args)
