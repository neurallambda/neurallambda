'''

Actual Neurallambdas, ie the tensor form of Lambdacalc.

'''

from torch import einsum, tensor, allclose
from typing import Dict, Union, List, Tuple
from typing import Union, List, Any, Type
import neurallambda.hypercomplex as H
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import neurallambda.memory as mem
from neurallambda.config import Number, N

###########
# PROBE TOOLS

da = lambda at_addr: vec_to_address(N.from_mat(at_addr[0]), N.from_mat(addresses))


##################################################
# Replace and KV-Insert

def replace(new_value, prev_value, tensr):
    ''' Replace any `prev_value` with `new_value`, according to cos_sim of
    prev_value with values in tensr'''
    # prev_valuex.shape = [BATCH, DIM]
    # prev_value.shape   = [BATCH, DIM]
    # tensr.shape     = [BATCH, N_ADDRESSES, DIM]

    # we'll interpolate into the `to` location with `sim_to`
    sim_to = H.cosine_similarity(prev_value.unsqueeze(1), tensr, dim=2)  # shape = [BATCH, N_ADDERSSES]
    keep = einsum('bndqr, bn -> bndqr', tensr, 1 - sim_to)
    rep = einsum('bn, bdqr -> bndqr', sim_to, new_value)
    return  keep + rep


def kv_insert(state_k, state_v, k, x, eps=1e-8, cos_sim_keys=True):
    '''
    Args:
      state_k: [batch?, address, key]   # batch dim optional
      state_v: [batch, address, val]
      k: [batch, key]
      v: [batch, val]
    Returns:
      state_v with the address of `k` updated to `x`
    eps is a hard cutoff for low-probability matches
    '''
    H.assert_is_probably_mat_form(state_k)
    H.assert_is_probably_mat_form(k)

    if state_k.ndim == 2 + 2:  # not batched + hypercomplex mat dim
        # the similarity of each rule key, to each state key
        alpha = H.cosine_similarity(k.unsqueeze(1), state_k.unsqueeze(0), dim=2) # [batch, address]
        if eps is not None:
            alpha = torch.where(alpha > eps, alpha, 0) # TODO: is this harmful to grads?
        state_v = (
            einsum('ba, bav -> bav', 1 - alpha, state_v) +
            einsum('ba, bv -> bav', alpha    , x)
        )
        return state_v

    elif state_k.ndim == 3 + 2:  # batched + hypercomplex mat dim
        alpha = H.cosine_similarity(
            state_k,
            k.unsqueeze(1),
            dim=2
        ) # the similarity of each rule key, to each state key
        if eps:
            alpha = torch.where(alpha > eps, alpha, 0) # TODO: is this harmful to grads?

        # TODO: this is such a hack. It might be better to newtype Hypercomplex
        #       numbers (maybe even mat/non-mat versions separately)
        if state_v.ndim == 3: # values are not hypercomplex
            state_v = (
                einsum('ba, bav -> bav', 1 - alpha, state_v) +
                einsum('ba, bv -> bav', alpha    , x)
            )
            return state_v
        elif state_v.ndim == 3 + 2: # values are hypercomplex. TODO: this is an ugly hack
            state_v = (
                einsum('ba, bavqr -> bavqr', 1 - alpha, state_v) +
                einsum('ba, bvqr -> bavqr', alpha    , x)
            )
            return state_v


##################################################
# Compile Memory to Tensors



##########

tag_names = [
    'NULL',

    # Lambda
    'App',
    'Fn',
    'Defn',
    'DefnName',

    # Base types
    'Var',
    'IntLit',
    'Empty',
    'ArithOp',
    'TrueLit',
    'FalseLit',

    # Not Lambda, Not Base
    'Cons',
    'Car',
    'Cdr',
    # 'LinkedList',
    # 'ListP',

]

literal_tags = {'IntLit', 'TrueLit', 'FalseLit', 'ArithOp', 'DefnName'}
nullary_tags = {'Empty'}
unary_tags   = {'Var', 'ArithOp', 'Car', 'Cdr'}
binary_tags  = {'App', 'Fn', 'Defn', 'Cons'}


# assure sorting
for expected_ix, tag in enumerate(tag_names):
    assert tag_names[expected_ix] == vec_to_tag(tag_to_vec[tag])


##########
# String Encodings

# Letters
chars = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'.split(' ')
char_to_int_ = {c: i for i, c in enumerate(chars)}
def char_to_int(c):
    if c in char_to_int_:
        return char_to_int_[c]
    return -666

int_to_char_ = {i: c for i, c in enumerate(chars)}
def int_to_char(i):
    if i in int_to_char_:
        return int_to_char_[i]
    return '#'

for c in chars:
    d = int_to_char(unproject_int(project_int(char_to_int(c))))
    assert c == d

# ArithOp
arithops = '+ - / *'.split(' ')
arithop_to_int = {c: i for i, c in enumerate(arithops)}
int_to_arithop = {i: c for i, c in enumerate(arithops)}


##########
# Compile to NeuralLambdas

def compile_neurallambdas(mem: Dict[mem.Address, Any]):
    ''' Given dictionary of memory `mem`, build tensor set representing the neurallamda. '''
    addresses = N.randn((BATCH_SIZE, N_ADDRESSES, VEC_SIZE)).to(DEVICE)
    tags  = torch.zeros((BATCH_SIZE, N_ADDRESSES, VEC_SIZE, N.dim), device=DEVICE) + zero_vec
    col1 = torch.zeros((BATCH_SIZE, N_ADDRESSES, VEC_SIZE, N.dim), device=DEVICE) + zero_vec
    col2 = torch.zeros((BATCH_SIZE, N_ADDRESSES, VEC_SIZE, N.dim), device=DEVICE) + zero_vec
    blocks = sorted(mem.items(), key=lambda item: item[0].i)
    for addr, block in blocks:
        tags[:, addr.i] = tag_to_vec[block[0]]

        # Column 1
        if len(block) >= 2 and isinstance(block[1], mem.Address):
            term1_addr = block[1]
            col1[:, addr.i] = addresses[:, term1_addr.i]

        # Column 2
        if len(block) >= 3 and isinstance(block[2], mem.Address):
            term2_addr = block[2]
            col2[:, addr.i] = addresses[:, term2_addr.i]

        # IntLit
        if len(block) >= 2 and block[0] == 'IntLit':
            val = block[1]
            col1[:, addr.i] = project_int(val)

        # Var
        if len(block) >= 2 and block[0] == 'Var':
            val = block[1]
            col1[:, addr.i] = project_int(char_to_int(val))

        # ArithOp
        if len(block) >= 2 and block[0] == 'ArithOp':
            val = block[1]
            col1[:, addr.i] = project_int(arithop_to_int[val])

    return addresses, tags, col1, col2


def string_to_neurallambda(s: str):
    parser = Lark(grammar, start='start', parser='lalr')
    transformer = LambdaTransformer()
    tree = parser.parse(s)
    ast = transformer.transform(tree)
    mem = terms_to_memory(ast)
    return compile_neurallambdas(mem)


##########
# Decompile Neural Lambdas

def read_col(tag, vec, addresses):
    ''' Project a neurallambda `vec` back to the machine language. A `tag` determines how it should be read. '''
    H.assert_is_probably_not_mat_form(vec)
    H.assert_is_probably_not_mat_form(addresses)

    if tag == 'IntLit':
        return unproject_int(vec)
    elif tag == 'Var':
        return int_to_char(unproject_int(vec))
    elif tag == 'ArithOp':
        return int_to_arithop[unproject_int(vec)]
    elif tag in {'TrueLit', 'FalseLit'}:
        return zero_vec
    elif tag == 'Empty':
        return zero_vec

    # NULL
    z = zero_vec if zero_vec.ndim == 1 else zero_vec[0] # single batch
    z = N.to_mat(z)
    c = H.cosine_similarity(N.to_mat(vec), z, dim=0)
    if c  > 0.5:
        return ('NULL', )

    # Address
    return mem.Address(vec_to_address(vec, addresses))

def neurallambda_to_mem(addresses, tags, col1, col2, n_ixs) -> Dict[mem.Address, Any]:
    ''' Reverse engineer the tensor set of a neurallambda back into a dictionary mem. '''
    H.assert_is_probably_not_mat_form(addresses)
    H.assert_is_probably_not_mat_form(tags)
    H.assert_is_probably_not_mat_form(col1)
    H.assert_is_probably_not_mat_form(col2)

    if tags.ndim == 2 + 1:  # non-batched, add 1 hypercomplex dim
        recon_mem = {}
        for i in range(n_ixs):
            ai = mem.Address(i)
            t = vec_to_tag(tags[i])

            if t == 'NULL':
                recon_mem[ai] = (t,)
                continue

            # NULLARY
            if t in nullary_tags:
                recon_mem[ai] = (t,)

            # UNARY or LITERAL
            if t in literal_tags or t in unary_tags:
                v = read_col(t, col1[i], addresses)
                recon_mem[ai] = (t, v)

            # BINARY
            if t in binary_tags:
                v1 = read_col(t, col1[i], addresses)
                v2 = read_col(t, col2[i], addresses)
                recon_mem[ai] = (t, v1, v2)
        return recon_mem

    elif tags.ndim == 3 + 1:  # batched, add 1 hypercomplex dim
        out = []

        if addresses.ndim == 2 + 1:  # static addresses per all batches, add 1 hypercomplex dim
            for t, c1, c2 in zip(tags, col1, col2):
                out.append(neurallambda_to_mem(addresses, t, c1, c2, n_ixs))

        elif addresses.ndim == 3 + 1:  # different addresses per batch, add 1 hypercomplex dim
            for a, t, c1, c2 in zip(addresses, tags, col1, col2):
                out.append(neurallambda_to_mem(a, t, c1, c2, n_ixs))

        return out
    raise ValueError(f"Saw neurallambda's tags with unexpected shape: {tags.shape}")


##############################
# Beta Reduction of NeuralLambdas

'''.

BRAINSTORMING how to reduce Neurallambdas?:

- 2 pass. First traverse tree, push everything into a stack. Second, pop from
  stack only, and reduce as you go. I think this wouldn't support online
  recursion.

- [X] Online traversal issue: If you see a base type, pop the stack. The next
  address likely references that base type, so, push it's location back onto the
  stack, and infinite loop.

- [X] solution? include two address-aligned "is_reduced" tensors for col1 and
  col2. It doesn't mark a term, it marks references to terms, as to whether
  they're reduced.

- solution? instead of 2 new tensors, how about superposing an address with
  "is_reduced" or "is_not_reduced"?  How to update that value? add the negative,
  scaled by cossim? Look out for addresses that have been noised, because the
  original superposed value will have drifted. (NOTE: Tried, too noisy)


BRAINSTORMING how to tag `is_(not_)reduced`:

How to store `is_reduced`:

    - [X] Separate tensors for noting which addresses are reduced

        - or one index on the address tensor devoted to `is_(not_)reduced`

    - Superposition of address and an `is_(not_)reduced` tag. In experiments,
      the superposed vecs got too noisy too fast.

    - 2 orthogonal sets of address ints. EG replace A(1) with orthogonal vec IsReducedA(1)

    - Separate cones of vector space: `is_reduced` could be same address * -1

    - Rotate points above / below a hyperplane (requires ensuring that all
      integers are above some hyperplane

    - Store in magnitude of vectors


When / how to process `is_reduced`

    - when non-app-fn reduction happens, peek ahead to see if they're reduced?

    - [X] when we've descended to a base type, it `is_reduced`, and we update
      every reference to this address simultaneously
'''


##########
# Reduction Functions

def assert_mat_form(x):
    ''' Not guaranteed, but, hopefully catches bad dims '''
    assert x.shape[-1] == x.shape[-2], f'hypercomplex number must be in matrix format (ie shape=[..., 2, 2]), but has shape={x.shape}'
    assert x.shape[-1] in {1, 2, 4}, f'hypercomplex number in matrix form should have dim in (1, 2, 4), but has shape={x.shape}'


def address_similarity(address, addresses):
    assert_mat_form(address)
    assert_mat_form(addresses)

    ##########
    # Cos Sim Solution
    if len(addresses.shape) == 2 + 2: # no batches + hypercomplex mat
        cs = H.cosine_similarity(address, addresses, dim=1)
        return cs
    elif len(addresses.shape) == 3 + 2:  # each batch has own addresses + hypercomplex mat
        cs = H.cosine_similarity(address, addresses, dim=2)
        return cs


def select_address(address, addresses, list_of_values):
    ''' Given an address, and a bunch of addresses-values, get each value at the
    given address.

    Args:
      address: ndarray([batch, address_size])
      addresses: ndarray([n_addresses, address_size])
      list_of_values: List[ ndarray(batch, n_addresses, arbitrary_size) ]
    '''
    cs = address_similarity(address, addresses)
    out = []
    for values in list_of_values:
        x = H.scale(values, cs.unsqueeze(-1))  # elem-wise multiplication
        x = x.sum(dim=1)  # collapse all addresses into one
        out.append(x)
    return out


def reduce_app_fn(at_addr, addresses, tags, col1, col2, gc=True):
    '''Reduce ((fn [x] x) y)  --to-->  y

    This assumes you know that the tag at `at_addr` is App, and the
    referred term in `col1` resolves to a `Fn`.

    Var addresses are never locally scoped, only globally. So when a reduction
    happens with a var at address=`a`, all references to `a` throughout the
    enitre memory are replaced with the substitute.

    Args:
      at_addr: ndarray([batch, vec_size])
      addresses: ndarray([batch, n_addresses, vec_size])
      tags, col1, col2: <same shape as addresses>

    TODO: OPTIM address lookup cos-sim stuff could be reused across kv_inserts
    '''
    assert_mat_form(at_addr)
    assert_mat_form(addresses)
    assert_mat_form(tags)
    assert_mat_form(col1)
    assert_mat_form(col2)

    fn_addr, arg_addr = select_address(at_addr, addresses, [col1, col2])
    fn_tag, param_addr, body_addr = select_address(fn_addr, addresses, [tags, col1, col2])

    # Replace values at parameter address with values from argument
    # address. This replaces eg `Var x` with the values that are getting subbed
    # in.
    #
    # Normally this doesn't matter, since the body of the function is likely
    # some structure that mereley refers to the address of `Var x`, and not the
    # values of `Var x`, and therefore we should be able to just replace the
    # address, not the values. But if you have an identity function, eg `(fn [x]
    # x)`, the body doesn't refer to `Var x`, it is `Var x`. So we can't just
    # replace addresses, we need to replace the values of `Var x` too.
    a_tag, a_l, a_r = select_address(arg_addr, addresses, [tags, col1, col2])
    tags = kv_insert(addresses, tags, param_addr, a_tag)
    col1 = kv_insert(addresses, col1, param_addr, a_l)
    col2 = kv_insert(addresses, col2, param_addr, a_r)

    # Convert param references/address into arg references/address (replace
    # every instance (THROUGHOUT ALL MEM / all col1 and col2) of `param_addr`
    # with `arg_addr`)
    #
    # NOTE: these lines are responsible for our INABILITY to have LOCALLY SCOPED
    #       vars.
    col1 = replace(arg_addr, param_addr, col1)  # replace(new_value, prev_value, tensor)
    col2 = replace(arg_addr, param_addr, col2)

    # Replace `App`'s address with the function's `body_addr`, thus eliminating an App and Fn
    #   kv_insert(state_k, state_v, k, x, eps=None)
    b_tag, b_l, b_r = select_address(body_addr, addresses, [tags, col1, col2])
    # b_tag: ndarray([batch, vec_size])
    tags = kv_insert(addresses, tags, at_addr, b_tag)
    col1 = kv_insert(addresses, col1, at_addr, b_l)
    col2 = kv_insert(addresses, col2, at_addr, b_r)

    # Garbage Collection: overwrite locations with zero_vec/NULL
    #
    # WARN: multiple passes can help get it right, but, too many hurts it. Also,
    #       this must be expensive/detrimental for backprop.

    for _ in range(GC_STEPS):
        # Erase Fn
        tags = kv_insert(addresses, tags, fn_addr, zero_vec_mat)
        col1 = kv_insert(addresses, col1, fn_addr, zero_vec_mat)
        col2 = kv_insert(addresses, col2, fn_addr, zero_vec_mat)

        # Erase Fn's Bound Var
        tags = kv_insert(addresses, tags, param_addr, zero_vec_mat)
        col1 = kv_insert(addresses, col1, param_addr, zero_vec_mat)
        col2 = kv_insert(addresses, col2, param_addr, zero_vec_mat)

        # Erase original Fn Body
        tags = kv_insert(addresses, tags, body_addr, zero_vec_mat)
        col1 = kv_insert(addresses, col1, body_addr, zero_vec_mat)
        col2 = kv_insert(addresses, col2, body_addr, zero_vec_mat)
    return tags, col1, col2


def is_base_type(x):
    ''' A Base type is that without any Addresses within it, eg Var, IntLit,
    Empty, etc. First dim of `x` is `batch_dim`.

    Args:
      x: shape=[BATCH, VEC_SIZE] in hypercomplex matrix format

    Warning: values can exceed (-1, 1)
    '''
    sims = H.cosine_similarity(
        x.unsqueeze(1),
        base_types_vecs.unsqueeze(0),
        dim=2).clip(0, 1).sum(dim=1)
    return sims


def reduce(at_addr,
           addresses, tags, col1, col2,
           ir1, ir2, # `is_reduced` col1 and col2
           gc:bool,  # should it garbage collect?
           stack,
           ):
    '''How to Reduce a Neurallambda?

    HIGH LEVEL:

    Reduce terms at an address if possible, control the stack, return next
    addresses to reduce at. This all must remain differentiable. This means that
    all possible cases are run in superposition at every step, but scaled so
    that incorrect operations have minimal effect on the process.

    DETAILS:

    1. Retrieve values at an address

    2. Three cases:

      A. base case/primitives. Pop the stack, return the address on top of stack
         as the next location to reduce at.

      B. (App (Fn ...) ...). We can try beta reduction!

      C. Non-basecase, Non App-Fn. So, a type that has open address slots.


    NOTE: variable names are all GLOBALLY SCOPED. so when `(App (Fn [x] _) _)`
          is reduced, that `x` will get substituted in the body, BUT ALSO EVERY
          `x` in memory will get substituted too with that specific value. I
          engineered this before inventing this project's Neuralstack, so it had
          to be that way. But with a Neuralstack, local scoping may be
          possible (?) (TODO)

    TODO: marking `is_reduced` has given me some troubles. I have mixed 2
          approaches, problematically:

        1. an approach where I try to read an address and determine if the thing
           is a base type that has been previously marked as reduced.

        2. an approach where base types are never marked as `is_reduced`, but when
           they're found, all references to them are marked as reduced.
    '''
    H.assert_is_probably_mat_form(at_addr)
    H.assert_is_probably_mat_form(addresses)
    H.assert_is_probably_mat_form(tags)
    H.assert_is_probably_mat_form(col1)
    H.assert_is_probably_mat_form(col2)

    # unpack stuff at `at_addr`
    head_tag, col1_addr, col2_addr = select_address(at_addr, addresses, [tags, col1, col2])

    # scalar ~(0, 1) per address, if similar to probe address; shape=[batch, n_addr]
    address_sim = address_similarity(at_addr, addresses).clip(0, 1)

    # scalar ~(0, 1) per batch, if that thing is reduced; shape=[batch]
    col1_ir = torch.sum(address_sim * ir1, dim=1).clip(0, 1)
    col2_ir = torch.sum(address_sim * ir2, dim=1).clip(0, 1)
    are_both_reduced = col1_ir * col2_ir  # Are Both Reduced?

    # Find everywhere this address is referenced
    ref_col1_sim = address_similarity(at_addr, col1).clip(0, 1)
    ref_col2_sim = address_similarity(at_addr, col2).clip(0, 1)

    ##########
    # Collect Case Probabilities: Base. App&Fn. Not App&Fn.
    #
    #   Determine which of the 3 cases should fire. All 3 fire everytime of
    #   course, to keep this pipeline differentiable. These probabilities scale
    #   the incorrect cases towards 0.

    # Base
    is_base_head = is_base_type(head_tag).clip(0, 1)

    # App-Fn
    col1_tag, = select_address(col1_addr, addresses, [tags])
    is_app_fn = (
        # head_tag, col1_tag:  ndarray([batch, vec_size])
        # app_tag_vec:  ndarray([vec_size])
        H.cosine_similarity(head_tag, app_tag_vec.unsqueeze(0), dim=1) * # head =? App
        H.cosine_similarity(col1_tag, fn_tag_vec.unsqueeze(0), dim=1)  # left term =? Fn
    ).clip(0, 1) # shape=[batch]

    # Which reduction step is right?
    red_base       = (is_base_head).clip(0, 1)
    red_app_fn     = (is_app_fn * are_both_reduced * (1 - is_base_head)).clip(0, 1)
    red_non_app_fn = ((1 - is_app_fn) * are_both_reduced * (1 - is_base_head)).clip(0, 1)

    ##########
    # BASE CASE: ie thing at `at_addr` is a base type
    #
    #  EG: Var, IntLit, Empty, ArithOp, TrueLit, FalseLit
    #
    #  NOTE: This case isn't needed if they're already marked as `is_reduced`
    #
    # If it's a base type, we should pop the stack and return the next address
    # on the stack

    base_should_push    = torch.zeros_like(col1_ir)
    base_should_pop     = (red_base).clip(0, 1)
    base_should_null_op = (1 - base_should_pop).clip(0, 1)

    # Update `is_reduced`
    #
    # We should tag references as being reduced or not. Terms can not become
    # unreduced, and we never need to interpolate between old ir and new ir, so
    # this step is merely additive (doesn't interpolate). Note, for this step we
    # update across ir1 and ir2 where the address references this term. So *not*
    # the col1_sim/col2_sim that are already in scope.
    ir1 = ir1 + ref_col1_sim * red_base
    ir2 = ir2 + ref_col2_sim * red_base

    ##########
    # APP-FN: reduce if term == App(Fn, _) and both col1 and col2 are reduced
    #         already
    #
    # If this rule fires, the new term is placed at the original address
    # `at_addr`.
    #
    # New values. Scale result of `reduce_app_fn` based on whether that rule
    # should have applied
    n_tags, n_col1, n_col2  = reduce_app_fn(at_addr, addresses, tags, col1, col2, gc=gc)
    tags = tags * (1 - red_app_fn) + red_app_fn * n_tags
    col1 = col1 * (1 - red_app_fn) + red_app_fn * n_col1
    col2 = col2 * (1 - red_app_fn) + red_app_fn * n_col2

    # If App-Fn fired, the Body moves to this current address, mark cols as NOT reduced
    ir1 = (ir1 - address_sim * red_app_fn).clip(0, 1)
    ir2 = (ir2 - address_sim * red_app_fn).clip(0, 1)

    ##########
    # NON APP-FN: reduce of term != App(Fn, _)
    #
    # Push addresses onto the stack if they're not base types (those without addresses)

    ##########
    # Col 1 and Col 2 are reduced, so reduce the whole term

    #####
    # Tag-type-dependent reduction
    #
    #   +, And, Or, Perceptron, etc.
    #
    #   (skip for now)
    #
    #   NOTE: This is where the magic of Neurallambdas may live. Pure lambda
    #   calculus uses 3 syntactic forms: var, lambda, app. This implementation
    #   of Neurallambdas obviously extends that notion because, who wants their
    #   ML alg to need to define Peano numbers, and all the funky combinators?
    #   So, it allows literal types like bools, ints, and arith ops.
    #
    #   One magical thing will be when we implement a Perceptron or FFNN
    #   literal. This will involve a matrix of weights that can be learned,
    #   maybe via backprop, and work seamlessly within this whole Neurallambda
    #   paradigm.

    # more_coolness = ???

    #####
    # Mark as reduced if both subterms are
    ir1 = ir1 + ref_col1_sim * red_non_app_fn
    ir2 = ir2 + ref_col2_sim * red_non_app_fn

    should_push_3    = torch.zeros_like(col1_ir)
    should_pop_3     = red_non_app_fn
    should_null_op_3 = 1 - red_non_app_fn

    ##########
    # Not Reduced Yet

    #####
    # Col 1 might need to be reduced
    should_push_1    = (1 - col1_ir) * (1 - is_base_head)  # it was already reduced, so don't push it
    should_pop_1     = torch.zeros_like(col1_ir) # no matter what, shouldn't indicate a pop here
    should_null_op_1 = 1 - should_push_1  # if col1_ir, then null_op on stack

    #####
    # Col 2 needs to be reduced
    should_push_2    = (1 - col2_ir) * (1 - is_base_head)  # it was already reduced, so don't push it
    should_pop_2     = torch.zeros_like(col2_ir) # no matter what, shouldn't indicate a pop here
    should_null_op_2 = 1 - should_push_2  # if col2_ir, then null_op on stack

    ##########
    # Stack stuff
    #
    # The above operations, which all happen in parallel each step, each have
    # different implications for what happens on the stack. Run those operations
    # now.

    # 1. If the current term is a base type, we will need to pop the stack.
    stack(base_should_push, base_should_pop, base_should_null_op, zero_vec_mat)

    # 2. If the current term is an unreduced reference, let's push it on the
    # stack to get dealt with.
    stack(should_push_1, should_pop_1, should_null_op_1, col1_addr)
    stack(should_push_2, should_pop_2, should_null_op_2, col2_addr)

    # 3. If this term has 2 references, we need to check if they both have been
    # reduced. This would happen if this term was pushed, then other stuff got
    # pushed, then eventually reduced. Then we finally pop this original thing,
    # and if it indeed has both references now reduced, we can pop it off the
    # stack.
    stack(should_push_3, should_pop_3, should_null_op_3, zero_vec_mat)

    ir1 = ir1.clip(0, 1)
    ir2 = ir2.clip(0, 1)

    return tags, col1, col2, ir1, ir2



##################################################
#

class Neurallambda:
    """


    TODO:

    - Init NL
    - Load memory (non differentiable step at least)

    - Initialize for Beta Reduction (IR, Stack)
    - Step Beta Reduction


    """

    def __init__(self, n_addresses, vec_size, n_stack, gc_steps, number_system):
        self.n_addresses = n_addresses
        self.vec_size = vec_size
        self.n_stack = n_stack
        self.gc_steps = gc_steps
        self.number_system = number_system


        #####
        # Projecting Ints
        #
        #   If we, linearly projected ints, they would be very susceptible to
        #   noise, and then project back to the wrong int. Instead, we assign
        #   each int its own random vec, which is uncorrelated with neighboring
        #   vecs, and then project-unproject is robust to noise.
        self.int_range_start = -200
        self.int_range_end   =  200

        # A matrix where each row ix represents the VEC_SIZE int
        int_vecs = torch.stack([N.randn((VEC_SIZE,)) for _ in range(self.int_range_start, self.int_range_end + 1)]).to(DEVICE)
        int_vecs_mat = N.to_mat(int_vecs)  # matrix form (ie of complex numbers)

        #####
        # Projecting Symbols

        # we can't cos_sim with zeroes, so here's a nice "Zero" vector
        zero_vec = torch.zeros((BATCH_SIZE, VEC_SIZE, N.dim), device=DEVICE) + 1e-1
        zero_vec_mat = N.to_mat(zero_vec)  # matrix form of complex numbers

        # a dense vector embedding for each tag
        tag_to_vec = {
            tag: N.randn((VEC_SIZE,)).to(DEVICE) if tag != 'NULL' else zero_vec[0]
            for tag in tag_names
        }

        tag_vecs = torch.stack([v for v in tag_to_vec.values()])
        tag_vecs_mat = N.to_mat(tag_vecs)

        app_tag_vec  = N.to_mat(tag_to_vec['App'])
        fn_tag_vec   = N.to_mat(tag_to_vec['Fn'])
        defn_tag_vec = N.to_mat(tag_to_vec['Defn'])

        # Base types
        var_tag_vec      = N.to_mat(tag_to_vec['Var'])
        intlit_tag_vec   = N.to_mat(tag_to_vec['IntLit'])
        empty_tag_vec    = N.to_mat(tag_to_vec['Empty'])
        arithop_tag_vec  = N.to_mat(tag_to_vec['ArithOp'])
        truelit_tag_vec  = N.to_mat(tag_to_vec['TrueLit'])
        falselit_tag_vec = N.to_mat(tag_to_vec['FalseLit'])
        base_types_vecs = torch.stack([
            var_tag_vec,
            intlit_tag_vec,
            empty_tag_vec,
            arithop_tag_vec,
            truelit_tag_vec,
            falselit_tag_vec,
        ])


    ##########
    # Projecting Ints
    #
    # We keep a different random projection for each int so they unproject neatly
    # (robust to noise) to the original int.

    def project_int(self, integer):
        """Projects an integer to a vector space."""
        index = integer - self.int_range_start
        return int_vecs[index]

    def unproject_int(self, vector):
        """Unprojects a vector from the vector space back to an integer.

        Assumes matrix formatted `vector`.
        """
        H.assert_is_probably_not_mat_form(vector)
        cs = H.cosine_similarity(N.to_mat(vector).unsqueeze(0), int_vecs_mat, dim=1)
        max_index = torch.argmax(cs).item()
        return max_index + self.int_range_start


    ##########
    # Projecting Symbols

    def vec_to_tag(self, vec):
        ''' return the most similar (cos_sim) tag index for a given vec.

        Expects `vec` in hypercomplex's matrix format.
        '''
        H.assert_is_probably_not_mat_form(vec)
        if vec.ndim == 1 + 1:  # +1 for hypercomplex
            sim = H.cosine_similarity(N.to_mat(vec), tag_vecs_mat, dim=1)
            return tag_names[sim.argmax().item()]
        elif vec.ndim == 2 + 1:  # +1 for hypercomplex
            out = []
            for v in vec:
                out.append(vec_to_tag(v))
            return out

    def vec_to_address(self, vec, addresses):
        ''' return the most similar (cos_sim) tag index for a given vec.

        Args:
          vec: ndarray([address_size])
          addresses: ndarray([batch?, n_addresses, address_size])
        '''
        H.assert_is_probably_not_mat_form(vec)
        H.assert_is_probably_not_mat_form(addresses)
        if addresses.ndim == 2 + 1:  # static addresses per all batches + hypercomplex mat
            sim = H.cosine_similarity(N.to_mat(vec), N.to_mat(addresses), dim=1)
            return sim.argmax().item()

        if addresses.ndim == 3 + 1:  # different addresses per batch + hypercomplex mat
            sim = H.cosine_similarity(N.to_mat(vec), N.to_mat(addresses), dim=2)
            return sim.argmax().item()



    def initialize_beta_reduction(self):
        pass

    def initialize(self):
        pass

    def initialize(self):
        pass

    def initialize(self):
        pass
