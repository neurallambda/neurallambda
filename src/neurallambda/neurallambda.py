'''

Actual Neurallambdas, ie the tensor form of Lambdacalc.

'''

from torch import einsum, tensor, allclose
from typing import Dict, Union, List, Tuple
from typing import Union, List, Any, Type
import neurallambda.language as L
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import neurallambda.memory as M
import neurallambda.stack as S
import neurallambda.symbol as Sym
from neurallambda.torch import cosine_similarity
import math


##################################################
# Permutation

def swap(x, swap1, swap2):
    ''' swap1 and swap2 are softmax vectors (think onehot) of rows of x that will
    be swapped. '''
    # Combine swap1 and swap2 into a single matrix
    P = torch.einsum('bx,by->bxy', swap1, swap2)
    P = P + P.transpose(1, 2)  # swap both directions
    # identity matrix to keep non-swapped data
    Id = torch.diag_embed(torch.ones_like(swap1) - swap1 - swap2)
    x_swapped = torch.einsum('bij,bjd->bid', P + Id, x)
    return x_swapped


def permute_one(xs, from_prob, to_prob):
    '''insert the item in a sequence at `from_prob` to `to_prob`.

    CAVEAT: if moving forward, the item will be placed *after* the element at
    to_prob, and if moving an item backward, it will be placed at the mentioned
    index, ie, before the thing that's currently at the to-index. This allows
    to move items from the very start of the list to the very end of the list,
    and vice versa.

    args:
      xs: [batch, sequence, embedding dim]
      from_prob, to_prob: [batch, sequence]

    '''
    B, S = xs.size(0), xs.size(1)

    fcs2 = torch.cumsum(to_prob, dim=1)
    fcs1 = ((fcs2 - 1) * -1)

    tcs2 = torch.cumsum(from_prob, dim=1)
    tcs1 = ((tcs2 - 1) * -1)

    # left block
    lb = torch.diag_embed(fcs1 * tcs1)

    # mid block
    mb = (
        # shift down for case where moving ix backward. If it's actually moving
        # ix forward, this should become all 0s.
        F.pad(torch.diag_embed(fcs2 * tcs1), [0, 0, 1, 0], value=0)[:, :-1] +

        # shift I up and to the right for case where moving ix forward. If it's
        # actually moving ix backward, this should become all 0s.
        F.pad(torch.diag_embed(F.pad(fcs1, [1, 0], value=0)[:, :S] *
                               F.pad(tcs2, [1, 0], value=0)[:, :S]), [0, 0, 0, 1])[:, 1:]
    )

    # value that's moved
    val = torch.einsum('bs, bt -> bst', to_prob, from_prob)

    # right block
    rb = torch.diag_embed(F.pad(tcs2, [1, 0], value=0)[:, :S] *
                          F.pad(fcs2, [1, 0], value=0)[:, :S])
    perm = (lb + mb + val + rb)

    return torch.einsum('bst, btd -> bsd', perm, xs)


# @@@@@@@@@@

# B = 2
# S = 10
# D = 3

# xs = torch.arange(B * S, dtype=torch.float).reshape(B, S).unsqueeze(2).repeat(1, 1, D)

# from_prob = torch.zeros((B, S)).float()
# from_prob[:, 8] = 0.1
# from_prob[:, 9] = 0.9

# to_prob = torch.zeros((B, S)).float()
# to_prob[:, 1] = 0.1
# to_prob[:, 2] = 0.9


# print()
# print('permute one')
# ys = permute_one(xs, from_prob, to_prob)
# print(ys)

# @@@@@@@@@@


##################################################
# Misc

def address_similarity(address, addresses):
    ''' Compare one address against many addresses, and return the scalar
    similarity of each address in `addresses` to the `address` (may be
    batched).'''
    if address.ndim == 1 and addresses.ndim == 2: # no batches
        cs = cosine_similarity(address, addresses, dim=1)
        return cs
    elif address.ndim == 2 and addresses.ndim == 3:  # each batch has own addresses
        assert address.size(0) == addresses.size(0), f'Batch size must be the same: address={address.shape} vs addresses={addresses.shape}'
        assert address.size(1) == addresses.size(2), f'Vector dimension must be the same: address={address.shape} vs addresses={addresses.shape}'

        br_address = address.unsqueeze(1).expand_as(addresses)
        cs = cosine_similarity(br_address, addresses, dim=2)
        return cs


##################################################
# Replace and KV-Insert

def replace(new_value, prev_value, tensr):
    ''' Replace any `prev_value` with `new_value`, according to cos_sim of
    prev_value with values in tensr'''
    assert new_value.shape == prev_value.shape
    assert tensr.size(0) == new_value.size(0) == prev_value.size(0), 'Inputs all need same batch size'
    assert tensr.size(-1) == new_value.size(-1) == prev_value.size(-1), 'Inputs all need same vector size'
    # new_value.shape  = [BATCH, DIM]
    # prev_value.shape = [BATCH, DIM]
    # tensr.shape      = [BATCH, N_ADDRESSES, DIM]

    # we'll interpolate into the `to` location with `sim_to`
    br_prev_value, br_tensr = torch.broadcast_tensors(prev_value.unsqueeze(1), tensr)
    sim_to = cosine_similarity(br_prev_value, br_tensr, dim=2)  # shape = [BATCH, N_ADDERSSES]
    keep = einsum('bnd, bn -> bnd', tensr, 1 - sim_to)
    rep = einsum('bn, bd -> bnd', sim_to, new_value)
    return  keep + rep


def kv_insert(state_k, state_v, k, x, eps=1e-8):
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

    if state_k.ndim == 2:  # not batched
        # the similarity of each rule key, to each state key
        alpha = cosine_similarity(k.unsqueeze(1), state_k.unsqueeze(0), dim=2) # [batch, address]
        if eps is not None:
            alpha = torch.where(alpha > eps, alpha, 0) # TODO: is this harmful to grads? Consider smooth Relus
        state_v = (
            einsum('ba, bav -> bav', 1 - alpha, state_v) +
            einsum('ba, bv -> bav', alpha    , x)
        )
        return state_v

    elif state_k.ndim == 3:  # batched
        assert state_k.ndim == state_v.ndim == 3, f'Expected state_k, and state_v to have same ndim: state_k={state_k.shape}, state_v={state_v.shape}'
        assert k.ndim == x.ndim == 2, f'Expected k and x to have same ndim: k={k.shape}, x={x.shape}'
        br_k, br_state_k = torch.broadcast_tensors(k.unsqueeze(1), state_k)
        alpha = cosine_similarity(br_state_k, br_k, dim=2 ) # the similarity of each rule key, to each state key
        if eps:
            alpha = torch.where(alpha > eps, alpha, 0) # TODO: is this harmful to grads? Consider smooth Relus

        if state_v.ndim == 3:
            state_v = (
                einsum('ba, bav -> bav', 1 - alpha, state_v) +
                einsum('ba, bv -> bav', alpha    , x)
            )
            return state_v

    raise ValueError(f'kv_insert called on state_k with unexpected shape: ndim={state_k.ndim}, shape={state_k.shape}')


##################################################
# Compile Memory to Tensors

##########

tag_names = [
    'UNRECOGNIZED',

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
    'NullLit',

    # Not Lambda, Not Base
    'Cons',
    'Car',
    'Cdr',
    # 'LinkedList',
    # 'ListP',

]

literal_tags = {'IntLit', 'TrueLit', 'FalseLit', 'NullLit', 'ArithOp', 'DefnName'}
nullary_tags = {'Empty'}
unary_tags   = {'Var', 'ArithOp', 'Car', 'Cdr'}
binary_tags  = {'App', 'Fn', 'Defn', 'Cons'}


##################################################
# NeuralLambdas


##########
# Builders

def build_empty_neurallambda(batch_size, n_addresses, vec_size, zero_vec_bias, device, addresses=None):
    if addresses is None:
        addresses = torch.randn((batch_size, n_addresses, vec_size)).to(device)
    tags  = torch.zeros((batch_size, n_addresses, vec_size), device=device) + zero_vec_bias
    col1 = torch.zeros((batch_size, n_addresses, vec_size), device=device) + zero_vec_bias
    col2 = torch.zeros((batch_size, n_addresses, vec_size), device=device) + zero_vec_bias
    return Neurallambda(addresses, tags, col1, col2, zero_vec_bias, device)


def build_neurallambdas(mem: Dict[M.Address, Any], batch_size, n_addresses, vec_size, zero_vec_bias, device, addresses=None):
    '''Given dictionary of memory `mem`, build a Neurallambda

    Args:
      mem:

      batch_size:

      n_addresses:

      vec_size:

      zero_vec_bias: if a memory location is empty, we'll define a vector of
        this value to indicate "zero". This is because if it were actually all
        zeros, cos_sim with it is undefined.

    '''
    nl = build_empty_neurallambda(batch_size, n_addresses, vec_size, zero_vec_bias, device, addresses)
    blocks = sorted(mem.items(), key=lambda item: item[0].i)
    for addr, block in blocks:
        nl.tags[:, addr.i] = nl.tag_to_vec[block[0]]

        # Column 1
        if len(block) >= 2 and isinstance(block[1], M.Address):
            term1_addr = block[1]
            nl.col1[:, addr.i] = nl.addresses[:, term1_addr.i]

        # Column 2
        if len(block) >= 3 and isinstance(block[2], M.Address):
            term2_addr = block[2]
            nl.col2[:, addr.i] = nl.addresses[:, term2_addr.i]

        # IntLit
        if len(block) >= 2 and block[0] == 'IntLit':
            val = block[1]
            nl.col1[:, addr.i] = nl.project_int(val)

        # Var
        if len(block) >= 2 and block[0] == 'Var':
            val = block[1]
            nl.col1[:, addr.i] = nl.project_int(Sym.char_to_int(val))

        # ArithOp
        if len(block) >= 2 and block[0] == 'ArithOp':
            val = block[1]
            nl.col1[:, addr.i] = nl.project_int(arithop_to_int[val])

    return nl


def string_to_neurallambda(s: str, batch_size, n_addresses, vec_size, zero_vec_bias, device):
    ''' This is probably mostly for demonstration and testing purposes. '''
    ast = L.string_to_terms(s)
    mem = M.terms_to_memory(ast)
    return build_neurallambdas(mem, batch_size, n_addresses, vec_size, zero_vec_bias, device)


##########
# Neurallambda Class

class Neurallambda:
    """A Neurallambda is a datastructure that supports a graph of tagged nodes made
    up of sum types and product types, IE an Abstract Syntax Tree. It is made
    out of tensors.

    So for instance, you can encode this program in a neurallambda:

    ((fn [x y z] '(x y z)) 1 2 3)

    And then restore it from the tensor representation all the way back to a
    pretty-printed string. More importantly, while it's a tensor, you can
    compute on it, such as doing the beta-reduction step of the Lambda Calculus.

    Using the Neuralbeta class, the program above can be reduced to:

    '(1 2 3)

    This list of numbers is fully encoded within tensors, but again, can be read
    back out into a human interpretable, pretty-printed program.


    On Symbols:

    How can a discrete symbol live within a tensor, you might ask? Research on
    Vector Symbolic Architectures (VSA), and Holographic Reduced Representations
    (HRR) informs this decision. In short, symbols can be jammed into vectors
    and it works great. Any 2 vectors drawn randomly are pseudo-orthogonal,
    meaning, they are very (very) likely to be nearly orthogonal. This allows us
    to use cosine similarity between 2 vectors to see if they represent the same
    symbol. Neat stuff happens in VSAs. For instance, an element-wise sum of 2
    symbols allows them to live in superposition, within one vector of the same
    size as the original vectors. This is lossy, but not too bad, and there are
    strategies for cleaning these up as well, to restore fidelity. Computations
    can also happen within these superposed states, and, well, VSAs may well be
    worth ones time to dig into more thoroughly elsewhere.

    Here, we barely use insights from VSAs to merely treat vectors as symbols,
    and where the encoded symbol can be computed upon, and reconstructed.

    """

    def __init__(self, addresses, tags, col1, col2, zero_vec_bias, device):
        self.addresses = addresses
        self.tags = tags
        self.col1 = col1
        self.col2 = col2

        self.batch_size, self.n_addresses, self.vec_size = addresses.shape

        self.device = device

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
        self.int_vecs = torch.stack([
            torch.randn((self.vec_size,))
            for _ in range(self.int_range_start, self.int_range_end + 1)
        ]).to(self.device)

        #####
        # Projecting Symbols

        # we can't cos_sim with zeroes, so here's a nice "Zero" vector
        # zero_vec = torch.zeros((BATCH_SIZE, vec_size), device=self.device) + 1e-1
        self.zero_vec = torch.zeros((self.vec_size), device=self.device) + zero_vec_bias

        # a dense vector embedding for each tag
        self.tag_to_vec = {
            tag: torch.randn((self.vec_size,)).to(self.device) if tag != 'UNRECOGNIZED' else self.zero_vec
            for tag in tag_names
        }

        self.tag_vecs = torch.stack([v for v in self.tag_to_vec.values()])
        self.app_tag_vec  = self.tag_to_vec['App']
        self.fn_tag_vec   = self.tag_to_vec['Fn']
        self.defn_tag_vec = self.tag_to_vec['Defn']
        self.cons_tag_vec = self.tag_to_vec['Cons']


        # Base types
        self.base_types_vecs = torch.stack([
            self.tag_to_vec['Var'],
            self.tag_to_vec['IntLit'],
            self.tag_to_vec['Empty'],
            self.tag_to_vec['ArithOp'],
            self.tag_to_vec['TrueLit'],
            self.tag_to_vec['FalseLit'],
            self.tag_to_vec['NullLit'],
        ])

    ##########
    # Projecting Ints
    #
    # We keep a different random projection for each int so they unproject neatly
    # (robust to noise) to the original int.

    def project_int(self, integer):
        """Projects an integer to a vector space."""
        index = integer - self.int_range_start
        return self.int_vecs[index]

    def unproject_int(self, vector):
        """Unprojects a vector from the vector space back to an integer.

        Assumes matrix formatted `vector`.
        """
        assert vector.ndim == 1, f'Expecting vectors, but got: {vector.shape}'
        bvector = vector.unsqueeze(0).expand_as(self.int_vecs)
        cs = cosine_similarity(bvector, self.int_vecs, dim=1)
        max_index = torch.argmax(cs).item()
        return max_index + self.int_range_start

    ##########
    # Projecting Symbols

    def vec_to_tag(self, vec):
        ''' return the most similar (cos_sim) tag index for a given vec. '''
        assert vec.ndim == 1, f'Expecting vectors, but got: {vec.shape}'
        bvec = vec.unsqueeze(0).expand_as(self.tag_vecs)
        sim = cosine_similarity(bvec, self.tag_vecs, dim=1)
        return tag_names[sim.argmax().item()]

    def vec_to_address(self, vec, addresses):
        ''' return the most similar (cos_sim) tag index for a given vec.

        Args:
          vec: ndarray([vec_size])
          addresses: ndarray([batch?, n_addresses, vec_size])
        '''
        assert vec.size(-1) == addresses.size(-1), f'Inputs must have same vec size: vec={vec.shape}, addresses={addresses.shape}'

        if addresses.ndim == 2:  # static addresses per all batches
            br_vec, br_addresses = torch.broadcast_tensors(vec, addresses)
            sim = cosine_similarity(br_vec, br_addresses, dim=1)
            return sim.argmax().item()

        raise ValueError(f'vec_to_address called on vec with unexpected shape: ndim={vec.ndim}, shape={vec.shape}')


    def is_base_type(self, x):
        ''' A Base type is that without any Addresses within it, eg Var, IntLit,
        Empty, etc. First dim of `x` is `batch_dim`.

        Args:
          x: shape=[BATCH, VEC_SIZE]

        Warning: values can exceed (-1, 1)
        '''
        assert x.size(1) == self.base_types_vecs.size(1), f'Expected same vec_size: x={x.shape}, base_types_vecs={base_types_vecs.shape}'
        br_x, br_base_types_vecs = torch.broadcast_tensors(x.unsqueeze(1), self.base_types_vecs.unsqueeze(0))
        sims = cosine_similarity(br_x, br_base_types_vecs, dim=2).clip(0, 1).sum(dim=1)
        return sims


##########
# Decompile Neural Lambdas back to Mem

def read_col(nl, tag, vec, addresses):
    ''' Project a neurallambda `vec` back to the machine language. A `tag` determines how it should be read. '''

    if tag == 'IntLit':
        return nl.unproject_int(vec)
    elif tag == 'Var':
        return Sym.int_to_char(nl.unproject_int(vec))
    elif tag == 'ArithOp':
        return Sym.int_to_arithop[nl.unproject_int(vec)]
    elif tag in {'TrueLit', 'FalseLit'}:
        return zero_vec
    elif tag == 'Empty':
        return zero_vec

    # UNRECOGNIZED
    z = nl.zero_vec if nl.zero_vec.ndim == 1 else nl.zero_vec[0] # single batch
    c = cosine_similarity(vec, z, dim=0)
    if c  > 0.5:
        return ('UNRECOGNIZED', )

    # Address
    return M.Address(nl.vec_to_address(vec, addresses))

def neurallambda_to_mem(nl, addresses, tags, col1, col2, n_ixs) -> Dict[M.Address, Any]:
    ''' Reverse engineer the tensor set of a neurallambda back into a dictionary mem.

    TODO: it's such laziness to pass nl and its weights separately
    '''

    if tags.ndim == 2:  # non-batched
        recon_mem = {}
        for i in range(n_ixs):
            ai = M.Address(i)
            t = nl.vec_to_tag(tags[i])

            if t == 'UNRECOGNIZED':
                recon_mem[ai] = (t,)
                continue

            # NULLARY
            if t in nullary_tags:
                recon_mem[ai] = (t,)

            # UNARY or LITERAL
            if t in literal_tags or t in unary_tags:
                v = read_col(nl, t, col1[i], addresses)
                recon_mem[ai] = (t, v)

            # BINARY
            if t in binary_tags:
                v1 = read_col(nl, t, col1[i], addresses)
                v2 = read_col(nl, t, col2[i], addresses)
                recon_mem[ai] = (t, v1, v2)
        return recon_mem

    elif tags.ndim == 3:  # batched
        out = []

        if addresses.ndim == 2:  # static addresses per all batches
            for t, c1, c2 in zip(tags, col1, col2):
                out.append(neurallambda_to_mem(nl, addresses, t, c1, c2, n_ixs))

        elif addresses.ndim == 3:  # different addresses per batch
            for a, t, c1, c2 in zip(addresses, tags, col1, col2):
                out.append(neurallambda_to_mem(nl, a, t, c1, c2, n_ixs))

        return out
    raise ValueError(f"Saw neurallambda's tags with unexpected shape: {tags.shape}")


##################################################
# Neuralbeta: A Neurallambda equipped with beta reduction stuff

'''.

BRAINSTORMING how to reduce Neurallambdas?:

- [X] Online traversal issue: If you see a base type, pop the stack. The next
  address likely references that base type, so, push it's location back onto the
  stack, and infinite loop.

- [X] solution? include two address-aligned "is_reduced" tensors for col1 and
  col2. It doesn't mark a term, it marks references to terms, as to whether
  they're reduced.

- 2 pass. First traverse tree, push everything into a stack. Second, pop from
  stack only, and reduce as you go. I think this wouldn't support online
  recursion.

- solution? instead of 2 new tensors, how about superposing an address with
  "is_reduced" or "is_not_reduced"?  How to update that value? add the negative,
  scaled by cossim? Look out for addresses that have been noised, because the
  original superposed value will have drifted. (NOTE: Tried, too noisy/lossy)


BRAINSTORMING how to tag `is_(not_)reduced`:

How to store `is_reduced`:

    - [X] Separate tensors for noting which addresses are reduced

        - or one index on the address tensor devoted to `is_(not_)reduced`

    - Superposition of address and an `is_(not_)reduced` tag. In experiments,
      the superposed vecs got too noisy too fast. With Error Correction, this
      idea might be worth revisiting. The generalized version of it says that
      you could congeal any info into one superposed vector, and that could be
      useful.

    - 2 orthogonal sets of address ints. EG replace A(1) with orthogonal vec
      IsReducedA(1)

    - Separate cones of vector space: `is_reduced` could be same address * -1

    - Rotate points above / below a hyperplane (requires ensuring that all
      integers are above some hyperplane

    - Store "is_reduced" in magnitude of vectors


When / how to process `is_reduced`

    - when non-app-fn reduction happens, peek ahead to see if they're reduced?

    - [X] when we've descended to a base type, we know it `is_reduced`, and we
      update every reference to this address simultaneously '''



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
        # x = H.scale(values, cs.unsqueeze(-1))  # elem-wise multiplication
        x = values * cs.unsqueeze(-1)  # elem-wise multiplication
        x = x.sum(dim=1)  # collapse all addresses into one
        out.append(x)
    return out


def reduce_app_fn(at_addr, nl, gc_steps:int):
    '''Reduce ((fn [x] x) y)  --to-->  y

    This assumes you know that the tag at `at_addr` is App, and the
    referred term in `col1` resolves to a `Fn`.

    Var addresses are never locally scoped, only globally. So when a reduction
    happens with a var at address=`a`, all references to `a` throughout the
    enitre memory are replaced with the substitute.

    Args:
      at_addr: ndarray([batch, vec_size])
      gc_steps: how many steps of garbage collection?

    TODO: OPTIM address lookup cos-sim stuff could be reused across kv_inserts
    '''

    addresses = nl.addresses
    tags = nl.tags
    col1 = nl.col1
    col2 = nl.col2
    zero_vec = nl.zero_vec.unsqueeze(0)

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

    for _ in range(gc_steps):
        # Erase Fn
        tags = kv_insert(addresses, tags, fn_addr, zero_vec)
        col1 = kv_insert(addresses, col1, fn_addr, zero_vec)
        col2 = kv_insert(addresses, col2, fn_addr, zero_vec)

        # Erase Fn's Bound Var
        tags = kv_insert(addresses, tags, param_addr, zero_vec)
        col1 = kv_insert(addresses, col1, param_addr, zero_vec)
        col2 = kv_insert(addresses, col2, param_addr, zero_vec)

        # Erase original Fn Body
        tags = kv_insert(addresses, tags, body_addr, zero_vec)
        col1 = kv_insert(addresses, col1, body_addr, zero_vec)
        col2 = kv_insert(addresses, col2, body_addr, zero_vec)
    return tags, col1, col2


def reduce_step(
        at_addr,
        nl,

        # Neuralbeta
        sharpen_pointer,
        ir1,
        ir2,
        ss: S.StackState,
        gc_steps:int
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

    # convenient rebind
    addresses = nl.addresses
    tags = nl.tags
    col1 = nl.col1
    col2 = nl.col2
    zero_vec = nl.zero_vec.unsqueeze(0)
    # batch_size = addresses.shape[0]
    # zero_vec = nl.zero_vec.unsqueeze(0).expand(batch_size, -1, -1, -1)



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
    is_base_head = nl.is_base_type(head_tag).clip(0, 1)

    # App-Fn
    col1_tag, = select_address(col1_addr, addresses, [tags])
    is_app_fn = (
        # head_tag, col1_tag:  ndarray([batch, vec_size])
        # app_tag_vec:  ndarray([vec_size])
        cosine_similarity(head_tag, nl.app_tag_vec.unsqueeze(0), dim=1) * # head =? App
        cosine_similarity(col1_tag, nl.fn_tag_vec.unsqueeze(0), dim=1)  # left term =? Fn
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
    n_tags, n_col1, n_col2  = reduce_app_fn(at_addr, nl, gc_steps=gc_steps)
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
    nss1, _ = S.push_pop_nop(ss, sharpen_pointer, base_should_push, base_should_pop, base_should_null_op, zero_vec)

    # 2. If the current term is an unreduced reference, let's push it on the
    # stack to get dealt with.
    nss2, _ = S.push_pop_nop(nss1, sharpen_pointer, should_push_1, should_pop_1, should_null_op_1, col1_addr)
    nss3, _ = S.push_pop_nop(nss2, sharpen_pointer, should_push_2, should_pop_2, should_null_op_2, col2_addr)

    # 3. If this term has 2 references, we need to check if they both have been
    # reduced. This would happen if this term was pushed, then other stuff got
    # pushed, then eventually reduced. Then we finally pop this original thing,
    # and if it indeed has both references now reduced, we can pop it off the
    # stack.
    nss4, _ = S.push_pop_nop(nss3, sharpen_pointer, should_push_3, should_pop_3, should_null_op_3, zero_vec)

    ir1 = ir1.clip(0, 1)
    ir2 = ir2.clip(0, 1)

    return tags, col1, col2, ir1, ir2, nss4


class Neuralbeta(nn.Module):
    def __init__(self, nl, n_stack, initial_sharpen_pointer):
        super(Neuralbeta, self).__init__()
        STACK_INITIAL_SHARPEN = 100
        STACK_ZERO_OFFSET = 1e-3

        self.n_stack = n_stack
        self.nl = nl

        # `is_reduced` tracks a bool at each address for columns 1 and 2 of
        # whether or not they're reduced already
        self.ir1 = torch.zeros((nl.batch_size, nl.n_addresses)).to(nl.device)
        self.ir2 = torch.zeros((nl.batch_size, nl.n_addresses)).to(nl.device)

        # Stack State
        self.ss = S.initialize(nl.vec_size, n_stack, nl.batch_size, STACK_ZERO_OFFSET, nl.device)
        self.sharpen_pointer = nn.Parameter(torch.tensor([initial_sharpen_pointer], dtype=torch.float))

    def push_address(self, address):
        ''' Push an address onto the stack. '''
        self.ss = S.push(self.ss, address)

    def select_address(self, address, list_of_values):
        ''' Get values out of columns (see docstring of top-level `select_address`. '''
        return select_address(address, self.nl.addresses, list_of_values)

    def reduce_step(self, at_addr, gc_steps:int):
        tags, col1, col2, ir1, ir2, nss = reduce_step(
            at_addr,
            self.nl,

            # Neuralbeta
            self.sharpen_pointer,
            self.ir1,
            self.ir2,
            self.ss,
            gc_steps,
        )
        self.nl.tags = tags
        self.nl.col1 = col1
        self.nl.col2 = col2
        self.ir1 = ir1
        self.ir2 = ir2
        self.ss = nss
        return None # this function mutates class's state


##################################################
# CosineSimilarity

# class Weight(nn.Module):
#     def __init__(self, input_features, output_features, init_method='kaiming'):
#         super(Weight, self).__init__()
#         self.weight = nn.Parameter(torch.empty(input_features, output_features))
#         self.init_method = init_method
#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.init_method == 'kaiming':
#             nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         elif self.init_method == 'xavier':
#             nn.init.xavier_uniform_(self.weight)
#         elif self.init_method == 'orthogonal':
#             nn.init.orthogonal_(self.weight)
#         else:
#             raise ValueError(f"Invalid initialization method: {self.init_method}")

#     def forward(self, input):
#         raise Exception("You must not call Weight's forward function ever.")

# class CosineSimilarity(nn.Module):
#     def __init__(self, weight, dim, unsqueeze_inputs=[], unsqueeze_weights=[]):
#         super(CosineSimilarity, self).__init__()
#         self.weight = weight.weight
#         self.dim = dim
#         self.unsqueeze_inputs = unsqueeze_inputs
#         self.unsqueeze_weights = unsqueeze_weights

#     def forward(self, input):
#         for ix in self.unsqueeze_inputs:
#             input = input.unsqueeze(ix)

#         weight = self.weight
#         for ix in self.unsqueeze_weights:
#             weight = weight.unsqueeze(ix)

#         return torch.cosine_similarity(input, weight, dim=self.dim)


#     def diagnose(self, input):
#         '''An interpreter-time helper to help devs get shapes right.'''
#         print()

#         # Normal weights
#         print('Input:', input.shape)
#         print('Weight:', self.weight.shape)
#         print()

#         # Unsquoze Input
#         uinput = torch.empty_like(input)
#         for ix in self.unsqueeze_inputs:
#             uinput = uinput.unsqueeze(ix)
#         print('Unsquoze Input :', uinput.shape)

#         # Unsquoze Weights
#         uweight = torch.empty_like(self.weight)
#         for ix in self.unsqueeze_weights:
#             uweight = uweight.unsqueeze(ix)
#         print('Unsquoze Weight:', uweight.shape)
#         print()

#         # Broadcasted
#         binput, bweight = torch.broadcast_tensors(uinput, uweight)
#         print('Broadcasted Weight:', binput.shape)
#         print('Broadcasted Exp Input :', bweight.shape)
#         print()

#         # Ouptut
#         out = self(input)
#         print('Out:', out.shape)


# @@@@@@@@@@
# Sandbox to align sizes

# batch_size = 7
# vec_size = 1024
# n_symbols = 13

# CosineSimilarity(Weight(vec_size, n_symbols),
#                  dim=1,
#                  unsqueeze_inputs=[2],
#                  unsqueeze_weights=[0]).diagnose(torch.zeros(batch_size, vec_size))
# print('----------')
# CosineSimilarity(Weight(vec_size, n_symbols),
#                  dim=2,
#                  unsqueeze_inputs=[-1],
#                  unsqueeze_weights=[0, 0]).diagnose(torch.zeros(batch_size, 41, vec_size))

# if False:
#     batch_size = 67
#     vec_size = 1024
#     n_symbols = 13
#     n_inp_vecs = 3
#     cs = CosineSimilarity(
#         Weight(vec_size, n_symbols),
#         dim=2,
#         unsqueeze_inputs=[-1],
#         unsqueeze_weights=[0, 0])
#     inp = torch.zeros(batch_size, n_inp_vecs, vec_size)
#     cs.diagnose(inp)
#     out = cs(inp)
#     assert out.shape == torch.Size([batch_size, n_inp_vecs, n_symbols])


# @@@@@@@@@@


class ReverseCosineSimilarity(nn.Module):
    '''When you project inputs forward, you can learn how similar each was to a
    known "symbol" within the `weight` tensor. This similarity is in [-1, 1]. If
    you project that scalar backwards, you can recover the original vector. If
    you project back something close to 0, it won't be too similar to anything
    in the known-symbols tensor.

    '''
    def __init__(self, cs):
        super(ReverseCosineSimilarity, self).__init__()
        self.cs = cs

    def forward(self, input):
        for ix in self.cs.unsqueeze_inputs:
            input = input.unsqueeze(ix)

        weight = self.cs.weight.t()
        for ix in self.cs.unsqueeze_weights:
            weight = weight.unsqueeze(ix)

        input, weight = torch.broadcast_tensors(input, weight)

        # breakpoint()
        return torch.sum(
            input * weight,
            dim=self.cs.dim
        )
