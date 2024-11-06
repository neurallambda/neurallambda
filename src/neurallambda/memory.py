'''.

The Dictionary-Memory form of a lambda calculus expression.

This is an intermediate form, not yet a neurallambda, where a lambda calc
program has been converted into a dictionary with integer key addresses, and
then each value is a `Block`, which is simply a tuple that represents the `Term`
at this address. The tuple `Block` form represents "tags" marking the type of
the `Term`. Then, most `Term`s also have `Terms` beneath them, and these are
held as references to other addresses in the memory dict.


An Expression is an ADT of syntactic pieces.

A Neurallambda is a tensor-based representation of that.

In order to convert from an ADT-like representation to a Neurallambda it is
convenient to go through a dictionary form which represents each term at
"addresses in memory". These addreses are just ints in the dictionary key
which will eventually refer to a row in the Neurallambda tensors.

KEYS: Addresses, which hold ints, which correspond directly to the index in
      the eventual neurallambda

VALUES: A 1-, 2-, or 3-ary tuple called `Block` (see definition for
        description.

'''

from dataclasses import dataclass
from typing import Dict, Union, List, Tuple, Union, List, Any, Type
from neurallambda.language import *
import neurallambda.debug as D


####################
# Convert Expression -> Dictionary Memory

@dataclass(frozen=True)
class Address:
    i: int
    def __str__(self):
        return f'A({self.i})'

A = Address  # shorthand

# A `Block` includes a type tag, and 0, 1, or 2 args.
Block = Union[
    Tuple[str,], # for {TrueLit, FalseLit}
    Tuple[str, Any], # For terms like Var('x') and IntLit(42)
    Tuple[str, Address],  # single address container
    Tuple[str, Address, Address],  # two address container
]

def terms_to_memory(terms: List[Term]) -> Dict[Address, Block]:
    memory = {}
    var_addresses = {}
    for term in terms:
        _, _, memory = process_term(term, 0, memory, var_addresses)
    return memory


def process_var(term: Var,
               next_address: int,
               memory: Dict[Address, Block],
               var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    addr = var_addresses.get(term.name, Address(next_address))

    # this var hasn't been seen
    if term.name not in var_addresses:
        memory[addr] = ('Var', term.name)
        var_addresses[term.name] = addr
        next_address += 1

    # this var has been seen
    else:
        pass
    return next_address, addr, memory


def curry_fn(fn_term: Fn) -> Fn:
    """
    Curry an Fn term with multiple parameters into nested unary Fn terms.
    Example: Fn(params=['x', 'y'], body=...) becomes Fn(params=['x'], body=Fn(params=['y'], body=...))
    """
    if len(fn_term.params) <= 1:
        # No currying needed for 0 or 1 parameter
        return fn_term

    # Curry the function by nesting Fn terms
    first_param, *remaining_params = fn_term.params
    nested_fn = Fn(params=remaining_params, body=fn_term.body)
    curried_fn = Fn(params=[first_param], body=curry_fn(nested_fn))
    return curried_fn


def process_fn(fn_term: Fn,
               next_address: int,
               memory: Dict[Address, Block],
               var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    curried_fn = curry_fn(fn_term)
    fn_addr = Address(next_address)
    next_address += 1

    if len(curried_fn.params) == 0:
        next_address, body_addr, memory = process_term(curried_fn.body, next_address, memory, var_addresses)
        memory[fn_addr] = ('Fn', None, body_addr)
    else:
        param_name = curried_fn.params[0]
        next_address, var_addr, memory = process_var(Var(param_name), next_address, memory, var_addresses)
        next_address, body_addr, memory = process_term(curried_fn.body, next_address, memory, var_addresses)
        memory[fn_addr] = ('Fn', var_addr, body_addr)

    return next_address, fn_addr, memory


def process_defn(defn: Defn, next_address: int, memory: Dict[Address, Block], var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    defn_addr = Address(next_address)
    next_address += 1

    defn_name_addr = Address(next_address)
    memory[defn_name_addr] = ('DefnName', defn.name)
    next_address += 1

    fn_term = Fn(params=defn.params, body=defn.body)
    next_address, fn_addr, memory = process_fn(fn_term, next_address, memory, var_addresses)

    memory[defn_addr] = ('Defn', defn_name_addr, fn_addr)

    return next_address, defn_addr, memory


def desugar_app(app_term: App) -> App:
    """
    Desugar an App term with more than two arguments into a left-associative chain of App terms.
    Example: App([a, b, c, d]) becomes App([App([App([a, b]), c]), d])
    """
    terms = app_term.terms
    if len(terms) <= 2:
        # No desugaring needed for 0, 1, or 2 terms
        return app_term

    # Start the left-associative chain with the first two terms
    desugared = App([terms[0], terms[1]])
    for term in terms[2:]:
        desugared = App([desugared, term])
    return desugared


def process_app(app_term: App, next_address: int, memory: Dict[Address, Block], var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    desugared_app = desugar_app(app_term)
    terms = desugared_app.terms

    if len(terms) == 1:
        # Single term, no actual application needed
        app_addr = Address(next_address)
        next_address += 1
        next_address, first_addr, memory = process_term(terms[0], next_address, memory, var_addresses)
        memory[app_addr] = ('App', first_addr)
        return next_address, app_addr, memory

    # Two terms, create a single App block
    app_addr = Address(next_address)
    next_address += 1
    next_address, first_addr, memory = process_term(terms[0], next_address, memory, var_addresses)
    next_address, second_addr, memory = process_term(terms[1], next_address, memory, var_addresses)
    memory[app_addr] = ('App', first_addr, second_addr)
    return next_address, app_addr, memory


def process_int_lit(int_lit: IntLit, next_address: int, memory: Dict[Address, Block]) -> (int, Address, Dict[Address, Block]):
    int_lit_addr = Address(next_address)
    next_address += 1
    memory[int_lit_addr] = ('IntLit', int_lit.value)
    return next_address, int_lit_addr, memory


def process_arith_op(arith_op: ArithOp, next_address: int, memory: Dict[Address, Block]) -> (int, Address, Dict[Address, Block]):
    arith_op_addr = Address(next_address)
    next_address += 1
    memory[arith_op_addr] = ('ArithOp', arith_op.operator)
    return next_address, arith_op_addr, memory


# List

def process_empty(int_lit: IntLit, next_address: int, memory: Dict[Address, Block]) -> (int, Address, Dict[Address, Block]):
    empty_addr = Address(next_address)
    next_address += 1
    memory[empty_addr] = ('Empty',)
    return next_address, empty_addr, memory


def process_cons(cons: Cons, next_address: int, memory: Dict[Address, Block], var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    cons_addr = Address(next_address)
    next_address += 1

    next_address, head_addr, memory = process_term(cons.head, next_address, memory, var_addresses)
    next_address, tail_addr, memory = process_term(cons.tail, next_address, memory, var_addresses)

    memory[cons_addr] = ('Cons', head_addr, tail_addr)
    return next_address, cons_addr, memory


def process_car(car: Car, next_address: int, memory: Dict[Address, Block], var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    car_addr = Address(next_address)
    next_address += 1

    next_address, list_addr, memory = process_term(car.list, next_address, memory, var_addresses)
    memory[car_addr] = ('Car', list_addr)
    return next_address, car_addr, memory


def process_cdr(cdr: Cdr, next_address: int, memory: Dict[Address, Block], var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    cdr_addr = Address(next_address)
    next_address += 1

    next_address, list_addr, memory = process_term(cdr.list, next_address, memory, var_addresses)
    memory[cdr_addr] = ('Cdr', list_addr)
    return next_address, cdr_addr, memory


def process_list_p(list_p: ListP, next_address: int, memory: Dict[Address, Block], var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    list_p_addr = Address(next_address)
    next_address += 1

    next_address, term_addr, memory = process_term(list_p.term, next_address, memory, var_addresses)
    memory[list_p_addr] = ('ListP', term_addr)
    return next_address, list_p_addr, memory


# Bool

def process_true_lit(true_lit: TrueLit, next_address: int, memory: Dict[Address, Block]) -> (int, Address, Dict[Address, Block]):
    true_lit_addr = Address(next_address)
    next_address += 1
    memory[true_lit_addr] = ('TrueLit',)
    return next_address, true_lit_addr, memory


def process_false_lit(false_lit: FalseLit, next_address: int, memory: Dict[Address, Block]) -> (int, Address, Dict[Address, Block]):
    false_lit_addr = Address(next_address)
    next_address += 1
    memory[false_lit_addr] = ('FalseLit',)
    return next_address, false_lit_addr, memory


def process_null_lit(null_lit: NullLit, next_address: int, memory: Dict[Address, Block]) -> (int, Address, Dict[Address, Block]):
    null_lit_addr = Address(next_address)
    next_address += 1
    memory[null_lit_addr] = ('NullLit',)
    return next_address, null_lit_addr, memory


def process_term(term: Term, next_address: int, memory: Dict[Address, Block], var_addresses: Dict[str, Address]) -> (int, Address, Dict[Address, Block]):
    if isinstance(term, Var):
        return process_var(term, next_address, memory, var_addresses)
    elif isinstance(term, Fn):
        return process_fn(term, next_address, memory, var_addresses)
    elif isinstance(term, Defn):
        return process_defn(term, next_address, memory, var_addresses)
    elif isinstance(term, App):
        return process_app(term, next_address, memory, var_addresses)

    # list
    elif isinstance(term, Empty):
        return process_empty(term, next_address, memory)
    elif isinstance(term, Cons):
        return process_cons(term, next_address, memory, var_addresses)
    elif isinstance(term, Car):
        return process_car(term, next_address, memory, var_addresses)
    elif isinstance(term, Cdr):
        return process_cdr(term, next_address, memory, var_addresses)
    elif isinstance(term, ListP):
        return process_list_p(term, next_address, memory, var_addresses)

    # math
    elif isinstance(term, IntLit):
        return process_int_lit(term, next_address, memory)
    elif isinstance(term, ArithOp):
        return process_arith_op(term, next_address, memory)

    # bool
    elif isinstance(term, TrueLit):
        return process_true_lit(term, next_address, memory)
    elif isinstance(term, FalseLit):
        return process_false_lit(term, next_address, memory)
    elif isinstance(term, NullLit):
        return process_null_lit(term, next_address, memory)

    else:
        raise NotImplementedError(f"Term type {type(term)} not implemented in process_term")


####################
#  Convert Dictionary `memory_to_terms` -> Expression

##########
# Lambda

def reconstruct_var(block: Block) -> Var:
    match block:
        case ('Var', name):
            return Var(name)
        case _:
            return Error(f'error reconstructing Var: {block}')

def reconstruct_fn(memory: Dict[Address, Block], address: Address,
                   resugar_app=True,
                   resugar_fn=True) -> Fn:
    if address not in memory:
        return Error(f'error reconstructing Fn, address={address} not found in memory')

    match memory[address]:
        case ('Fn', param_addr, body_addr):
            # param
            if param_addr is None: # nullary fn
                params = []
            else:
                if param_addr not in memory:
                    params = [Error(f'error reconstructing Fn, param_addr={param_addr}, not found in memory')]
                else:
                    param  = reconstruct_var(memory[param_addr])
                    match param:
                        case Var():
                            params = [param.name]
                        case _:
                            params = [Error(f'error reconstructing Fn, param={param} is not a Var')]

            # body
            body = memory_to_terms(memory, body_addr, resugar_app, resugar_fn)

            # Check if body is a nested function (currying) and combine parameters/resugar to a single multi-param function
            if isinstance(body, Fn) and resugar_fn:
                return Fn(params=params + body.params, body=body.body)
            else:
                return Fn(params=params, body=body)
        case _:
            return Error(f'error constructing Fn at address={address}')

def reconstruct_defn(memory: Dict[Address, Block], address: Address,
                     resugar_app=True,
                     resugar_fn=True
                     ) -> Defn:
    _, name_addr, fn_addr = memory[address]
    name_block = memory[name_addr]
    _, name = name_block
    fn = memory_to_terms(memory, fn_addr, resugar_app, resugar_fn)
    if not isinstance(fn, Fn):
        raise ValueError(f"Expected Fn term at {fn_addr}, found {type(fn)}")
    return Defn(name, fn.params, fn.body)

def collect_terms(app_term: App) -> List[Term]:
    # Recursively collect terms from nested App structures for help resugaring
    if isinstance(app_term.terms[0], App):
        return collect_terms(app_term.terms[0]) + [app_term.terms[1]]
    else:
        return app_term.terms


def reconstruct_app(memory: Dict[Address, Block], address: Address,
                    resugar_app=True,
                    resugar_fn=True
                    ) -> App:
    block = memory[address]
    func_addr = block[1]

    # Check if there is an argument address
    arg_addr = block[2] if len(block) > 2 else None

    func = memory_to_terms(memory, func_addr, resugar_app, resugar_fn)
    arg = memory_to_terms(memory, arg_addr, resugar_app, resugar_fn) if arg_addr is not None else None

    # Resugar app
    if isinstance(func, App) and resugar_app:
        terms = collect_terms(App([func, arg])) if arg else collect_terms(func)
    else:
        terms = [func] if arg is None else [func, arg]

    return App(terms)


##########
# Lists

def reconstruct_cons(memory: Dict[Address, Block], address: Address,
                     resugar_app=True,
                     resugar_fn=True
                     ) -> Cons:
    _, head_addr, tail_addr = memory[address]
    head = memory_to_terms(memory, head_addr, resugar_app, resugar_fn)
    tail = memory_to_terms(memory, tail_addr, resugar_app, resugar_fn)
    return Cons(head, tail)

def reconstruct_car(memory: Dict[Address, Block], address: Address,
                    resugar_app=True,
                    resugar_fn=True
                    ) -> Car:
    _, list_addr = memory[address]
    list_term = memory_to_terms(memory, list_addr, resugar_app, resugar_fn)
    return Car(list_term)

def reconstruct_cdr(memory: Dict[Address, Block], address: Address,
                    resugar_app=True,
                    resugar_fn=True
                    ) -> Cdr:
    _, list_addr = memory[address]
    list_term = memory_to_terms(memory, list_addr, resugar_app, resugar_fn)
    return Cdr(list_term)

def reconstruct_list_p(memory: Dict[Address, Block], address: Address,
                       resugar_app=True,
                       resugar_fn=True
                       ) -> ListP:
    _, term_addr = memory[address]
    term = memory_to_terms(memory, term_addr, resugar_app, resugar_fn)
    return ListP(term)


##########
# Math

def reconstruct_int_lit(memory: Dict[Address, Block], address: Address) -> IntLit:
    _, num = memory[address]
    return IntLit(num)


def reconstruct_arith_op(memory: Dict[Address, Block], address: Address) -> ArithOp:
    _, symbol = memory[address]
    return ArithOp(symbol)


def memory_to_terms(memory: Dict[Address, Block], address: Address,
                    resugar_app=True,
                    resugar_fn=True
                    ) -> Term:
    ''' Convert a dictionary representation of the memory into an ADT of terms. '''
    if address in memory:
        block = memory[address]
    else:
        return Error(f'Lookup Error: {address} not in {memory}')

    block_type = block[0]

    if block_type == 'Var':
        try: return reconstruct_var(block);
        except ValueError: return 'Var <error>'

    elif block_type == 'Fn':
        try: return reconstruct_fn(memory, address, resugar_app, resugar_fn)
        except ValueError: return 'Fn <error>'

    elif block_type == 'Defn':
        return reconstruct_defn(memory, address, resugar_app, resugar_fn)
    elif block_type == 'App':
        return reconstruct_app(memory, address, resugar_app, resugar_fn)

    # list
    elif block_type == 'Empty':
        return Empty()
    elif block_type == 'Cons':
        return reconstruct_cons(memory, address, resugar_app, resugar_fn)
    elif block_type == 'Car':
        return reconstruct_car(memory, address, resugar_app, resugar_fn)
    elif block_type == 'Cdr':
        return reconstruct_cdr(memory, address, resugar_app, resugar_fn)
    elif block_type == 'ListP':
        return reconstruct_list_p(memory, address, resugar_app, resugar_fn)

    # ArithOp
    elif block_type == 'IntLit':
        return reconstruct_int_lit(memory, address)

    elif block_type == 'ArithOp':
        return reconstruct_arith_op(memory, address)

    # bool
    elif block_type == 'TrueLit':
        return TrueLit()
    elif block_type == 'FalseLit':
        return FalseLit()

    elif block_type == 'NullLit':
        return NullLit()

    # null
    elif block_type == 'UNRECOGNIZED':
        return Unrecognized()

    else:
        raise ValueError(f"Unknown block type {block_type}")


##########
# Debugging Dictionary-Memory

def str_tup(tup, left_is_reduced, right_is_reduced):
    ''' A helper function for `print_mem` that. It is for printing references in
    expressions, and coloring them red if they have not been recorded as reduced
    yet, and green if they have been reduced.'''
    if (left_is_reduced is not None and
        right_is_reduced is not None):
        s = "("
        for i, (t, ir_color) in enumerate(zip(tup, [None, left_is_reduced.item(), right_is_reduced.item()])):
            s += D.colorize(str(t), value=ir_color)
            # br()

            # add commas
            if i < len(tup)-1:
                s += ', '
        s += ")"
        return s
    else:
        s = "("
        for i, t in enumerate(tup):
            s += str(t)

            # add commas
            if i < len(tup)-1:
                s += ', '
        s += ")"
        return s


def print_mem(mem, ir1, ir2, resugar_app=False, resugar_fn=False):
    '''Print a human-readable version of the neurallabda's memory. Useful for
    debugging.

    Args:
      mem: a dictionary representation of memory with block representation of
           the terms
      ir1: tensor, shape(BATCH, N_ADDRESSES). Records (0, 1) for whether or not
           the the reference in col1 is reduced yet.
      ir2: same as ir1 but for col2
    '''
    ss = sorted(mem.items(), key=lambda item: item[0].i)
    for i, (a, x) in enumerate(ss):
        terms = memory_to_terms(mem, a, resugar_app=resugar_app, resugar_fn=resugar_fn)
        if terms != Unrecognized():
            if ir1 is not None and ir2 is not None:
                print(f'A({a.i :> 3d}) {str_tup(x, ir1[i], ir2[i])}  ::  {pretty_print(terms)}')
            else:
                print(f'A({a.i :> 3d}) {str_tup(x, None, None)}  ::  {pretty_print(terms)}')

pm = print_mem
