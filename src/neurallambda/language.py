'''

A Simple Lambdacalculus Language Definition

'''

from dataclasses import dataclass
from lark import Lark, Transformer, Token, Tree
from torch import einsum, tensor, allclose
from typing import Dict, Union, List, Tuple
from typing import Union, List, Any, Type


##################################################
# Syntax

# Recursive type definition using forward declaration
Term: Type['Term'] = None


##########
# Lambda Calculus

@dataclass
class Defn:
    name: str
    params: List[str]
    body: Term

@dataclass
class Fn:
    params: List[str]
    body: Term

@dataclass
class Var:
    name: str

@dataclass
class App:
   terms: List[Term]


##########
# Math Literals

@dataclass
class IntLit:
    value: int

@dataclass
class ArithOp:
    operator: str


##########
# Boolean Literals

@dataclass
class TrueLit:
    pass

@dataclass
class FalseLit:
    pass

@dataclass
class NullLit:
    pass


##########
# List Literals

# Recursive type definition using forward declaration
LinkedList = None

@dataclass
class Empty:
    pass

@dataclass
class Cons:
    head: Term
    tail: LinkedList

LinkedList: Type['LinkedList'] = Union[Empty, LinkedList]

@dataclass
class Car:
    list: Term

@dataclass
class Cdr:
    list: LinkedList

@dataclass
class ListP:
    term: Term


##########
# Error Types

@dataclass
class Unrecognized:
    pass

@dataclass
class Error:
    message: str


##########
# Syntax

Term = Union[
    # Lambda Calc
    Defn, Fn, Var, App,
    # Boolean
    TrueLit, FalseLit,
    # Math
    IntLit, ArithOp,
    # List
    Empty, Cons, LinkedList, Car, Cdr, ListP
]


##################################################
# Parsing

grammar = """
start: s_expression+
s_expression: atomic_symbol
            | defn
            | fn
            | list
            | number
            | operator

            | true
            | false
            | null

            | cons
            | car
            | cdr
            | listp

            | application

list: "'(" s_expression* ")"
atomic_symbol: VAR
VAR: /[a-z]+/
function_symbol: FUNC
FUNC: /[a-z]+/

true: "true"
false: "false"
null: "null"

number: NUMBER
NUMBER: /[0-9]+/
operator: OPERATOR
OPERATOR: "+" | "-" | "*" | "/"
args: "[" atomic_symbol* "]"
defn: "(defn" function_symbol args s_expression ")"
fn: "(fn" args s_expression ")"

cons: "(cons" s_expression s_expression ")"
car: "(car" s_expression ")"
cdr: "(cdr" s_expression ")"
listp: "(listp" s_expression ")"

application: "(" s_expression s_expression* ")"

%ignore " "
%ignore "\\n"
"""

class LambdaTransformer(Transformer):
   def start(self, children: List[Term]) -> Term:
       return children

   def s_expression(self, children: List[Term]) -> Term:
       if len(children) == 1:
           return children[0]
       else:
           return App(children)

   def application(self, children: List[Term]) -> Term:
       return App(children)

   def true(self, _):
       return TrueLit()

   def false(self, _):
       return FalseLit()

   def null(self, _):
       return NullLit()

   def list(self, children: List[Term]) -> LinkedList:
       if len(children) == 0:
           return Empty()
       else:
           return Cons(children[0], self.list(children[1:]))

   def atomic_symbol(self, children: List[Token]) -> Var:
       return Var(children[0].value)

   def function_symbol(self, children: List[Token]) -> Var:
       return children[0].value

   def number(self, children: List[Token]) -> IntLit:
       return IntLit(int(children[0].value))

   def operator(self, children: List[Token]) -> str:
       return ArithOp(children[0].value)

   def args(self, children: List[Var]) -> List[str]:
       return [child.name for child in children]

   def defn(self, children: List[Any]) -> Defn:
       name = children[0]
       params = children[1]
       body = children[2]
       return Defn(name, params, body)

   def fn(self, children: List[Any]) -> Fn:
       params = children[0]
       body = children[1]
       return Fn(params, body)

   def cons(self, children: List[Term]) -> Cons:
       return Cons(children[0], children[1])

   def car(self, children: List[Term]) -> Car:
       return Car(children[0])

   def cdr(self, children: List[Term]) -> Cdr:
       return Cdr(children[0])

   def listp(self, children: List[Term]) -> ListP:
       return ListP(children[0])


def string_to_terms(s):
    parser = Lark(grammar, start='start', parser='lalr')
    transformer = LambdaTransformer()
    tree = parser.parse(s)
    return transformer.transform(tree)

##################################################
# Pretty Printer
#
#  Human-readable expressions
#

def pretty_print(term):
    match term:
        case Var():
            return term.name
        case App():
            return '(' + ' '.join(map(pretty_print, term.terms)) + ')'
        case IntLit():
            return str(term.value)
        case ArithOp():
            return term.operator
        case Defn():
            params = ' '.join(term.params)
            return f'(defn {term.name} [{params}] {pretty_print(term.body)})'
        case Fn(params, _) if isinstance(params, list) and all(isinstance(p, str) for p in params):
            params = ' '.join(term.params)
            return f'(fn [{params}] {pretty_print(term.body)})'
        case Fn(_, _):
            return f'corrupt Fn: {term}'
        case Empty():
            return "'()"
        case Cons():
            return pretty_print_list(term)
        case Car():
            return f'(car {pretty_print(term.list)})'
        case Cdr():
            return f'(cdr {pretty_print(term.list)})'
        case ListP():
            return f'(listp {pretty_print(term.term)})'
        case TrueLit():
            return 'true'
        case FalseLit():
            return 'false'
        case NullLit():
            return 'null'
        case Unrecognized():
            return 'Unrecognized'
        case _:
            return f'UnknownTerm: {str(term)}'


def pretty_print_list(term):
    """Helper function to pretty print a list made of Cons cells."""
    elements = []
    while isinstance(term, Cons):
        elements.append(pretty_print(term.head))
        term = term.tail
    if isinstance(term, Empty):
        return "'(" + ' '.join(elements) + ")"
    else:  # Not a proper list, so fall back to cons
        if isinstance(term, Cons):
            return f'(cons {pretty_print(term.head)} {pretty_print(term.tail)})'
        else:
            return f'(cons {term})'


def pretty_print_program(program):
    return '\n'.join(map(pretty_print, program))


##################################################
# Beta reduction
#
#   NOTE: This function is NOT a part of demonstrating / using Neurallambdas, it
#         is just stock lambda-calc stuff for testing/experimentation. It is
#         however, in broadstrokes, a demonstration of what the Neurallambda
#         technique is doing under the hood, via other means.
#

def beta(term: Term) -> Term:
    match term:
        case App():
            terms = [beta(t) for t in term.terms]
            match terms[0]:
                case Fn() as func:
                    args = terms[1:]
                    if len(func.params) != len(args):
                        raise ValueError("Function called with incorrect number of arguments")
                    body = func.body
                    for param, arg in zip(func.params, args):
                        body = substitute(body, param, arg)
                    return beta(body)
                case _:
                    return App(terms)
        case Defn():
            return Defn(term.name, term.params, beta(term.body))

        case TrueLit():
            return TrueLit()

        case FalseLit():
            return FalseLit()

        ##########
        # Lists
        case Cons():
            return Cons(beta(term.head), beta(term.tail))
        case Car():
            if isinstance(term.list, Cons):
                return beta(term.list.head)
            else:
                raise ValueError("Car called on a non-cons term")
        case Cdr():
            if isinstance(term.list, Cons):
                return beta(term.list.tail)
            else:
                raise ValueError("Cdr called on a non-cons term")
        case ListP():
            evaluated_term = beta(term.term)
            if isinstance(evaluated_term, (Cons, Empty)):
                return TrueLit()
            else:
                return FalseLit()
        case _:
            return term

def substitute(term: Term, var: str, replacement: Term) -> Term:
    match term:
        case Var() if term.name == var:
            return replacement
        case Var():
            return term
        case App():
            return App([substitute(t, var, replacement) for t in term.terms])
        case Fn() if var in term.params:
            return term
        case Fn():
            return Fn(term.params, substitute(term.body, var, replacement))
        case Defn() if var in term.params:
            return term
        case Defn():
            return Defn(term.name, term.params, substitute(term.body, var, replacement))
        case _:
            return term
