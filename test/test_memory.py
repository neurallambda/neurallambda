'''

Tests for Dictionary-Memory form of a lamda expression.

'''


from dataclasses import dataclass
from lark import Lark, Transformer, Token, Tree
from torch import einsum, tensor, allclose
from torch.nn import functional as F
from typing import Dict, Union, List, Tuple
from typing import Union, List, Any, Type
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.memory import *
import neurallambda.language as L

parser = Lark(L.grammar, start='start', parser='lalr')
transformer = L.LambdaTransformer()


def test_to_memory():
    '''
    Test converting from terms to memory form
    '''

    # Test for Fn with no parameters
    fn_term = Fn(params=[], body=Var('x'))
    curried_fn = curry_fn(fn_term)
    assert curried_fn == fn_term, "Failed for Fn with no parameters"

    # Test for Fn with one parameter
    fn_term = Fn(params=['x'], body=Var('x'))
    curried_fn = curry_fn(fn_term)
    assert curried_fn == fn_term, "Failed for Fn with one parameter"

    # Test for Fn with two parameters
    fn_term = Fn(params=['x', 'y'], body=Var('x'))
    curried_fn = curry_fn(fn_term)
    expected_fn = Fn(params=['x'], body=Fn(params=['y'], body=Var('x')))
    assert curried_fn == expected_fn, "Failed for Fn with two parameters"

    # Test for Fn with multiple parameters
    fn_term = Fn(params=['x', 'y', 'z'], body=Var('x'))
    curried_fn = curry_fn(fn_term)
    expected_fn = Fn(params=['x'], body=Fn(params=['y'], body=Fn(params=['z'], body=Var('x'))))
    assert curried_fn == expected_fn, "Failed for Fn with multiple parameters"

    # No params
    fn_term = Fn(params=[], body=Var('y'))
    _, _, memory = process_fn(fn_term, 0, {}, {})
    expected_memory = {
        A(0): ('Fn', None, A(1)),
        A(1): ('Var', 'y')
    }
    assert memory == expected_memory, f"Failed for no parameters: {pm(memory)}"


    # One param
    fn_term = Fn(params=['x'], body=Var('x'))
    _, _, memory = process_fn(fn_term, 0, {}, {})
    expected_memory = {
        A(0): ('Fn', A(1), A(1)),
        A(1): ('Var', 'x')
    }
    assert memory == expected_memory, f"Failed for one parameter: {pm(memory)}"


    # Multi-params
    fn_term = Fn(params=['x', 'y'], body=Var('x'))
    _, _, memory = process_fn(fn_term, 0, {}, {})
    expected_memory = {
        A(0): ('Fn', A(1), A(2)),
        A(1): ('Var', 'x'),
        A(2): ('Fn', A(3), A(1)),
        A(3): ('Var', 'y')
    }
    assert memory == expected_memory, f"Failed for two parameters: {pm(memory)}"

    # @@@@@@@@@@
    # desugar_app

    # Test with No Terms
    assert desugar_app(App([])) == App([]), "Failed for no terms"

    # Test with One Term
    assert desugar_app(App([Var('x')])) == App([Var('x')]), "Failed for one term"

    # Test with Two Terms
    assert desugar_app(App([Var('x'), Var('y')])) == App([Var('x'), Var('y')]), "Failed for two terms"

    # Test with Three Terms
    assert desugar_app(App([Var('x'), Var('y'), Var('z')])) == App([App([Var('x'), Var('y')]), Var('z')]), "Failed for three terms"

    # Test with Multiple Terms
    assert desugar_app(App([Var('a'), Var('b'), Var('c'), Var('d')])) == App([App([App([Var('a'), Var('b')]), Var('c')]), Var('d')]), "Failed for multiple terms"


    # @@@@@@@@@@
    # `process_app`

    # Apply with no arguments
    app_term = App([Var('x')])
    _, _, memory = process_app(app_term, 0, {}, {})
    expected_memory = {
        A(0): ('App', A(1)),
        A(1): ('Var', 'x'),
    }
    assert memory == expected_memory, f"Failed for no-arg App: {pm(memory)}"

    # Simple
    app_term = App([Var('x'), Var('y')])
    _, _, memory = process_app(app_term, 0, {}, {})
    expected_memory = {
        A(0): ('App', A(1), A(2)),
        A(1): ('Var', 'x'),
        A(2): ('Var', 'y'),
    }
    assert memory == expected_memory, f"Failed for simple App: {pm(memory)}"

    # Test with Nested App Terms
    nested_app_term = App([Var('f'), App([Var('g'), Var('h')])])
    _, _, memory = process_app(nested_app_term, 0, {}, {})
    expected_memory = {
        A(0): ('App', A(1), A(2)),
        A(1): ('Var', 'f'),
        A(2): ('App', A(3), A(4)),
        A(3): ('Var', 'g'),
        A(4): ('Var', 'h'),
    }
    assert memory == expected_memory, f"Failed for Nested App Terms: {pm(memory)}"

    # Test with More than Two Arguments: (f x y z)
    multi_arg_app_term = App([Var('f'), Var('x'), Var('y'), Var('z')])
    _, _, memory = process_app(multi_arg_app_term, 0, {}, {})
    expected_memory = dict(
        [(A(0), ('App', A(1), A(6))),
         (A(1), ('App', A(2), A(5))),
         (A(2), ('App', A(3), A(4))),
         (A(3), ('Var', 'f')),
         (A(4), ('Var', 'x')),
         (A(5), ('Var', 'y')),
         (A(6), ('Var', 'z'))])
    assert memory == expected_memory, f"Failed for More than Two Arguments: {pm(memory)}"


    # @@@@@@@@@@
    # `process_term`

    x1 = Fn(params=['x'], body=App([Var(name='x'), Var(name='x')]))
    _, _, mem = process_fn(x1, 0, {}, {})
    assert mem == dict([(A(0), ('Fn', A(1), A(2))),
                        (A(1), ('Var', 'x')),
                        (A(2), ('App', A(1), A(1)))])

    x2 = Fn(params=[], body=Var(name='y'))
    _, _, mem = process_fn(x2, 0, {}, {})
    assert mem == dict([(A(0), ('Fn', None, A(1))),
                        (A(1), ('Var', 'y'))])

    x3 = Fn(params=['a', 'b'], body=App([Var(name='a'), Var(name='b')]))
    _, _, mem = process_fn(x3, 0, {}, {})
    assert mem == dict([(A(0), ('Fn', A(1), A(2))),
                        (A(1), ('Var', 'a')),
                        (A(2), ('Fn', A(3), A(4))),
                        (A(3), ('Var', 'b')),
                        (A(4), ('App', A(1), A(3)))])

    x4 = App([Var(name='x'), Var(name='y'), Var(name='z')])
    _, _, mem = process_app(x4, 0, {}, {})
    assert mem == dict([(A(0), ('App', A(1), A(4))),
                        (A(1), ('App', A(2), A(3))),
                        (A(2), ('Var', 'x')),
                        (A(3), ('Var', 'y')),
                        (A(4), ('Var', 'z'))])

    x5 = Fn(params=['x'], body=Fn(params=['y'], body=App([Var(name='x'), Var(name='y')])))
    _, _, mem = process_fn(x5, 0, {}, {})
    assert mem == dict([(A(0), ('Fn', A(1), A(2))),
                        (A(1), ('Var', 'x')),
                        (A(2), ('Fn', A(3), A(4))),
                        (A(3), ('Var', 'y')),
                        (A(4), ('App', A(1), A(3)))])

    x6 = App([App([Var(name='x'), Var(name='y')]), Var(name='z')])
    _, _, mem = process_app(x6, 0, {}, {})
    assert mem == dict([(A(0), ('App', A(1), A(4))),
                        (A(1), ('App', A(2), A(3))),
                        (A(2), ('Var', 'x')),
                        (A(3), ('Var', 'y')),
                        (A(4), ('Var', 'z'))])


    # @@@@@@@@@@

    x7 = IntLit(value=42)
    _, _, mem = process_int_lit(x7, 0, {})
    assert mem == dict([(A(0), ('IntLit', 42))]), "Failed for IntLit"

    x8 = ArithOp(operator='+')
    _, _, mem = process_arith_op(x8, 0, {})
    assert mem == dict([(A(0), ('ArithOp', '+'))]), "Failed for ArithOp"

    x9 = TrueLit()
    _, _, mem = process_true_lit(x9, 0, {})
    assert mem == dict([(A(0), ('TrueLit',))]), "Failed for TrueLit"

    x10 = FalseLit()
    _, _, mem = process_false_lit(x10, 0, {})
    assert mem == dict([(A(0), ('FalseLit',))]), "Failed for FalseLit"

    print('term_to_memory tests passed')


    # @@@@@@@@@@
    # Lists

    # Test Cases for Lists
    x11 = Cons(Var('x'), Empty())
    _, _, mem = process_cons(x11, 0, {}, {})
    assert mem == dict([(A(0), ('Cons', A(1), A(2))),
                        (A(1), ('Var', 'x')),
                        (A(2), ('Empty',))]), "Failed for Cons"

    x12 = Car(Cons(Var('x'), Empty()))
    _, _, mem = process_car(x12, 0, {}, {})
    assert mem == dict([(A(0), ('Car', A(1))),
                        (A(1), ('Cons', A(2), A(3))),
                        (A(2), ('Var', 'x')),
                        (A(3), ('Empty',))]), "Failed for Car"

    x13 = Cdr(Cons(Var('x'), Empty()))
    _, _, mem = process_cdr(x13, 0, {}, {})
    assert mem == dict([(A(0), ('Cdr', A(1))),
                        (A(1), ('Cons', A(2), A(3))),
                        (A(2), ('Var', 'x')),
                        (A(3), ('Empty',))]), "Failed for Cdr"

    x14 = ListP(Var('x'))
    _, _, mem = process_list_p(x14, 0, {}, {})
    assert mem == dict([(A(0), ('ListP', A(1))),
                        (A(1), ('Var', 'x'))]), "Failed for ListP"

    # '(1 2 3 4)
    x15 = Cons(head=IntLit(value=1), tail=Cons(head=IntLit(value=2), tail=Cons(head=IntLit(value=3), tail=Cons(head=IntLit(value=4), tail=Empty()))))
    _, _, mem = process_cons(x15, 0, {}, {})
    assert mem == dict(
    [(A(0), ('Cons', A(1), A(2))),
     (A(1), ('IntLit', 1)),
     (A(2), ('Cons', A(3), A(4))),
     (A(3), ('IntLit', 2)),
     (A(4), ('Cons', A(5), A(6))),
     (A(5), ('IntLit', 3)),
     (A(6), ('Cons', A(7), A(8))),
     (A(7), ('IntLit', 4)),
     (A(8), ('Empty',))]
    ), "Failed for long list"

    print('List tests passed')


    # @@@@@@@@@@
    # Defn

    # Test Cases for Defn
    x15 = Defn(name='f', params=['x'], body=Var('x'))
    _, _, mem = process_defn(x15, 0, {}, {})
    assert mem == dict([(A(0), ('Defn', A(1), A(2))),
                        (A(1), ('DefnName', 'f')),
                        (A(2), ('Fn', A(3), A(3))),
                        (A(3), ('Var', 'x'))]), "Failed for Defn"

    x16 = Defn(name='g', params=[], body=App([Var('x'), Var('y')]))
    _, _, mem = process_defn(x16, 0, {}, {})
    assert mem == dict([(A(0), ('Defn', A(1), A(2))),
                        (A(1), ('DefnName', 'g')),
                        (A(2), ('Fn', None, A(3))),
                        (A(3), ('App', A(4), A(5))),
                        (A(4), ('Var', 'x')),
                        (A(5), ('Var', 'y'))]), "Failed for Defn with App body"

    print('Defn tests passed')


    # @@@@@@@@@@
    # Integration tests

    # Nested function definitions
    x17 = Defn(name='outer', params=['x'],
               body=Defn(name='inner', params=['y'],
                         body=App([Var('x'), Var('y')])))

    _, _, mem = process_term(x17, 0, {}, {})
    assert mem == dict([(A(0), ('Defn', A(1), A(2))),
                        (A(1), ('DefnName', 'outer')),
                        (A(2), ('Fn', A(3), A(4))),
                        (A(3), ('Var', 'x')),
                        (A(4), ('Defn', A(5), A(6))),
                        (A(5), ('DefnName', 'inner')),
                        (A(6), ('Fn', A(7), A(8))),
                        (A(7), ('Var', 'y')),
                        (A(8), ('App', A(3), A(7)))])

    # Function with list operations
    x18 = Defn(name='processList', params=['lst'],
               body=Car(Cdr(Var('lst'))))

    _, _, mem = process_term(x18, 0, {}, {})
    assert mem == dict([(A(0), ('Defn', A(1), A(2))),
                        (A(1), ('DefnName', 'processList')),
                        (A(2), ('Fn', A(3), A(4))),
                        (A(3), ('Var', 'lst')),
                        (A(4), ('Car', A(5))),
                        (A(5), ('Cdr', A(3)))])

    # Complex application of functions
    x19 = App([Defn(name='add', params=['x', 'y'],
                    body=App([ArithOp('+'), Var('x'), Var('y')])),
               IntLit(5),
               IntLit(10)])
    _, _, mem = process_term(x19, 0, {}, {})

    assert mem == dict([(A(0), ('App', A(1), A(12))),
                        (A(1), ('App', A(2), A(11))),
                        (A(2), ('Defn', A(3), A(4))),
                        (A(3), ('DefnName', 'add')),
                        (A(4), ('Fn', A(5), A(6))),
                        (A(5), ('Var', 'x')),
                        (A(6), ('Fn', A(7), A(8))),
                        (A(7), ('Var', 'y')),
                        (A(8), ('App', A(9), A(7))),
                        (A(9), ('App', A(10), A(5))),
                        (A(10), ('ArithOp', '+')),
                        (A(11), ('IntLit', 5)),
                        (A(12), ('IntLit', 10))])


def test_from_memory():
    '''
    Test converting back from memory to terms
    '''

    # @@@@@@@@@@
    # Fn + Var

    # Test for Var
    memory = {A(0): ('Var', 'x')}
    term = memory_to_terms(memory, A(0))
    assert Var('x') == term, "Failed for Var"

    # Test for Fn with no parameters
    memory = {A(0): ('Fn', None, A(1)),
              A(1): ('Var', 'y')}
    term = memory_to_terms(memory, A(0))
    assert Fn([], Var('y')) == term, "Failed for Fn with no parameters"

    # Test for Fn with one parameter
    memory = {A(0): ('Fn', A(1), A(2)),
              A(1): ('Var', 'x'),
              A(2): ('Var', 'y')}
    term = memory_to_terms(memory, A(0))
    assert Fn(['x'], Var('y')) == term, "Failed for Fn with one parameter"

    # Test for Fn with multiple parameters (re-sugared from curried form)
    memory = {A(0): ('Fn', A(1), A(2)),
              A(1): ('Var', 'x'),
              A(2): ('Fn', A(3), A(4)),
              A(3): ('Var', 'y'),
              A(4): ('Var', 'z')}
    term = memory_to_terms(memory, A(0))
    assert Fn(['x', 'y'], Var('z')) == term, "Failed for Fn with multiple parameters"

    # Test for Fn with multiple parameters and NO resugaring
    memory = {A(0): ('Fn', A(1), A(2)),
              A(1): ('Var', 'x'),
              A(2): ('Fn', A(3), A(4)),
              A(3): ('Var', 'y'),
              A(4): ('Var', 'z')}
    term = memory_to_terms(memory, A(0), resugar_fn=False)
    assert Fn(['x'], Fn(['y'], Var('z'))) == term, "Failed for Fn with multiple parameters and no resugaring"

    # Test for Fn with multiple parameters and NO resugaring
    prog = "(fn [x y z] q)"
    tree = parser.parse(prog)
    ast = transformer.transform(tree)
    memory = terms_to_memory(ast)
    term = memory_to_terms(memory, A(0), resugar_fn=False)
    print(pretty_print(term))
    assert Fn(['x'], Fn(['y'], Fn(['z'], Var('q')))) == term, "Failed for Fn with multiple parameters and no resugaring"

    # Test for App
    memory = {A(0): ('App', A(1), A(2)),
              A(1): ('Var', 'f'),
              A(2): ('Var', 'x')}
    term = memory_to_terms(memory, A(0))
    assert App([Var('f'), Var('x')]) == term, 'Failed for App'

    print("Basic memory_to_terms tests passed")


    # @@@@@@@@@@
    # Defn

    # Test for Defn with no parameters (def f [] y)
    memory = {
        A(0): ('Defn', A(1), A(2)),
        A(1): ('DefnName', 'f'),
        A(2): ('Fn', None, A(3)),
        A(3): ('Var', 'y')
    }
    term = memory_to_terms(memory, A(0))
    assert term == Defn('f', [], Var(name='y')), "Failed for Defn with no parameters"

    # Test for Defn with parameters (def f [x] x)
    memory = {
        A(0): ('Defn', A(1), A(2)),
        A(1): ('DefnName', 'f'),
        A(2): ('Fn', A(3), A(4)),
        A(3): ('Var', 'x'),
        A(4): ('Var', 'x')
    }
    term = memory_to_terms(memory, A(0))
    assert term == Defn('f', ['x'], Var(name='x')), "Failed for Defn with parameters"

    print("All Defn tests passed")


    # @@@@@@@@@@
    # Bool

    # Test for TrueLit
    memory = {A(0): ('TrueLit',)}
    term = memory_to_terms(memory, A(0))
    assert term == TrueLit(), "Failed for TrueLit"

    # Test for FalseLit
    memory = {A(0): ('FalseLit',)}
    term = memory_to_terms(memory, A(0))
    assert term == FalseLit(), "Failed for FalseLit"


    # @@@@@@@@@@
    # Arithmetic

    memory = {A(0): ('IntLit', 42)}
    term = memory_to_terms(memory, A(0))
    assert term == IntLit(42), "Failed for IntLit"

    memory = {A(0): ('ArithOp', '+')}
    term = memory_to_terms(memory, A(0))
    assert term == ArithOp('+'), "Failed for ArithOp"

    # (+ 152 42)
    memory = {A(0): ('App', A(1), A(5)),
              A(1): ('App', A(3), A(4)),
              A(3): ('ArithOp', '+'),
              A(4): ('IntLit', 152),
              A(5): ('IntLit', 42),
              }
    term = memory_to_terms(memory, A(0))
    assert term == App([ArithOp('+'), IntLit(152), IntLit(42)]), "Failed for complex ArithOp"


    # @@@@@@@@@@
    # Lists

    # Test for Cons: (Cons x Empty)
    memory = {
        A(0): ('Cons', A(1), A(2)),
        A(1): ('Var', 'x'),
        A(2): ('Empty',)
    }
    term = memory_to_terms(memory, A(0))
    assert term == Cons(Var(name='x'), Empty()), "Failed for Cons"

    # Test for Car: (Car (Cons x Empty))
    memory = {
        A(0): ('Car', A(1)),
        A(1): ('Cons', A(2), A(3)),
        A(2): ('Var', 'x'),
        A(3): ('Empty',)
    }
    term = memory_to_terms(memory, A(0))
    assert term == Car(Cons(Var(name='x'), Empty())), "Failed for Car"

    # Test for Cdr: (Cdr (Cons x Empty))
    memory = {
        A(0): ('Cdr', A(1)),
        A(1): ('Cons', A(2), A(3)),
        A(2): ('Var', 'x'),
        A(3): ('Empty',)
    }
    term = memory_to_terms(memory, A(0))
    assert term == Cdr(Cons(Var(name='x'), Empty())), "Failed for Cdr"

    # Test for ListP: (ListP x)
    memory = {
        A(0): ('ListP', A(1)),
        A(1): ('Var', 'x')
    }
    term = memory_to_terms(memory, A(0))
    assert term == ListP(Var(name='x')), "Failed for ListP"

    print("All list tests passed")



    # @@@@@@@@@@
    # re-sugar App

    # Test for App with single function (f)
    memory = {
        A(0): ('App', A(1)),
        A(1): ('Var', 'f')
    }
    term = memory_to_terms(memory, A(0))
    assert term == App(terms=[Var(name='f')]), "Failed for App (f)"

    # Test for App with a function and single argument (f x)
    memory = {
        A(0): ('App', A(1), A(2)),
        A(1): ('Var', 'f'),
        A(2): ('Var', 'x')
    }
    term = memory_to_terms(memory, A(0))
    assert term == App(terms=[Var(name='f'), Var(name='x')]), "Failed for App (f x)"

    # Test for App with two arguments: (f x y)
    memory = {
        A(0): ('App', A(1), A(4)),  # Outer application
        A(1): ('App', A(2), A(3)),  # Nested application
        A(2): ('Var', 'f'),         # Function f
        A(3): ('Var', 'x'),         # First argument x
        A(4): ('Var', 'y')          # Second argument y
    }
    term = memory_to_terms(memory, A(0))
    assert term == App(terms=[Var(name='f'), Var(name='x'), Var(name='y')]), "Failed for App (f x y)"

    # Test for App with multiple arguments: (f x y z)
    memory = {
        A(0): ('App', A(1), A(6)),
        A(1): ('App', A(2), A(5)),
        A(2): ('App', A(3), A(4)),
        A(3): ('Var', 'f'),
        A(4): ('Var', 'x'),
        A(5): ('Var', 'y'),
        A(6): ('Var', 'z')
    }
    term = memory_to_terms(memory, A(0))
    assert term == App(terms=[Var(name='f'), Var(name='x'), Var(name='y'), Var(name='z')]), "Failed for App with multiple arguments"

    print("resuagaring memory_to_terms tests passed")


    # @@@@@@@@@@
    # don't re-sugar App

    # Test for App with single function (f)
    memory = {
        A(0): ('App', A(1)),
        A(1): ('Var', 'f')
    }
    term = memory_to_terms(memory, A(0), resugar_app=False)
    assert term == App(terms=[Var(name='f')]), "Failed for App (f)"

    # Test for App with a function and single argument (f x)
    memory = {
        A(0): ('App', A(1), A(2)),
        A(1): ('Var', 'f'),
        A(2): ('Var', 'x')
    }
    term = memory_to_terms(memory, A(0), resugar_app=False)
    assert term == App(terms=[Var(name='f'), Var(name='x')]), "Failed for App (f x)"

    # Test for App with two arguments: (f x y)
    memory = {
        A(0): ('App', A(1), A(4)),  # Outer application
        A(1): ('App', A(2), A(3)),  # Nested application
        A(2): ('Var', 'f'),         # Function f
        A(3): ('Var', 'x'),         # First argument x
        A(4): ('Var', 'y')          # Second argument y
    }
    term = memory_to_terms(memory, A(0), resugar_app=False)
    assert App(terms=[App(terms=[Var(name='f'), Var(name='x')]), Var(name='y')]) == term, "Failed for App (f x y)"

    # Test for App with multiple arguments: (f x y z)
    memory = {
        A(0): ('App', A(1), A(6)),
        A(1): ('App', A(2), A(5)),
        A(2): ('App', A(3), A(4)),
        A(3): ('Var', 'f'),
        A(4): ('Var', 'x'),
        A(5): ('Var', 'y'),
        A(6): ('Var', 'z')
    }
    term = memory_to_terms(memory, A(0), resugar_app=False)
    assert App(terms=[App(terms=[App(terms=[Var(name='f'), Var(name='x')]), Var(name='y')]), Var(name='z')]) == term, "Failed for App with multiple arguments"

    # Test List of Lists: Integration test
    prog = "'('(z z) '(z z) '(z z))"
    tree = parser.parse(prog)
    ast = transformer.transform(tree)
    two = Cons(Var('z'), Cons(Var('z'), Empty()))
    three = Cons(two, Cons(two, Cons(two, Empty())))
    assert ast[0] == three
    mem = terms_to_memory(ast)
    recon = memory_to_terms(mem, A(0))
    assert prog == pretty_print(recon)
