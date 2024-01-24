'''

Tests for Lambdacalc language spec

'''


from dataclasses import dataclass
from lark import Lark, Transformer, Token, Tree
from typing import Dict, Union, List, Tuple
from typing import Union, List, Any, Type
import neurallambda.language as L
from neurallambda.language import *
import numpy as np
import pprint
import random

def test():

    ##################################################
    # Paser Tests

    parser = Lark(L.grammar, start='start', parser='lalr')
    transformer = L.LambdaTransformer()

    tree = parser.parse('a')
    ast = transformer.transform(tree)
    assert ast == [Var('a')]

    tree = parser.parse('1')
    ast = transformer.transform(tree)
    assert ast == [IntLit(1)]

    tree = parser.parse('(a b)')
    ast = transformer.transform(tree)
    assert ast == [App([Var('a'), Var('b')])]

    tree = parser.parse('(a b )')
    ast = transformer.transform(tree)
    assert ast == [App([Var('a'), Var('b')])]

    tree = parser.parse('(+ 1 2)')
    ast = transformer.transform(tree)
    assert ast == [App([ArithOp('+'), IntLit(1), IntLit(2)])]

    tree = parser.parse('(fn [x] x)')
    ast = transformer.transform(tree)
    assert ast == [Fn(['x'], Var('x'))]

    tree = parser.parse('(defn add [x y] (+ x y))')
    ast = transformer.transform(tree)
    assert ast == [Defn('add', ['x', 'y'], App([ArithOp('+'), Var('x'), Var('y')]))]

    tree = parser.parse('((fn [x] x) (fn [y] y))')
    ast = transformer.transform(tree)
    assert ast == [App([Fn(['x'], Var('x')), Fn(['y'], Var('y'))])]

    tree = parser.parse('(defn add [x y] (+ x y)) (add 1 2)')
    ast = transformer.transform(tree)
    assert ast == [Defn('add', ['x', 'y'], App([ArithOp('+'), Var('x'), Var('y')])),
                   App([Var('add'), IntLit(1), IntLit(2)])]

    tree = parser.parse('(defn add [x y] (- x y)) (add 1 2)')
    ast = transformer.transform(tree)
    assert ast == [Defn('add', ['x', 'y'], App([ArithOp('-'), Var('x'), Var('y')])),
                   App([Var('add'), IntLit(1), IntLit(2)])]

    tree = parser.parse('(defn add [x y] (* x y)) (add 1 2)')
    ast = transformer.transform(tree)
    assert ast == [Defn('add', ['x', 'y'], App([ArithOp('*'), Var('x'), Var('y')])),
                   App([Var('add'), IntLit(1), IntLit(2)])]

    tree = parser.parse('(defn add [x y] (/ x y)) (add 1 2)')
    ast = transformer.transform(tree)
    assert ast == [Defn('add', ['x', 'y'], App([ArithOp('/'), Var('x'), Var('y')])),
                   App([Var('add'), IntLit(1), IntLit(2)])]

    tree = parser.parse('(defn add [x y] (+ x y)) (add 1 2) (add 3 4)')
    ast = transformer.transform(tree)
    assert ast == [Defn('add', ['x', 'y'], App([ArithOp('+'), Var('x'), Var('y')])),
                   App([Var('add'), IntLit(1), IntLit(2)]),
                   App([Var('add'), IntLit(3), IntLit(4)])]

    tree = parser.parse('((fn [x] (+ x 1)))')
    ast = transformer.transform(tree)
    assert ast == [App([Fn(['x'], App([ArithOp('+'), Var('x'), IntLit(1)]))])]


    # @@@@@@@@@@
    # Lists

    # Test: Parsing a Simple List
    # Description: Ensuring that 'cons' constructs a list correctly.
    tree = parser.parse("(cons 1 '())")
    ast = transformer.transform(tree)
    assert ast == [Cons(IntLit(1), Empty())]

    # List of several items
    tree = parser.parse("'(1 2 3)")
    ast = transformer.transform(tree)
    assert ast == [Cons(head=IntLit(value=1),
                        tail=Cons(head=IntLit(value=2),
                                  tail=Cons(head=IntLit(value=3),
                                            tail=Empty())))]

    tree = parser.parse("(cons 1 '(2 3))")
    ast = transformer.transform(tree)
    assert ast == [Cons(head=IntLit(value=1),
                        tail=Cons(head=IntLit(value=2),
                                  tail=Cons(head=IntLit(value=3),
                                            tail=Empty())))]

    tree = parser.parse("(cons 1 (cons 2 '(3)))")
    ast = transformer.transform(tree)
    assert ast == [Cons(head=IntLit(value=1),
                        tail=Cons(head=IntLit(value=2),
                                  tail=Cons(head=IntLit(value=3),
                                            tail=Empty())))]

    # Test: Parsing Car Operation
    tree = parser.parse("(car (cons 1 '()))")
    ast = transformer.transform(tree)
    assert ast == [Car(Cons(IntLit(1), Empty()))]

    # Test: Parsing Cdr Operation
    tree = parser.parse("(cdr (cons 1 '()))")
    ast = transformer.transform(tree)
    assert ast == [Cdr(Cons(IntLit(1), Empty()))]

    # Test: Parsing List Predicate on a Non-list
    tree = parser.parse("(listp 1)")
    ast = transformer.transform(tree)
    assert ast == [ListP(IntLit(1))]

    # Test: Parsing List Predicate on an Empty List
    tree = parser.parse("(listp '())")
    ast = transformer.transform(tree)
    assert ast == [ListP(Empty())]

    # Test: Parsing List Predicate on a Non-empty List
    tree = parser.parse("(listp (cons 1 '()))")
    ast = transformer.transform(tree)
    assert ast == [ListP(Cons(IntLit(1), Empty()))]

    # Test List of Lists
    tree = parser.parse("'('(z z) '(z z) '(z z))")
    ast = transformer.transform(tree)
    two = Cons(Var('z'), Cons(Var('z'), Empty()))
    three = Cons(two, Cons(two, Cons(two, Empty())))
    assert ast == [three]


    print('All Parser Tests Passed')


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Pretty Printer Tests

    # Example test for list sugar
    assert pretty_print(Cons(IntLit(1), Cons(IntLit(2), Empty()))) == "'(1 2)"
    assert pretty_print(Cons(IntLit(1), Cons(IntLit(2), Cons(IntLit(3), Empty())))) == "'(1 2 3)"
    assert pretty_print(Cons(IntLit(1), Cons(Var('x'), Empty()))) == "'(1 x)"

    # Test: Pretty print a variable
    assert pretty_print(Var('x')) == 'x'

    # Test: Pretty print an application
    assert pretty_print(App([Var('f'), Var('x')])) == '(f x)'

    # Test: Pretty print an integer literal
    assert pretty_print(IntLit(42)) == '42'

    # Test: Pretty print an arithmetic operation
    assert pretty_print(ArithOp('+')) == '+'

    # Test: Pretty print a function definition
    assert pretty_print(Defn('add', ['x', 'y'], App([ArithOp('+'), Var('x'), Var('y')]))) == '(defn add [x y] (+ x y))'

    # Test: Pretty print a function
    assert pretty_print(Fn(['x'], Var('x'))) == '(fn [x] x)'

    # Test: Pretty print list operations
    assert pretty_print(Cons(IntLit(1), Empty())) == "'(1)"
    assert pretty_print(Car(Cons(IntLit(1), Empty()))) == "(car '(1))"
    assert pretty_print(Cdr(Cons(IntLit(1), Empty()))) == "(cdr '(1))"
    assert pretty_print(ListP(Empty())) == "(listp '())"

    # Test: Pretty print true and false literals
    assert pretty_print(TrueLit()) == 'true'
    assert pretty_print(FalseLit()) == 'false'

    # Test: Pretty print a program
    program = [Defn('add', ['x', 'y'], App([ArithOp('+'), Var('x'), Var('y')]))]
    assert pretty_print_program(program) == '(defn add [x y] (+ x y))'

    print('All pretty printer tests passed')

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Stock beta reduction tests

    class BETA_TESTS:
        pass  # for jumping to this location in code

    # Test Suite for Lambda Calculus Compiler

    # Test: Simple Function Application
    # Description: Basic function application with a single parameter.
    tree = parser.parse('((fn [x] (+ x 1)) 5)')
    ast = transformer.transform(tree)
    assert beta(ast[0]) == App([ArithOp('+'), IntLit(5), IntLit(1)])

    # Test: Nested Function Application
    # Description: Applying a function within another function.
    tree = parser.parse('((fn [x] ((fn [y] (+ x y)) 3)) 2)')
    ast = transformer.transform(tree)
    assert beta(ast[0]) == App([ArithOp('+'), IntLit(2), IntLit(3)])

    # Test: Function with Multiple Parameters
    # Description: Function application with multiple parameters.
    tree = parser.parse('((fn [x y] (+ x y)) 7 8)')
    ast = transformer.transform(tree)
    assert beta(ast[0]) == App([ArithOp('+'), IntLit(7), IntLit(8)])

    # Test: Identity Function
    # Description: Testing the identity function.
    tree = parser.parse('((fn [x] x) 9)')
    ast = transformer.transform(tree)
    assert beta(ast[0]) == IntLit(9)

    # Test: Function with No Parameters
    # Description: Function that takes no parameters and is immediately invoked.
    tree = parser.parse('((fn [] 10))')  # The extra parentheses represent an application.
    ast = transformer.transform(tree)
    assert beta(ast[0]) == IntLit(10)

    # Test: Function Application with Variable Shadowing
    # Description: Inner function shadows a variable from the outer function.
    tree = parser.parse('((fn [x] ((fn [x] (+ x 1)) 5)) 4)')
    ast = transformer.transform(tree)
    assert beta(ast[0]) == App([ArithOp('+'), IntLit(5), IntLit(1)])

    # Test: Function Application with No Arguments
    # Description: Testing error handling for function called with no arguments.
    tree = parser.parse('((fn [x] (+ x 1)))')
    ast = transformer.transform(tree)
    try:
        beta(ast[0])
        assert False, "Expected ValueError for incorrect number of arguments"
    except ValueError:
        pass

    # Test: Function with Unused Parameters
    # Description: Function has parameters that are not used in the body.
    tree = parser.parse('((fn [x y] (+ x 2)) 3 4)')
    ast = transformer.transform(tree)
    assert beta(ast[0]) == App([ArithOp('+'), IntLit(3), IntLit(2)])

    # Test: Complex Nested Functions
    # Description: Testing deeply nested functions with multiple parameters.
    tree = parser.parse('((fn [x] ((fn [y] ((fn [z] (+ x (+ y z))) 6)) 5)) 4)')
    ast = transformer.transform(tree)
    assert beta(ast[0]) == App([ArithOp('+'), IntLit(4), App([ArithOp('+'), IntLit(5), IntLit(6)])])

    # Test: Function Application with Extra Arguments
    # Description: Testing error handling for function called with too many arguments.
    tree = parser.parse('((fn [x] (+ x 1)) 2 3)')
    ast = transformer.transform(tree)
    try:
        beta(ast[0])
        assert False, "Expected ValueError for too many arguments"
    except ValueError:
        pass

    # Test: Function Application with Insufficient Arguments
    # Description: Testing error handling for function called with too few arguments.
    tree = parser.parse('((fn [x y] (+ x y)) 1)')
    ast = transformer.transform(tree)
    try:
        beta(ast[0])
        assert False, "Expected ValueError for insufficient arguments"
    except ValueError:
        pass


    # @@@@@@@@@@
    # List

    # Test: Car on a Non-empty List
    # Description: Testing 'car' to retrieve the first element of a list.
    tree = parser.parse("(car (cons 1 '()))")
    ast = transformer.transform(tree)
    assert beta(ast[0]) == IntLit(1)

    # Test: Cdr on a Non-empty List
    # Description: Testing 'cdr' to retrieve the rest of the list.
    tree = parser.parse("(cdr (cons 1 '()))")
    ast = transformer.transform(tree)
    assert beta(ast[0]) == Empty()

    # Test: Car on an Empty List
    # Description: Testing error handling for 'car' called on an empty list.
    tree = parser.parse("(car '())")
    ast = transformer.transform(tree)
    try:
        beta(ast[0])
        assert False, "Expected ValueError for 'car' called on an empty list"
    except ValueError:
        pass

    # Test: Cdr on an Empty List
    # Description: Testing error handling for 'cdr' called on an empty list.
    tree = parser.parse("(cdr '())")
    ast = transformer.transform(tree)
    try:
        beta(ast[0])
        assert False, "Expected ValueError for 'cdr' called on an empty list"
    except ValueError:
        pass

    # Test: List Predicate on a Non-list
    # Description: Testing 'listp' to check if a term is a list.
    tree = parser.parse("(listp 1)")
    ast = transformer.transform(tree)
    assert beta(ast[0]) == FalseLit()

    # Test: List Predicate on an Empty List
    # Description: Testing 'listp' to check if '()' is a list.
    tree = parser.parse("(listp '())")
    ast = transformer.transform(tree)
    assert beta(ast[0]) == TrueLit()

    # Test: List Predicate on a Non-empty List
    # Description: Testing 'listp' to check if '(cons 1 '())' is a list.
    tree = parser.parse("(listp (cons 1 '()))")
    ast = transformer.transform(tree)
    assert beta(ast[0]) == TrueLit()
