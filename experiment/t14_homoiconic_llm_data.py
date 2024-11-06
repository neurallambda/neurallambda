'''

Generate a toy dataset of arithmetic problems

Example:

a=1^
b=2^
c=a+b^
solve(c)


'''

from dataclasses import dataclass
from typing import Union, List, Dict
import random

@dataclass
class IntLiteral:
    value: int

@dataclass
class Variable:
    name: str

@dataclass
class BinaryOperation:
    op: str
    left: Union[IntLiteral, Variable]
    right: Union[IntLiteral, Variable]

Expression = Union[IntLiteral, Variable, BinaryOperation]

@dataclass
class Puzzle:
    variables: Dict[str, Expression]
    solve: str

def generate_simple_expression(available_vars: List[str], nums: List[int], ops: List[str]) -> Expression:
    if not available_vars or random.random() < 0.3:
        return IntLiteral(random.choice(nums))
    elif random.random() < 0.5:
        return Variable(random.choice(available_vars))
    else:
        op = random.choice(ops)
        left = random.choice([IntLiteral(random.choice(nums)), Variable(random.choice(available_vars))])
        right = random.choice([IntLiteral(random.choice(nums)), Variable(random.choice(available_vars))])
        return BinaryOperation(op, left, right)

def make_puzzle(vars: List[str], nums: List[int], ops: List[str]) -> Puzzle:
    variables = {}
    available_vars = []
    for var in vars:
        variables[var] = generate_simple_expression(available_vars, nums, ops)
        available_vars.append(var)

    solve = random.choice(vars)
    return Puzzle(variables, solve)

def evaluate(expr: Expression, variables: Dict[str, Expression]) -> int:
    match expr:
        case IntLiteral(value):
            return value
        case Variable(name):
            return evaluate(variables[name], variables)
        case BinaryOperation(op, left, right):
            left_val = evaluate(left, variables)
            right_val = evaluate(right, variables)
            match op:
                case '+':
                    return left_val + right_val
                case '-':
                    return left_val - right_val
                case '*':
                    return left_val * right_val
                case _:
                    raise ValueError(f"Unknown operator: {op}")
        case _:
            raise TypeError(f"Unknown expression type: {type(expr)}")

def pretty_print(puzzle: Puzzle) -> str:
    result = []
    for var, value in puzzle.variables.items():
        result.append(f"{var} = {expression_to_str(value)}")
    result.append(f"solve({puzzle.solve}) = {evaluate(Variable(puzzle.solve), puzzle.variables)}")
    return '\n'.join(result)

def expression_to_str(expr: Expression) -> str:
    match expr:
        case IntLiteral(value):
            return str(value)
        case Variable(name):
            return name
        case BinaryOperation(op, left, right):
            return f"{expression_to_str(left)} {op} {expression_to_str(right)}"
        case _:
            raise TypeError(f"Unknown expression type: {type(expr)}")


##########
# Demo
vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
nums = [-2, -1, 0, 1, 2]
ops = ['+', '-', '*']

for _ in range(5):
    puzzle = make_puzzle(vars, nums, ops)
    print(pretty_print(puzzle))
    print()
