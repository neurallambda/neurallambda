'''

Different datasets to use in tests

'''

import random
from typing import Dict, List, Tuple

BOS_SYMBOL = '^'
PAUSE_SYMBOL = '.'
REFLECT_SYMBOL = '|'


def palindrome(num_samples,
               min_length,
               max_length,
               lang) -> Dict[str, List[str]]:
    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        hlength = length // 2
        seed = [random.choice(lang) for _ in range(hlength)]
        # add pauses to inputs and outputs
        inputs = [BOS_SYMBOL] + seed + [REFLECT_SYMBOL] + [PAUSE_SYMBOL] * hlength
        outputs = [PAUSE_SYMBOL] * (hlength + 2) + seed[::-1]
        # convert all symbols to str
        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'inputs': inputs,
            'outputs': outputs,
        })
    return data


def even_pairs(num_samples: int,
               min_length: int,
               max_length: int,
               lang: List[str]) -> List[Dict[str, List[str]]]:
    ''' a "pair" counts as any consecutive symbols that aren't equal. Output
    whether the sum is even or odd '''
    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        inputs = [random.choice(lang) for _ in range(length)]

        pair_count = 0
        for i in range(length - 1):
            if inputs[i] != inputs[i + 1]:
                pair_count += 1

        # Add a final pause token to the inputs
        inputs.append(PAUSE_SYMBOL)

        outputs = [PAUSE_SYMBOL] * length
        if pair_count % 2 == 0:
            outputs.append('2')
        else:
            outputs.append('1')

        # convert all symbols to str
        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))

        data.append({
            'inputs': inputs,
            'outputs': outputs,
        })

    return data



def binary_arithmetic(num_samples, min_value, max_value, op: str):
    assert op in {'+', '*'}
    if op == '+':
        f = lambda x, y: x + y
    elif op == '*':
        f = lambda x, y: x * y
    test_data = []

    for _ in range(num_samples):
        a = random.randint(min_value, max_value)
        b = random.randint(min_value, max_value)

        binary_a   = list(bin(a)[2:])  # peel off '0b...'
        binary_b   = list(bin(b)[2:])
        sum_binary = list(bin(f(a, b))[2:])

        inputs = list(binary_a) + [op] + list(binary_b)
        outputs = list(sum_binary)
        inputs, outputs = (  # depend on each others' lengths. also add BOS.
            [BOS_SYMBOL] + inputs + [PAUSE_SYMBOL] * len(outputs),
            [PAUSE_SYMBOL] + [PAUSE_SYMBOL] * len(inputs) + outputs
        )

        test_data.append({'inputs': inputs, 'outputs': outputs})

    return test_data

# # Example usage
# num_samples = 5
# min_value = 0
# max_value = 255
# test_data = binary_arithmetic(num_samples, min_value, max_value, op='*')

# for sample in test_data:
#     print(f"Inputs : {sample['inputs']}")
#     print(f"Outputs: {sample['outputs']}")
#     print()


def generate_expression(length,
                        numbers: List[int],
                        operations: List[str],
                        brackets: Tuple[str, str]):
    tokens = []
    stack = []
    opens = [x[0] for x in brackets]
    closes = [x[1] for x in brackets]
    unsatisfied_op = False
    while len(tokens) + len(stack) < length:
        r = random.random()
        if tokens and (tokens[-1] in closes or tokens[-1] in numbers): # must be an op
            op = random.choice(operations)
            tokens.append(op)
            unsatisfied_op = True
        if brackets and r < 0.2 and (not tokens or tokens and tokens[-1] not in numbers): # open bracket
            opn, cls = random.choice(brackets)
            tokens.append(opn)
            stack.append(cls)
        elif brackets and r < 0.4 and stack and tokens[-1] not in (opens + operations + numbers): # close bracket
            cls = stack.pop()
            tokens.append(cls)
        elif not tokens or tokens[-1] in opens or tokens[-1] in operations: # add number
            n = random.choice(numbers)
            tokens.append(n)
            unsatisfied_op = False
    if unsatisfied_op:
        n = random.choice(numbers)
        tokens.append(n)
    while tokens and tokens[-1] in opens:  # trim dangling bracket
        tokens = tokens[:-1]
        stack.pop()
    return tokens + stack


def arithmetic_expressions(num_samples: int,
                           min_length: int,
                           max_length: int,
                           numbers: List[int],
                           modulus: int,
                           operations: List[str],
                           brackets: Tuple[str, str]) -> List[Dict[str, List[str]]]:
    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        expr = generate_expression(length, numbers, operations, brackets)

        # evaluate expression
        if len(expr) == 0:
            continue
        result = str(eval(''.join(map(str, expr))) % modulus)

        # pad with pause symbols
        inputs = expr
        outputs = list(result)
        inputs, outputs = (
            [BOS_SYMBOL] + inputs + [PAUSE_SYMBOL] * len(outputs),
            [PAUSE_SYMBOL] + [PAUSE_SYMBOL] * len(inputs) + outputs
        )

        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'inputs': inputs,
            'outputs': outputs
        })

    return data


# num_samples = 5
# min_length = 5
# max_length = 10
# numbers = [1, 2, 3]
# modulus = 3
# operations = ['+', '-', '*']
# brackets = [('(', ')')]

# test_data = arithmetic_expressions(num_samples, min_length, max_length, numbers, modulus, operations, brackets)

# for sample in test_data:
#     print(f"Inputs : {sample['inputs']}")
#     print(f"Outputs: {sample['outputs']}")
#     print()
