'''

Different datasets to use in tests

'''

import random
from typing import Dict, List, Tuple, Union, Any

BOS_SYMBOL = '^'
PAUSE_SYMBOL = '.'
REFLECT_SYMBOL = '|'


##################################################
# Seq2Seq, transduction (ie not autoregressive, ie has separate inputs and outputs)
#   INPUT : input + pad
#   OUTPUT: pad   + output

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
        accuracy_mask = [0] * (hlength + 2) + [1] * hlength
        # convert all symbols to str
        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'inputs': inputs,
            'outputs': outputs,
            'accuracy_mask': accuracy_mask,
        })
    return data


# print()
# test_data = palindrome(5, 10, 15, ['a', 'b', 'c'])
# for x in test_data:
#     print_grid([[ ['inputs:'] + x['inputs'],
#                  ['outputs:'] + x['outputs'],
#                  ['acc mask:'] + x['accuracy_mask']
#                 ]], ['', '', ''])



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
        accuracy_mask = [0] * length + [1]

        # convert all symbols to str
        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))

        data.append({
            'inputs': inputs,
            'outputs': outputs,
            'accuracy_mask': accuracy_mask,
        })

    return data

# print()
# test_data = even_pairs(5, 10, 15, ['a', 'b', 'c'])
# for x in test_data:
#     print_grid([[ ['inputs:'] + x['inputs'],
#                  ['outputs:'] + x['outputs'],
#                  ['acc mask:'] + x['accuracy_mask']
#                 ]], ['', '', ''])


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

        inputs, outputs, accuracy_mask = (  # depend on each others' lengths. also add BOS.
            [BOS_SYMBOL] + inputs + [PAUSE_SYMBOL] * len(outputs),
            [PAUSE_SYMBOL] + [PAUSE_SYMBOL] * len(inputs) + outputs,
            [0] * (len(inputs) + 1)  + [1] * len(outputs)
        )

        test_data.append({
            'inputs': inputs,
            'outputs': outputs,
            'accuracy_mask': accuracy_mask,
        })

    return test_data

# # Example usage
# num_samples = 5
# min_value = 0
# max_value = 255
# test_data = binary_arithmetic(num_samples, min_value, max_value, op='*')

# from neurallambda.util import print_grid

# print()
# for x in test_data:
#     print_grid([[ ['inputs:'] + x['inputs'],
#                  ['outputs:'] + x['outputs'],
#                  ['acc mask:'] + x['accuracy_mask']
#                 ]], ['', '', ''])


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
        inputs, outputs, accuracy_mask = (
            [BOS_SYMBOL] + inputs + [PAUSE_SYMBOL] * len(outputs),
            [PAUSE_SYMBOL] + [PAUSE_SYMBOL] * len(inputs) + outputs,
            [0] * (len(inputs) + 1) + [1] * len(outputs)
        )

        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'inputs': inputs,
            'outputs': outputs,
            'accuracy_mask': accuracy_mask,
        })

    return data


# print()
# test_data = arithmetic_expressions(5, 10, 15, [0,1,2,3,5], 5, ['+'], [('(', ')')])
# for x in test_data:
#     print_grid([[ ['inputs:'] + x['inputs'],
#                  ['outputs:'] + x['outputs'],
#                  ['acc mask:'] + x['accuracy_mask']
#                 ]], ['', '', ''])


##################################################
# Seq2Seq, transduction (ie not autoregressive, ie has separate inputs and outputs)
#   INPUT : input (no padding L or R)
#   OUTPUT: output (no padding L or R)

def swap_max_and_min(num_samples,
                     min_length,
                     max_length,
                     lang,
                     mask_type='swapped',
                     sample_with_replacement=True) -> Dict[str, List[str]]:
    ''' x: a sequence of tokens that have an ordering, but are unordered, eg [9, 0, 3, 1]
        y: a sequence where the max and min have been swapped
        accuracy_mask: a mask indicating which positions are correct '''
    assert mask_type in {'all', 'swapped'}
    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        if sample_with_replacement:
            seq = [random.choice(lang) for _ in range(length)]
        else:
            seq = random.sample(lang, length)

        inputs = seq
        outputs = seq.copy()

        # find the maximum and minimum values in the sequence
        max_val = max(outputs)
        min_val = min(outputs)

        # swap the maximum and minimum values in the output sequence
        max_index = outputs.index(max_val)
        min_index = outputs.index(min_val)
        outputs[max_index] = min_val
        outputs[min_index] = max_val

        # generate accuracy mask based on the mask_type argument
        if mask_type == "all":
            accuracy_mask = [1] * length
        elif mask_type == "swapped":
            accuracy_mask = [1 if i in [max_index, min_index] else 0 for i in range(length)]
        else:
            raise ValueError(f"Invalid mask_type: {mask_type}")

        # convert all symbols to str
        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'inputs': inputs,
            'outputs': outputs,
            'accuracy_mask': accuracy_mask,
        })
    return data


# # Example usage
# # lang = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
# toy_data_all = swap_max_and_min(num_samples=5, min_length=3, max_length=7, lang=lang, mask_type="all")
# print("Toy data with all positions marked as correct:")
# print(toy_data_all)

# toy_data_swapped = swap_max_and_min(num_samples=5, min_length=3, max_length=7, lang=lang, mask_type="swapped")
# print("\nToy data with only swapped positions marked as correct:")
# print(toy_data_swapped)


# START_BLOCK_4

# import torch
# import torch.nn as nn

# def selection_sort(arr):
#     n = len(arr)

#     for i in range(n):
#         # Find the minimum element in the unsorted portion
#         min_idx = i
#         for j in range(i + 1, n):
#             if arr[j] < arr[min_idx]:
#                 min_idx = j

#         # Swap the minimum element with the first element of the unsorted portion
#         arr[i], arr[min_idx] = arr[min_idx], arr[i]

#     return arr


# def one_pass_sort(arr):
#     n = len(arr)
#     first_unsorted_idx = None
#     swap_idx = None

#     # Find the first unsorted index
#     for i in range(n - 1):
#         if arr[i] > arr[i + 1]:
#             first_unsorted_idx = i + 1
#             break

#     if first_unsorted_idx is None:
#         # The array is already sorted
#         return None, None, arr

#     # Find the index of the element to swap with
#     for j in range(first_unsorted_idx):
#         if arr[j] > arr[first_unsorted_idx]:
#             swap_idx = j
#             break

#     if swap_idx is None:
#         # The element at first_unsorted_idx is smaller than all previous elements
#         swap_idx = first_unsorted_idx - 1

#     # Perform the swap
#     arr[swap_idx], arr[first_unsorted_idx] = arr[first_unsorted_idx], arr[swap_idx]

#     return first_unsorted_idx, swap_idx, arr

# x = [1,3,5,3,7,8,9,0]
# x = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# # print(selection_sort(x))

# for _ in range(43):
#     i,j,x = one_pass_sort(x)
#     print(i, j, x)


# def inversion_count(lst):
#     ''' check 'sortedness' of list '''
#     n = len(lst)
#     # M = (lst.unsqueeze(1) < lst.unsqueeze(0)).float()
#     M = (lst.unsqueeze(0) - lst.unsqueeze(1)).relu()
#     L = torch.tril(torch.ones(n, n), diagonal=-1)
#     P = M * L
#     inversion_count = torch.sum(P)
#     return inversion_count

# # def differentiable_measure_of_disorder(lst):
# #     sorted_indices = torch.argsort(lst)
# #     original_indices = torch.arange(len(lst))
# #     disorder = torch.sum(torch.abs(sorted_indices - original_indices))
# #     return disorder

# # Example lists as tensors
# lists = [
#     nn.Parameter(torch.tensor([1., 2, 3, 4, 5])),
#     nn.Parameter(torch.tensor([5., 4, 3, 2, 1])),
#     nn.Parameter(torch.tensor([3., 1, 4, 2, 5])),
#     nn.Parameter(torch.tensor([2., 4, 1, 5, 3]))
# ]

# # Calculate and print the inversion count for each list
# for i, lst in enumerate(lists, start=1):
#     ic = inversion_count(lst)
#     print(f"List {i}: {lst.tolist()}")
#     print(f"Inversion Count {i}: {ic.item()}")
#     print()

# # END_BLOCK_4



# START_BLOCK_5
def insert_min_into_sorted_prefix(
    num_samples: int,
    min_length: int,
    max_length: int,
    lang: List[Any],
    mask_type: str = 'inserted',
    sample_with_replacement: bool = True
) -> List[Dict[str, List[str]]]:
    assert mask_type in {'all', 'inserted'}
    assert min_length <= max_length
    if not sample_with_replacement:
        assert len(lang) >= max_length

    sorted_lang = sorted(set(lang))

    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        seq = random.choices(sorted_lang, k=length) if sample_with_replacement else random.sample(sorted_lang, k=length)

        sorted_prefix_length = random.randint(1, length - 1)
        sorted_prefix = sorted(seq[:sorted_prefix_length])

        # Ensure the first unsorted element is smaller than the last sorted element
        unsorted_portion = seq[sorted_prefix_length:]
        smaller_elements = [x for x in sorted_lang if x < sorted_prefix[-1]]
        if smaller_elements:
            unsorted_portion[0] = random.choice(smaller_elements)

        inputs = sorted_prefix + unsorted_portion

        # Generate outputs
        min_val = min(unsorted_portion)
        min_index = inputs.index(min_val, sorted_prefix_length)
        insertion_index = next((i for i in range(sorted_prefix_length) if min_val < inputs[i]), sorted_prefix_length)

        outputs = inputs.copy()
        outputs.insert(insertion_index, min_val)
        outputs.pop(min_index + 1)

        accuracy_mask = [1] * length if mask_type == "all" else [int(i == insertion_index) for i in range(length)]

        data.append({
            'inputs': list(map(str, inputs)),
            'outputs': list(map(str, outputs)),
            'accuracy_mask': accuracy_mask,
        })
    return data


# def insert_min_into_sorted_prefix(num_samples,
#                                   min_length,
#                                   max_length,
#                                   lang,
#                                   mask_type='inserted',
#                                   sample_with_replacement=True) -> Dict[str, List[str]]:
#     '''

#     Returns:
#       x: a sequence of tokens that have an ordering, but are unordered, eg [0, 1, 3, 9]
#       y: a sequence where the minimum value from the unsorted half is inserted into the sorted prefix
#       accuracy_mask: a mask indicating which positions are correct

#     NOTE: this insertion index is exponentially biased towards lower positions,
#     so, very unbalanced dataset

#     '''
#     assert mask_type in {'all', 'inserted'}
#     data = []
#     for _ in range(num_samples):
#         length = random.randint(min_length, max_length)
#         if sample_with_replacement:
#             seq = [random.choice(lang) for _ in range(length)]
#         else:
#             seq = random.sample(lang, length)

#         inputs = seq

#         # generate a random length for the sorted prefix
#         sorted_prefix_length = random.randint(1, length)

#         # sort the prefix of the sequence
#         sorted_prefix = sorted(inputs[:sorted_prefix_length])
#         inputs[:sorted_prefix_length] = sorted_prefix


#         outputs = seq.copy()

#         # find the minimum value in the unsorted half
#         if sorted_prefix_length < length:
#             min_val = min(outputs[sorted_prefix_length:])
#             min_index = outputs.index(min_val, sorted_prefix_length)

#             # insert the minimum value into the sorted prefix
#             insertion_index = sorted_prefix_length
#             while insertion_index > 0 and min_val < outputs[insertion_index - 1]:
#                 insertion_index -= 1
#             outputs.insert(insertion_index, min_val)
#             outputs.pop(min_index + 1)  # remove the original minimum value
#         else:
#             insertion_index = length

#         # generate accuracy mask based on the mask_type argument
#         if mask_type == "all":
#             accuracy_mask = [1] * length
#         elif mask_type == "inserted":
#             accuracy_mask = [1 if i == insertion_index else 0 for i in range(length)]
#         else:
#             raise ValueError(f"Invalid mask_type: {mask_type}")

#         # convert all symbols to str
#         inputs = list(map(str, inputs))
#         outputs = list(map(str, outputs))
#         data.append({
#             'inputs': inputs,
#             'outputs': outputs,
#             'accuracy_mask': accuracy_mask,
#         })
#     return data



# xs = insert_min_into_sorted_prefix(num_samples=1000,
#                                    min_length=10,
#                                    max_length=10,
#                                    lang='0 1 2 3 4 5 6 7 8 9'.split(' '),
#                                    mask_type='all',
#                                    sample_with_replacement=False)
# # print()
# # for x in xs:
# #     print(x)


# END_BLOCK_5
