
from neurallambda.lab.datasets import *




def test_evaluate_arithmetic():
    ops1 = [{'symbol': '+', 'priority': 1}, {'symbol': '-', 'priority': 1}, {'symbol': '*', 'priority': 1}]
    ops2 = [{'symbol': '+', 'priority': 1}, {'symbol': '-', 'priority': 1}, {'symbol': '*', 'priority': 2}]
    test_cases = [
        (["7"], 7, ops1),
        (["2", "+", "3"], 5, ops1),
        (["1", "-", "4"], 7, ops1),
        (["4", "*", "6"], 4, ops1),
        (["4", "+", "9", "*", "2"], 6, ops1),
        (["1", "-", "4", "*", "2"], 4, ops1),

        (["7"], 7, ops2),
        (["2", "+", "3"], 5, ops2),
        (["1", "-", "4"], 7, ops2),
        (["4", "*", "6"], 4, ops2),
        (["4", "+", "9", "*", "2"], 2, ops2),
        (["1", "-", "4", "*", "2"], 3, ops2),
    ]

    for tokens, expected_output, operations in test_cases:
        assert evaluate_arithmetic(tokens, operations, 10) == expected_output, f"Test case failed: {' '.join(tokens)}"
