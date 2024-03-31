'''

Utility functions

'''

from typing import Any, Iterable, List, Optional
import re

def transform_runs(input_list, is_equivalent, transform_func):
    """
    Transforms runs of equivalent elements in a list based on a given transformation function and an equivalence function.

    The function iterates through the input list, identifying 'runs' of elements. A 'run' is a sequence of elements
    where each element is considered equivalent to the next, as determined by the 'is_equivalent' function. Once a run
    is identified, it is transformed using the 'transform_func' and then added to the output list. Elements not part
    of a run are added to the output list as they are.

    Parameters:
    - input_list (list): The list to be analyzed and transformed. It can contain elements of any type.
    - transform_func (function): A function that takes a list (a run) and returns a transformed version of it.
      This function is applied to each identified run in 'input_list'.
    - is_equivalent (function): A function that takes two elements and returns True if they are considered equivalent,
      otherwise False. This function is used to identify runs in 'input_list'.

    Returns:
    - list: A new list containing the transformed runs and individual elements from the original list.

    Example Usage:
    >>> transform_runs([1, 1, 2, 3, 3, 3], lambda run: sum(run) if len(run) > 1 else run[0], lambda x, y: x == y)
    [2, 2, 9]  # Sums the runs of identical elements and leaves single elements as they are.
    """
    if not input_list:
        return []

    output_list = []
    run_start = 0

    for i in range(1, len(input_list)):
        if not is_equivalent(input_list[i], input_list[run_start]):
            # Run ends here; apply transform function to the run
            run = input_list[run_start:i]
            transformed_run = transform_func(run)
            output_list.append(transformed_run)
            run_start = i

    # Handle the last run
    final_run = input_list[run_start:]
    output_list.append(transform_func(final_run))

    return output_list


##################################################
# ANSI Colors/Formatting

def bold(text):
    return f"\033[1m{str(text)}\033[0m"

def italics(text):
    return f"\033[3m{str(text)}\033[0m"

def underline(text):
    return f"\033[4m{str(text)}\033[0m"

def blink(text):
    return f"\033[5m{str(text)}\033[0m"


##########
# Foreground

def red(text):
    return f"\033[31m{str(text)}\033[0m"

def green(text):
    return f"\033[32m{str(text)}\033[0m"

def yellow(text):
    return f"\033[33m{str(text)}\033[0m"

def blue(text):
    return f"\033[34m{str(text)}\033[0m"

def magenta(text):
    return f"\033[35m{str(text)}\033[0m"

def cyan(text):
    return f"\033[36m{str(text)}\033[0m"

def white(text):
    return f"\033[37m{str(text)}\033[0m"


##########
# Background

def bg_black(text):
    return f"\033[40m{str(text)}\033[0m"

def bg_red(text):
    return f"\033[41m{str(text)}\033[0m"

def bg_green(text):
    return f"\033[42m{str(text)}\033[0m"

def bg_yellow(text):
    return f"\033[43m{str(text)}\033[0m"

def bg_blue(text):
    return f"\033[44m{str(text)}\033[0m"

def bg_magenta(text):
    return f"\033[45m{str(text)}\033[0m"

def bg_cyan(text):
    return f"\033[46m{str(text)}\033[0m"

def bg_white(text):
    return f"\033[47m{str(text)}\033[0m"

def colored(x):
    # Define the ANSI escape codes for the desired colors
    BLACK = '\033[30m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    RESET = '\033[0m'  # Resets the color to default terminal color

    # Retrieve the single torch value
    if isinstance(x, tuple):
        string, value = x
    else:
        value = x.item()
        string = f'{value:>.1f}'

    # Determine the color based on the value
    if 0.0 <= value < 0.4:
        color = BLACK
    elif 0.4 <= value < 0.6:
        color = YELLOW
    else:
        color = RED
    return f'{color}{string}{RESET}'

def strip_ansi_codes(input_str):
    """
    Strips all ANSI codes from the given string.

    Parameters:
    - input_str: The string from which to strip ANSI codes.

    Returns:
    - A string with all ANSI codes removed.
    """
    # ANSI escape code pattern
    ansi_escape_pattern = re.compile(r'''
        \x1b  # ESC
        \[    # [
        [0-?]*  # 0-9: Parameters for ANSI code
        [ -/]*  # Intermediate bytes
        [@-~]  # Final byte
    ''', re.VERBOSE)
    return ansi_escape_pattern.sub('', input_str)


# @@@@@@@@@@

test_cases = [
    ("\x1b[31m1.2\x1b[0m", "1.2"),
    ("\x1b[1;32mHello\x1b[0m \x1b[34mWorld!\x1b[0m", "Hello World!"),
    ("Normal string", "Normal string"),
    ("\x1b[0;31;51mRed Background Text\x1b[0m", "Red Background Text"),
    ("", ""),
]

all_passed = True
for i, (test_input, expected_output) in enumerate(test_cases):
    result = strip_ansi_codes(test_input)
    assert result == expected_output, f"Test {i + 1}: Failed - Expected '{expected_output}', got '{result}'"

# @@@@@@@@@@


def format_number(num):
    """
    Formats a number with suffixes 'k', 'M', or 'B' for thousands, millions, and billions respectively.

    Parameters:
    - num (int): The number to format.

    Returns:
    - str: The formatted number as a string.
    """
    if abs(num) >= 1_000_000_000:  # Billion
        formatted_num = f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:  # Million
        formatted_num = f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:  # Thousand
        formatted_num = f"{num / 1_000:.1f}k"
    else:
        formatted_num = str(num)

    return formatted_num


##################################################
# Print Grids

def calculate_column_widths(row: Iterable[Iterable[Any]]) -> List[int]:
    """Calculate and return the maximum width needed for each column across all rows and sublists."""
    column_widths = []
    for column in row:
        for i, item in enumerate(column):
            item_length = len(strip_ansi_codes(str(item)))  # Use strip_ansi_codes to get the true length
            if len(column_widths) <= i:
                column_widths.append(item_length)
            else:
                column_widths[i] = max(column_widths[i], item_length)
    return column_widths

def print_row(row: Iterable[Iterable[Any]], column_widths: List[int], label: str, max_label_len: int) -> None:
    """Print a single row of data."""
    print(f"{label.rjust(max_label_len)}", end=" ")
    for i, item in enumerate(row):
        # Use strip_ansi_codes to ensure proper length calculation for justification
        item_str = str(item)
        stripped_item = strip_ansi_codes(item_str)
        formatted_item = f'{item:.2f}' if isinstance(item, float) else item_str
        padding = column_widths[i] - len(stripped_item) + len(item_str)
        print(f"{formatted_item.rjust(padding)}", end=" ")
    print()

def print_grid(data: Iterable[Iterable[Iterable[Any]]], labels: Optional[List[str]] = None) -> None:
    """Prints a grid of data with optional labels."""
    if labels:
        max_label_len = max(len(label) for label in labels)
    else:
        max_label_len = 0

    for row in data:
        column_widths = calculate_column_widths(row)
        for label, sub_row in zip(labels, row):
            print_row(sub_row, column_widths, label, max_label_len)
        print("-" * (sum(column_widths) + max_label_len + len(row) * 3))

# @@@@@@@@@@
if False:
    # Assuming ANSI coded strings or any data to demonstrate
    data = [
        [
            ["\033[94mItem1\033[0m", 200, 3.14],
            ["\033[92mItem2\033[0m", 150, 2.718]
        ],
        [
            ["\033[91mItem3\033[0m", 300, 1.618],
            ["\033[93mItem4\033[0m", 250, 0.577]
        ]
    ]
    labels = ["Label 1", "Label 2"]
    print_grid(data, labels)
# @@@@@@@@@@



##################################################
#

# NOTE: this is an idea for preloading weights with superpositions of symbols

# def generate_combinations(elements, max_length):
#     """
#     Generate all possible combinations of the elements up to a specified maximum length.

#     :param elements: A list of elements to combine.
#     :param max_length: The maximum length of the combinations.

#     Example sizes of combos:

#         len(generate_combinations(range(2), 2))  ==  3
#         len(generate_combinations(range(3), 2))  ==  6
#         len(generate_combinations(range(3), 3))  ==  7
#         len(generate_combinations(range(4), 2))  ==  10
#         len(generate_combinations(range(4), 3))  ==  14
#         len(generate_combinations(range(4), 4))  ==  15
#         len(generate_combinations(range(5), 2))  ==  15
#         len(generate_combinations(range(5), 3))  ==  25
#         len(generate_combinations(range(5), 4))  ==  30
#         len(generate_combinations(range(5), 5))  ==  31
#     """
#     # Store all combinations in a list
#     all_combinations = []

#     # Generate combinations for every length up to max_length
#     for length in range(1, max_length + 1):
#         # itertools.combinations generates combinations of the current length
#         combinations = itertools.combinations(elements, length)
#         # Add the current combinations to the total list
#         all_combinations.extend(combinations)

#     return all_combinations

# # Test the function
# elements = ['a', 'b', 'c', 'd', 'e']  # A list of length 5
# max_length = 3  # Generate combinations up to length 3

# # Generate and print all combinations
# combinations = generate_combinations(elements, max_length)
# for combo in combinations:
#     print(combo)
