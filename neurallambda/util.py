'''

Utility functions

'''



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
