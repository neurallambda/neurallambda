'''

Sliding-window-add-and-modulo Game

----------
HOW THE GAME WORKS:

Window=2 Example Dataset
1 2 <- seed
1 2 3 5 8 3 [1 4] _ _ _ _

Window=3 Example Dataset
1 2 3 <- seed
1 2 3 6 1 0 7 8 [5 0 3] _ _ _ _


----------
LVL1: All data is generated with locked window size.

LVL2: Agent has to infer the correct window size.

'''

def generate_sequence(seed, window_size, sequence_length):
    """
    Generates a sequence using a seed and a sliding window.

    The function starts with a seed, which is a list of initial numbers.
    It then extends this seed into a longer sequence by repeatedly summing the last 'window_size' numbers,
    and adding this sum (modulo 10) to the sequence.

    Args:
    seed (list): The initial numbers of the sequence.
    window_size (int): The number of elements to sum in each step.
    sequence_length (int): The total length of the sequence to generate.

    Returns:
    list: The generated sequence with the specified length.
    """
    sequence = seed.copy()
    while len(sequence) < sequence_length:
        window_sum = sum(sequence[-window_size:]) % 10
        sequence.append(window_sum)
    return sequence

# Test 1: Window Size = 2
assert generate_sequence([1, 2], 2, sequence_length=10) == [1, 2, 3, 5, 8, 3, 1, 4, 5, 9]

# Test 2: Window Size = 3
assert generate_sequence([1, 2, 3], 3, sequence_length=15) == [1, 2, 3, 6, 1, 0, 7, 8, 5, 0, 3, 8, 1, 2, 1]

##########

import random
from datasets import Dataset

def generate_synthetic_data(num_samples, max_window_size, max_sequence_length):
    data = []
    for _ in range(num_samples):
        window_size = random.randint(2, max_window_size)
        sequence_length = random.randint(window_size+1, max_sequence_length)
        seed = [random.randint(0, 9) for _ in range(window_size)]
        sequence = generate_sequence(seed, window_size, sequence_length)
        data.append({
            'seed': seed,
            'window_size': window_size,
            'sequence': sequence
        })
    return data

# Parameters for dataset generation
num_samples = 1000  # total number of samples in the dataset
max_window_size = 5  # maximum window size
max_sequence_length = 20  # maximum length of the sequence

# Generate the synthetic dataset
synthetic_data = generate_synthetic_data(num_samples, max_window_size, max_sequence_length)

# Splitting the dataset into train and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(synthetic_data))
train_data = synthetic_data[:train_size]
val_data = synthetic_data[train_size:]

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset   = Dataset.from_list(val_data)
