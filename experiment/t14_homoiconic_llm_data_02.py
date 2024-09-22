'''

Generate a toy dataset of arithmetic problems. This version of the module makes
it compatible with the HF hub.

Example:

  a=1
  b=2
  c=a+b
  solve(c)

'''

import random
from dataclasses import dataclass
from typing import Union, List, Dict
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder
from typing import Union, Dict
import json
import matplotlib.pyplot as plt
from collections import Counter
import logging
import copy
import math

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
    vars = copy.copy(vars)
    random.shuffle(vars)
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


def puzzle_to_chunks(puzzle: Puzzle) -> List[str]:
    chunks = []
    for var, value in puzzle.variables.items():
        chunks.append(f"{var}={expression_to_str(value)}")
    chunks.append(f"solve({puzzle.solve})=")
    chunks.append(str(evaluate(Variable(puzzle.solve), puzzle.variables)))
    return chunks


def generate_dataset(num_examples: int, num_vars: int, nums: List[int], ops: List[str], filter_num_size=100) -> Dict[str, List[Dict]]:
    vars = [f'var_{i}' for i in range(num_vars)]
    dataset = set()
    last_unique_sample = 0
    pbar = tqdm(total=num_examples, desc=f"Generating dataset (vars: {num_vars})")

    while len(dataset) < num_examples and last_unique_sample < 100:
        puzzle = make_puzzle(vars, nums, ops)
        chunks = puzzle_to_chunks(puzzle)

        # Skip puzzles that result in answers that are too large
        if abs(int(chunks[-1])) > filter_num_size:
            continue

        example = (tuple(chunks[:-1]), chunks[-1])  # Make it hashable

        if example not in dataset:
            dataset.add(example)
            last_unique_sample = 0
            pbar.update(1)
        else:
            last_unique_sample += 1

    pbar.close()

    # Convert set back to list of dicts
    dataset = [{"input": list(ex[0]), "output": ex[1]} for ex in dataset]

    # Shuffle the dataset
    random.shuffle(dataset)

    # Split into train and test
    split_index = int(0.8 * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    return {"train": train_data, "test": test_data}


def save_dataset(dataset: List[Dict], filename: str):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


def upload_to_huggingface(dataset_name: str, dataset_files: Dict[str, str], readme: str):
    logging.info(f"Starting upload process for dataset: {dataset_name}")

    try:
        # Load the datasets
        datasets = {}
        for split, file in dataset_files.items():
            with open(file, 'r') as f:
                data = json.load(f)
            datasets[split] = Dataset.from_dict({
                "input": [example["input"] for example in data],
                "output": [example["output"] for example in data]
            })

        # Create a DatasetDict
        dataset_dict = DatasetDict(datasets)

        # Get the token
        token = HfFolder.get_token()
        if not token:
            raise ValueError("No Hugging Face token found. Please login using `huggingface-cli login` or set the HUGGINGFACE_TOKEN environment variable.")

        # Push to the Hugging Face Hub
        api = HfApi()
        api.create_repo(dataset_name, repo_type="dataset", exist_ok=True)

        dataset_dict.push_to_hub(dataset_name, token=token)

        # Update dataset metadata
        api.update_repo_visibility(dataset_name, private=False, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset",
            token=token
        )

        logging.info(f"Successfully uploaded dataset: {dataset_name}")
    except Exception as e:
        logging.error(f"Error uploading dataset {dataset_name}: {str(e)}")
        raise

def load_dataset_from_source(source: Union[str, Dict[str, str]], is_local: bool = False) -> DatasetDict:
    if is_local:
        if isinstance(source, str):
            # Load a single JSON file
            with open(source, 'r') as f:
                data = json.load(f)
            dataset = Dataset.from_dict({
                "input": [example["input"] for example in data],
                "output": [example["output"] for example in data]
            })
            return DatasetDict({"dataset": dataset})
        elif isinstance(source, dict):
            # Load multiple JSON files
            splits = {}
            for split_name, file_path in source.items():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                splits[split_name] = Dataset.from_dict({
                    "input": [example["input"] for example in data],
                    "output": [example["output"] for example in data]
                })
            return DatasetDict(splits)
        else:
            raise ValueError("For local files, source must be a string (single file) or a dictionary (multiple files)")
    else:
        # Load from Hugging Face Hub
        return load_dataset(source)


def generate_output_histograms(source: Union[str, Dict[str, str]], is_local: bool = False):
    # Load the dataset
    dataset = load_dataset_from_source(source, is_local)

    # Calculate the number of rows and columns needed
    num_splits = len(dataset)
    num_cols = min(3, num_splits)  # Max 3 columns
    num_rows = math.ceil(num_splits / num_cols)

    # Set up the plot
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8*num_cols, 6*num_rows))
    fig.suptitle('Output Frequency Histograms for Each Split', fontsize=16)

    # Ensure axs is always a 2D array
    if num_splits == 1:
        axs = np.array([[axs]])
    elif num_rows == 1:
        axs = axs.reshape(1, -1)

    # Generate histograms for each split
    for idx, (split_name, split_data) in enumerate(dataset.items()):
        # Get the outputs for this split and convert to integers
        outputs = [int(x) for x in split_data['output']]

        # Calculate row and column for the current subplot
        row = idx // num_cols
        col = idx % num_cols

        # Plot the histogram
        axs[row, col].hist(outputs, bins=50, edgecolor='black')
        axs[row, col].set_title(f'{split_name} Output Frequency')
        axs[row, col].set_xlabel('Output Value')
        axs[row, col].set_ylabel('Frequency')

    # Remove any unused subplots
    for idx in range(num_splits, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        fig.delaxes(axs[row, col])

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


def demonstrate_dataset(source: Union[str, Dict[str, str]], is_local: bool = False):
    # Load the dataset
    splits = load_dataset_from_source(source, is_local)

    print("Dataset structure:")
    print(splits)

    for split_name, split_data in splits.items():
        print(f"\nSamples from {split_name} split:")
        for i in range(3):  # Print 3 samples from each split
            if i < len(split_data):
                sample = split_data[i]
                print(f"Sample {i+1}:")
                print("Input:", sample['input'])
                print("Output:", sample['output'])
                print()
            else:
                print(f"Not enough samples in {split_name} split")



readme = '''
# Arithmetic Puzzles Dataset

A collection of arithmetic puzzles with heavy use of variable assignment. Current LLMs struggle with variable indirection/multi-hop reasoning, this should be a tough test for them.

Inputs are a list of strings representing variable assignments (`c=a+b`), and the output is the integer answer.

Outputs are filtered to be between [-100, 100], and self-reference/looped dependencies are forbidden.

Splits are named like:

- `train_N` 8k total examples of puzzles with N variables
- `test_N` 2k more examples with N variables

Train/test leakage is prevented: all training examples are filtered out of the test set.

Conceptually the data looks like this:

```python
Input:

  a=1
  b=2
  c=a+b
  solve(c)=

Output:
  3
```

In actuality it looks like this:

```python
{
    "input": ['var_0=1', 'var_1=2', 'var_2=a+b', 'solve(var_2)='],
    "output": 3
}
```


### Loading the Dataset

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("neurallambda/arithmetic_dataset")

# Load specific splits
train_small = load_dataset("neurallambda/arithmetic_dataset", split="train_10")
test_small = load_dataset("neurallambda/arithmetic_dataset", split="test_10")
```

### Preparing Inputs

To prepare the inputs as concatenated strings, you can do this:

```python
def prepare_input(example):
    return {
        "input_text": "\n".join(example["input"]),
        "output": example["output"]
    }

# Apply the preparation to a specific split
train_small_prepared = train_small.map(prepare_input)

# Example of using the prepared dataset
for example in train_small_prepared.select(range(5)):  # Show first 5 examples
    print("Input:", example["input_text"])
    print("Output:", example["output"])
    print()
```

This will produce output similar to:

```
Input: var_0=5
var_1=2
var_2=-2 + -8
var_3=3
var_4=4
var_5=var_2
var_6=var_3 * 10
var_7=var_2 - var_0
var_8=var_1
var_9=-2 - 9
solve(var_3)=
Output: 3
```
'''




if True:
    SEED = 152
    random.seed(SEED)

    dataset_name = "neurallambda/arithmetic_dataset"

    # Generate datasets
    nums = list(range(-10, 11))  # integer literals
    ops = ['+', '-', '*']

    n_samples = 10_000
    sizes = [2, 4, 6, 8, 10, 20, 30, 50, 100]

    local_files = {}

    for size in sizes:
        print(f"\nGenerating dataset for {size} variables:")
        dataset = generate_dataset(n_samples, size, nums, ops)

        # Save datasets locally
        train_filename = f'arithmetic_train_{size}.json'
        test_filename = f'arithmetic_test_{size}.json'

        save_dataset(dataset['train'], train_filename)
        save_dataset(dataset['test'], test_filename)

        local_files[f'train_{size}'] = train_filename
        local_files[f'test_{size}'] = test_filename

        print(f"Train set size: {len(dataset['train'])}")
        print(f"Test set size: {len(dataset['test'])}")

    print("\n" + "=" * 50 + "\n")

    # From local JSON files
    print("Loading from local JSON files:")
    print(local_files)

    # # Visualize
    # demonstrate_dataset(local_files, is_local=True)
    # generate_output_histograms(local_files, is_local=True)

    # Save to HF Hub
    upload_to_huggingface(dataset_name, local_files, readme)

    # Load and inspect

    # From Hugging Face Hub
    print("Loading from Hugging Face Hub:")
    demonstrate_dataset("neurallambda/arithmetic_dataset")
