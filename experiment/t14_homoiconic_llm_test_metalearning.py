'''

Test Metalearning on a smaller simpler model.

----------
TAKEAWAYS:
  update_lors: dramatically simplified
  select_spans: dramatically simpler compared to select_ixs

----------
PROBLEM EXAMPLE:

1. Phase 1

a=1
b=2
c=3

2. Phase 2

solve(c)=3
solve(a)=1
solve(b)=2

Crucially, phase 2 cannot attend tokens from phase 1. It must rely on information saved into the weights during phase 1.

----------

TODO:
- [X] make data
- [X] build dataset. combine data as [Text+Meta, Text+Meta, ...]
- [X] init model
- [X] new parse function (spans)
- [X] fix `update_lors`

- [ ] run_epoch
- [ ] metaweights will be randomly initialized (not tokens anymore)
- [ ] project lors
- [ ] metalearn lors
- [ ] phase 2 tests

'''

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # let cuda give debug info

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

from typing import Optional, Tuple, Union, List, Dict, Any
import warnings
import math

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
import warnings
import itertools

from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from itertools import product

current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = SummaryWriter(f'log/t14_homoiconic_llm_test_metalearning/{current_time}')
LOG = True

try:
    importlib.reload(Q)
    print('RELOADING Q')
except NameError:
    import t14_homoiconic_llm_model_03 as Q


SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

# # Extra determinism
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

DEVICE = 'cuda:1'
WHICH_LOR = 1


##################################################
# Load Model

model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-7B")

def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        isnan = torch.isnan(output).any()
        isinf = torch.isinf(output).any()
        if isnan or isinf:
            print(f"NaN or inf detected in {module.__class__.__name__}: {isnan=}, {isinf=}")
            print(f"Module: {module}")
            print(f"Module name: {module_names.get(module, 'Unknown')}")
            print(f"Input shape: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}")
            print(f"Output shape: {output.shape}")
            breakpoint()
            # raise RuntimeError("NaN or inf detected: {isnan=}, {isinf=}")

module_names = {}

def add_hooks(model):
    for name, module in model.named_modules():
        module_names[module] = name
        module.register_forward_hook(hook_fn)

try:
    already_loaded
except:
    print('Loading model')
    model = Q.Qwen2ForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.bfloat16,  # HERE BE DRAGONS
        torch_dtype=torch.float32,
        # torch_dtype=torch.float64,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    num_layers = model.config.num_hidden_layers

    add_hooks(model)
    already_loaded = True


##################################################
# Parsing/Handling LORS

def update_lors(
        lor_models,  # shaped like `empty_lors`. Contains lor_proj, and per-head models
        lor_cache,  # shaped like `empty_lors`. Contains previously parsed lors
        lor_ix_spans: List[torch.Tensor],  # List[spans], where a span is [batch, (start_ix, end_ix)]
        hidden_states,  # from out.hidden_states, so, contains all layers
        num_layers,
):
    ''' Update the LORs by interpreting hidden states as new lor blocks '''
    lor_keys = lor_cache.keys()
    for k in lor_keys:
        assert (
            len(lor_models[k]) ==  # one (optional) lor module per layer
            len(lor_cache[k]) ==  # cache per layer
            lor_ix_spans.shape[1]
        ), (f'''
{len(lor_models[k])=} ==  # one (optional) lor module per layer
{len(lor_cache[k])=} ==  # cache per layer
{lor_ix_spans.shape[1]=} ==  # lor ix spans
''')

    h_emb = hidden_states[-1]  # final layer states

    per_layer_parses = []

    # iterate over all layers
    for layer_ix in range(num_layers):
        # skip non-lor'd layers
        if lor_models['lor_proj'][layer_ix] is None:
            assert lor_ix_spans[layer_ix] is None, f'lor_proj is defined for layer {layer_ix}, but there are no lor_ix_spans parse ixs'
            continue

        parses = per_layer_parses.append(select_spans(h_emb, lor_ix_spans[layer_ix]))

        # no parses implied anywhere
        if (parses > -1).sum() == 0:
            continue

        # run lor_proj. Returns tuple of L and R singular values, per key, eg: (lor_qs_l, lor_qs_r, ...)
        projs = lor_models['lor_proj'][layer_ix](parses)
        proj_pairs = zip(projs[::2], projs[1::2])

        # update cache
        for k, (l, r) in zip(lor_keys, proj_pairs):
            if lor_cache['lor_qs'][layer_ix] is None:  # is first pass, no cache yet
                lor_cache[k][layer_ix] = (l.unsqueeze(2), r.unsqueeze(2))  # [B, DIM, RANK]
            else:
                lor_cache[k][layer_ix] = (torch.cat([lor_cache[k][layer_ix][0], l.unsqueeze(2)], dim=2),
                                          torch.cat([lor_cache[k][layer_ix][1], r.unsqueeze(2)], dim=2))  # [B, DIM, RANK]

    return lor_cache


def select_spans(x, indices):
    """Selects spans from a 3D tensor (`[batch, seq, dim]`) along dim=1 using
    provided start and end indices.

    Perform span selection on a 3D tensor. If `indices` contains [-1, -1] for a
    batch, that location will be filled with 0s.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, seq, dim] where:
            batch: number of sequences in the batch
            seq: length of each sequence
            dim: dimensionality of each token representation

        indices (torch.Tensor): 2D tensor of indices for span selection, with shape
            [batch, 2]. Each row contains [start, end] indices for the span.
            Start and end are inclusive. If a row is [-1, -1], the corresponding
            output will be filled with 0s.

    Returns:
        torch.Tensor: Output tensor of shape [batch, max_span_length, dim]

    Example:
        >>> x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        ...                   [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]])
        >>> indices = torch.tensor([[1, 2], [-1, -1]])
        >>> select_spans(x, indices)
        tensor([[[ 4,  5,  6],
                 [ 7,  8,  9],
                 [ 0,  0,  0]],
                [[ 0,  0,  0],
                 [ 0,  0,  0],
                 [ 0,  0,  0]]])
    """
    B, S, D = x.shape  # batch, sequence length, dimension

    # Create a mask for valid spans (not [-1, -1])
    mask = (indices[:, 0] != -1)  # we assume -1s are paired correctly with another -1

    # Calculate span lengths
    span_lengths = torch.where(mask, indices[:, 1] - indices[:, 0] + 1, torch.zeros_like(indices[:, 0]))
    max_span_length = span_lengths.max().item()

    # Create position indices for each element in the max span
    positions = torch.arange(max_span_length, device=x.device).unsqueeze(0).expand(B, -1)

    # Calculate absolute indices for each position
    abs_indices = indices[:, 0].unsqueeze(1) + positions

    # Create a mask for valid positions within each span
    valid_positions = positions < span_lengths.unsqueeze(1)

    # Combine the span mask and position mask
    final_mask = mask.unsqueeze(1) & valid_positions

    # Create batch indices
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, max_span_length)

    # Gather values using absolute indices, with out-of-bounds handling
    gathered_values = torch.zeros((B, max_span_length, D), device=x.device, dtype=x.dtype)
    valid_abs_indices = abs_indices[final_mask]
    valid_batch_indices = batch_indices[final_mask]
    gathered_values[final_mask] = x[valid_batch_indices, valid_abs_indices]

    return gathered_values


# @@@@@@@@@@
# Hand check
if False:
    # [batch, seq, dim]
    x = torch.tensor(
        [[[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12]],
         [[13, 14, 15],
          [16, 17, 18],
          [19, 20, 21],
          [22, 23, 24]],
         [[25, 26, 27],
          [28, 29, 30],
          [31, 32, 33],
          [34, 35, 36]]
         ]
    )
    indices = torch.tensor([
        [1, 2],
        [-1, -1],
        [0, 3],
    ])
    result = select_spans(x, indices)
    print(result)
# @@@@@@@@@@


# @@@@@@@@@@
# Test `select_spans`

def slow_select_spans(x, indices):
    """A slow, loop-based reference implementation of span selection."""
    B, S, D = x.shape

    # Calculate the actual maximum span length across valid spans
    max_span_length = 0
    for b in range(B):
        start, end = indices[b]
        if start >= 0 and end >= 0 and start <= end and end < S:
            max_span_length = max(max_span_length, end - start + 1)

    result = torch.zeros((B, max_span_length, D), dtype=x.dtype, device=x.device)

    for b in range(B):
        start, end = indices[b]
        if start >= 0 and end >= 0 and start <= end and end < S:
            span_length = end - start + 1
            result[b, :span_length] = x[b, start:end + 1]

    return result

def test_select_spans(num_tests=100, max_batch=10, max_seq=20, max_dim=5):
    """Test the select_spans function against a slow implementation with random inputs."""
    for _ in range(num_tests):
        # Generate random dimensions
        B = random.randint(1, max_batch)
        S = random.randint(1, max_seq)
        D = random.randint(1, max_dim)

        # Generate random input tensor
        x = torch.randint(0, 100, (B, S, D))

        # Generate random indices
        if S == 1:
            indices = torch.zeros((B, 2), dtype=torch.long)
        else:
            indices = torch.randint(0, S - 1, (B, 2))
        # Ensure start <= end for valid spans
        indices = torch.sort(indices, dim=1).values
        # Randomly set some spans to [-1, -1]
        mask = torch.rand(B) < 0.2
        indices[mask] = -1

        # Run both implementations
        fast_result = select_spans(x, indices)
        slow_result = slow_select_spans(x, indices)

        # Check if results match
        assert torch.all(fast_result == slow_result), f"Mismatch found in test {_+1}"

        # Additional checks
        for b in range(B):
            start, end = indices[b]
            if start == -1 and end == -1:
                assert torch.all(fast_result[b] == 0), f"Non-zero values for [-1, -1] in batch {b}, test {_+1}"
            elif 0 <= start <= end < S:
                span_length = end - start + 1
                assert torch.all(fast_result[b, :span_length] == x[b, start:end + 1]), \
                    f"Incorrect span selection in batch {b}, test {_+1}"
                assert torch.all(fast_result[b, span_length:] == 0), \
                    f"Non-zero padding in batch {b}, test {_+1}"

    print(f"All {num_tests} tests passed successfully!")

if False:
    test_select_spans()

# @@@@@@@@@@


##################################################
# Data

# START_BLOCK_1

def check_tokenization(tokenizer, variables, ldelim, rdelim):
    ''' check that var gets tokenized on its own, and not combined with a delimiter '''
    failed_variables = []

    for var in variables:
        wrapped_var = f"{ldelim}{var}{rdelim}"
        tokens = tokenizer.tokenize(wrapped_var)

        if len(tokens) != 3:
            failed_variables.append((var, len(tokens), tokens))

    if failed_variables:
        print("The following variables did not tokenize into exactly 3 tokens:")
        for var, token_count, decoded in failed_variables:
            print(f"Variable: '{var}', Token count: {token_count}, {decoded=}")
        raise ValueError(f"didn't tokenize properly with {ldelim=} {rdelim=}")
    else:
        print(f"All variables tokenized into exactly 3 tokens when wrapped in {ldelim=} {rdelim=}.")


def check_clause_tokenization(tokenizer, variables, values):
    ''' check that "var=val" tokenizes the same as doing "var", "=", "val" each separately '''
    mismatched_clauses = []

    for var, val in product(variables, values):
        # Tokenize parts separately
        var_tokens = tokenizer.tokenize(var)
        equals_tokens = tokenizer.tokenize("=")
        val_tokens = tokenizer.tokenize(val)

        # Combine separate tokenizations
        separate_tokens = var_tokens + equals_tokens + val_tokens

        # Tokenize the full clause
        full_clause = f"{var}={val}"
        full_tokens = tokenizer.tokenize(full_clause)

        # Check if tokens are equivalent
        if separate_tokens != full_tokens:
            mismatched_clauses.append((var, val, separate_tokens, full_tokens))

    if mismatched_clauses:
        print("The following clauses have mismatched tokenizations:")
        for var, val, separate_tokens, full_tokens in mismatched_clauses:
            print(f"Clause: '{var}={val}'")
            print(f"  Separate tokenization: {separate_tokens}")
            print(f"  Full clause tokenization: {full_tokens}")
        raise ValueError("clauses didn't tokenize properly")
    else:
        print("All clauses have matching tokenizations when tokenized separately and as a whole.")


variables = (
    'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ') +
    'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' ')
)
values = [str(x) for x in range(0, 100)]
ldelim = '~'
rdelim = '~'

# Check that things tokenize as expected.
if True:
    check_tokenization(tokenizer, variables, ldelim, rdelim)
    check_clause_tokenization(tokenizer, variables, values)

# END_BLOCK_1



def make_puzzle(variables, values, n):
    '''Make a "puzzle" where random variables are set to random values, and then
the challenge is remebering which variable was assigned to which value, tested
in a random order, eg for n=3:

# inputs
"""a=1
b=2
c=3"""

# challenge
[
  "solve(c)=3",
  "solve(a)=1",
  "solve(b)=2",
]
'''
    # Ensure n is not greater than the number of variables or values
    n = min(n, len(variables), len(values))

    # Randomly select n variables and values
    selected_vars = random.sample(variables, n)
    selected_vals = random.sample(values, n)

    # Create the input string
    inputs = "\n".join(f"{var}={val}" for var, val in zip(selected_vars, selected_vals))

    # Create the challenge string
    challenge_pairs = list(zip(selected_vars, selected_vals))
    random.shuffle(challenge_pairs)
    challenge = [(f"solve{ldelim}{var}{rdelim}=", f"{val}") for var, val in challenge_pairs]

    return (inputs, challenge)

if False:
    inputs, challenge = make_puzzle(variables, values, 3)
    print("Inputs:")
    print(inputs)
    print("\nChallenge:")
    print(challenge)




class PuzzleDataset(Dataset):
    def __init__(self, variables, values, n_samples, n_variables):
        self.variables = variables
        self.values = values
        self.n_samples = n_samples
        self.n_variables = n_variables
        self.data = [make_puzzle(variables, values, n_variables) for _ in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]



def left_pad_sequence(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        padding = torch.full((max_len - len(seq),), padding_value, dtype=seq.dtype)
        padded_seqs.append(torch.cat((padding, seq)))
    return torch.stack(padded_seqs)



def collate_fn(batch: List[Tuple[str, List[Tuple[str, str]]]], tokenizer):
    inputs, challenges = zip(*batch)

    # Process inputs
    input_encoded = tokenizer(list(inputs), padding='max_length', truncation=True, return_tensors='pt')
    input_ids = input_encoded['input_ids']
    input_attention_mask = input_encoded['attention_mask']

    # Process challenges
    challenge_input_ids = []
    challenge_attention_masks = []
    loss_masks = []

    for challenge in challenges:
        challenge_tokens = []
        loss_mask = []

        for q, a in challenge:
            q_tokens = tokenizer.encode(q, add_special_tokens=False)
            a_tokens = tokenizer.encode(a, add_special_tokens=False)

            challenge_tokens.extend(q_tokens + a_tokens)
            loss_mask.extend([0] * len(q_tokens) + [1] * len(a_tokens))

            # Add space token between QA pairs, except for the last pair
            if q != challenge[-1][0]:  # If not the last question
                challenge_tokens.append(tokenizer.encode(' ', add_special_tokens=False)[0])
                loss_mask.append(0)

        challenge_input_ids.append(torch.tensor(challenge_tokens))
        challenge_attention_masks.append(torch.ones(len(challenge_tokens)))
        loss_masks.append(torch.tensor(loss_mask))

    # Left pad sequences
    challenge_input_ids = left_pad_sequence(challenge_input_ids, padding_value=tokenizer.pad_token_id)
    challenge_attention_masks = left_pad_sequence(challenge_attention_masks, padding_value=0)
    loss_masks = left_pad_sequence(loss_masks, padding_value=0)

    return {
        'input_ids': input_ids,
        'input_attention_mask': input_attention_mask,
        'challenge_input_ids': challenge_input_ids,
        'challenge_attention_mask': challenge_attention_masks,
        'loss_mask': loss_masks
    }

batch_size = 32
n_samples = 1000
n_variables = 3

dataset = PuzzleDataset(variables, values, n_samples=n_samples, n_variables=n_variables)

# Assuming you have a tokenizer defined, e.g., from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer)
)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Check in on the dataset

if True:
    def visualize_sample(tokenizer, input_ids, attention_mask, loss_mask=None):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Remove padding tokens from the beginning
        non_pad_start = attention_mask.nonzero()[0].item()
        tokens = tokens[non_pad_start:]
        input_ids = input_ids[non_pad_start:]
        attention_mask = attention_mask[non_pad_start:]
        if loss_mask is not None:
            loss_mask = loss_mask[non_pad_start:]

        # Prepare rows
        token_row = "Tokens:      " + " ".join(f"{t:<6}" for t in tokens)
        input_id_row = "Input IDs:   " + " ".join(f"{id:<6}" for id in input_ids)
        attn_mask_row = "Attn Mask:   " + " ".join(f"{m:<6}" for m in attention_mask)

        # Print rows
        print(token_row)
        print(input_id_row)
        print(attn_mask_row)

        if loss_mask is not None:
            loss_mask_row = "Loss Mask:   " + " ".join(f"{m:<6}" for m in loss_mask)
            print(loss_mask_row)

    # check some of the dataloader
    for batch in dataloader:
        print("Batch shapes:")
        print("  input_ids:", batch['input_ids'].shape)
        print("  input_attention_mask:", batch['input_attention_mask'].shape)
        print("  challenge_input_ids:", batch['challenge_input_ids'].shape)
        print("  challenge_attention_mask:", batch['challenge_attention_mask'].shape)
        print("  loss_mask:", batch['loss_mask'].shape)

        # Verify that there are 1s at the end of each loss mask
        last_ones = batch['loss_mask'].sum(dim=1) - batch['loss_mask'].flip(dims=[1]).cumsum(dim=1).argmax(dim=1) - 1
        print("  Position of last 1 in each loss mask:", last_ones)
        print("  All loss masks have 1s at the end:", (last_ones == (batch['challenge_attention_mask'].sum(dim=1) - 1)).all().item())
        break

    # Visualize the first 5 samples in a batch
    print('--------------------------------------------------')
    for batch in dataloader:
        for i in range(min(5, batch['input_ids'].shape[0])):
            print(f"\nSample {i+1}:")
            print("Input Puzzle:")
            visualize_sample(
                tokenizer,
                batch['input_ids'][i],
                batch['input_attention_mask'][i].to(dtype=torch.long)
            )
            print("\nChallenge:")
            visualize_sample(
                tokenizer,
                batch['challenge_input_ids'][i],
                batch['challenge_attention_mask'][i].to(dtype=torch.long),
                batch['loss_mask'][i]
            )
            print('____________________')
        break  # Only process the first batch

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
