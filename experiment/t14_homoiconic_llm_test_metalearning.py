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
- [X] run_epoch
- [X] metaweights will be randomly initialized (not tokens anymore)
- [X] project lors
- [X] phase 2 tests
- [ ] metalearn lors


RESULTS: didn't build metalearning of lors, moved on to variant _2 where I try N-way k-shot learning on omniglot/mini-imagenet, a standard dataset. Should probably revisit this one though and add metalearning!

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
from functools import partial
from neurallambda.lab.common import print_model_info
import importlib

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
N_METAWEIGHTS = 14  # QKVOGUD * 2
LOR_LAYER = 14  # half wayish


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
    tokenizer.padding_side = 'left'
    input_encoded = tokenizer(list(inputs), padding=True, return_tensors='pt')
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

batch_size = 128
n_train_samples = batch_size * 10
n_test_samples = batch_size * 2
n_variables = 3

train_data = PuzzleDataset(variables, values, n_samples=n_train_samples, n_variables=n_variables)
train_dl = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer)
)

test_data = PuzzleDataset(variables, values, n_samples=n_test_samples, n_variables=n_variables)
test_dl = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer)
)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Check in on the dataset

if False:
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
    for batch in train_dl:
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
    for batch in train_dl:
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



##################################################
# Run it

def empty_lors(num_layers):
    '''per-layer cache of all lor blocks parsed during generation, and used in
future generation. The None values at a given layer index will be replaced with
a tuple of tensors shaped like ([BATCH, DIM, N], [BATCH, N, DIM]). N is the
rank of the LOR matrix. In the current implementation, these are just
concatenated, and never erased (within a batch, ofc). '''
    lors = {
        # low rank attention params
        "lor_qs": [None] * num_layers,
        "lor_ks": [None] * num_layers,
        "lor_vs": [None] * num_layers,
        "lor_os": [None] * num_layers,

        # low rank mlp params
        "lor_gs": [None] * num_layers,
        "lor_us": [None] * num_layers,
        "lor_ds": [None] * num_layers,
    }
    return lors


def partially_apply_models(lor_models, lor_cache):
    '''deep in the transformer stack, the LORModules get applied, but they need to
reference the current lor cache. This is where they get it from.'''
    fs = {}
    lor_ix_keys = lor_cache.keys()
    for k in lor_ix_keys:
        fs[k] = []
        for m, c in zip(lor_models[k], lor_cache[k]):
            if m is not None:
                f = partial(m, c)
            else:
                f = None
            fs[k].append(f)
    return fs


def run_epoch(model, lor_models, dataloader, optimizer, device, train=True, debug=False):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    num_layers = model.config.num_hidden_layers
    D = model.config.hidden_size

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):
            B, S = batch['input_ids'].shape
            device = model.device

            #####
            # Move to cuda

            input_ids = batch['input_ids'].to(device)
            input_attention_mask = batch['input_attention_mask'].to(device)
            challenge_input_ids = batch['challenge_input_ids'].to(device)
            challenge_attention_mask = batch['challenge_attention_mask'].to(device)
            challenge_loss_mask = batch['loss_mask'].to(device)


            #####
            # Run Inputs and metaweights to populate lor_cache

            empty_lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch

            metaweights = torch.randn(B, N_METAWEIGHTS, D, device=device) * 1e-3  # TODO: what should this be?
            inputs_embeds = torch.cat([
                model.model.embed_tokens(input_ids),
                metaweights], dim=1)

            meta_attention_mask = torch.ones((B, N_METAWEIGHTS), device=device, dtype=torch.long)
            attention_mask = torch.cat([
                input_attention_mask,
                meta_attention_mask
            ], dim=1)

            uncausal_mask = torch.cat([
                torch.zeros_like(input_attention_mask, dtype=torch.bool),
                torch.ones_like(meta_attention_mask, dtype=torch.bool)
            ], dim=1)


            # span ixs of metaweights to parse out (inclusive span). currently
            # lor_ixs is only defined for LOR_LAYER, other spans are None.
            lor_ixs = torch.zeros(B, 2, dtype=torch.long, device=device)
            with torch.no_grad():
                lor_ixs[:, 0] = S
                lor_ixs[:, 1] = S + N_METAWEIGHTS - 1

            lor_ixs_per_layer = (
                [None] * LOR_LAYER +
                [lor_ixs]  +
                [None] * (num_layers - LOR_LAYER - 1)
            )

            out = model(inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=True,
                        uncausal_mask=uncausal_mask,
                        **empty_lor_cache,
                        )

            lor_cache = update_lors(lor_models, empty_lor_cache, lor_ixs_per_layer, out.hidden_states, num_layers)


            #####
            # Run Challenge with lor_cache, but no attention to original inputs (answers must live in weights ie lor_cache)

            lorm = partially_apply_models(lor_models, lor_cache)

            challenge_out = model(
                input_ids=challenge_input_ids,
                attention_mask=challenge_attention_mask,
                **lorm,
            )


            m = challenge_loss_mask[..., 1:].contiguous().view(-1)

            vocab_size = challenge_out.logits.shape[2]
            logits = challenge_out.logits[:, :-1].contiguous().view(-1, vocab_size)  # [B, S-1, D] -> [B * (S-1), D]
            target = challenge_input_ids[:, 1:].contiguous().view(-1)  # [B, S-1] -> [B * (S-1)]
            loss = F.cross_entropy(logits[m], target[m])

            if torch.isnan(loss):
                print('NaN encountered:')
                # print(f'  all loss was masked out?: {sum([x.to(dtype=torch.long).sum() for x in loss_masks]) == 0}')
                # print(f'  nan in out_logits?: {torch.cat(out_logits, dim=1).isnan().sum() > 0}')
                # print('  ragged batch sizes? (double check by hand if necessary)')
                breakpoint()

            if train:
                optimizer.zero_grad()
                loss.backward()

                MAX_GRAD_NORM = 1.0
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(lor_models.parameters(), max_norm=MAX_GRAD_NORM)

                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples

    # Log weight histogram
    if LOG and train:
        try:
            # for name, param in itertools.chain(model.named_parameters(), lor_models.named_parameters()):
            for name, param in lor_models.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'weights/{name}', param.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'grads/{name}', param.grad.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
        except Exception as e:
            warnings.warn(f'Failed to write to tensorboard: {e}')

    if debug:
        warnings.warn('only debugging the last batch, values are not accumulated across batches (loss *is* averaged though)')
        return avg_loss, out, challenge_out, lor_cache
    else:
        return avg_loss


##################################################
# Parsing/Handling LORS

def update_lors(
        lor_models,  # shaped like `empty_lors`. Contains lor_proj, and per-head models
        lor_cache,  # shaped like `empty_lors`. Contains previously parsed lors
        lor_ixs_per_layer: List[torch.Tensor],  # List[spans], where a span is [batch, (start_ix, end_ix)]
        hidden_states,  # from out.hidden_states, so, contains all layers
        num_layers,
):
    ''' Update the LORs by interpreting hidden states as new lor blocks '''

    # check that lor_models and lor_cache are defined (at least None values)
    # for each layer, and that there's a model for every cache key
    lor_keys = lor_cache.keys()
    for k in lor_keys:
        assert (
            len(lor_models[k]) ==  # one (optional) lor module per layer
            len(lor_cache[k]) ==  # cache per layer
            num_layers
        ), (f'''
{len(lor_models[k])=} ==  # one (optional) lor module per layer
{len(lor_cache[k])=} ==  # cache per layer
{num_layers=}
''')

    h_emb = hidden_states[-1]  # final layer states

    # iterate over all layers
    for layer_ix in range(num_layers):
        lor_ix_spans = lor_ixs_per_layer[layer_ix]

        # skip non-lor'd layers
        if lor_models['lor_proj'][layer_ix] is None:
            assert lor_ix_spans is None, f'lor_proj is not defined for layer {layer_ix}, but there are lor_ix_spans is defined for this layer'
            continue

        # check that spans are within bounds
        assert isinstance(lor_ix_spans, torch.Tensor)
        assert lor_ix_spans.min() >= -1
        assert lor_ix_spans.max() <= hidden_states[-1].shape[1]


        parses = select_spans(h_emb, lor_ix_spans)

        # no parses implied anywhere
        if (parses > -1).sum() == 0:
            continue

        # run lor_proj. Returns tuple of L and R singular values, per key, eg: (lor_qs_l, lor_qs_r, ...)
        projs = lor_models['lor_proj'][layer_ix](parses)
        proj_pairs = zip(projs[::2], projs[1::2])

        # update cache
        for k, (l, r) in zip(lor_keys, proj_pairs):
            if lor_cache[k][layer_ix] is None:  # is first pass, no cache yet
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
# LOR Models


def assert_no_biases(model):
    '''Because of how batching interacts with parsing LoR weights, the LORModule
must not have biases. See LORModule for more details.'''
    bias_info = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            bias_info.append(f"Linear bias found in {name}")

        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.bias is not None:
            bias_info.append(f"Convolutional bias found in {name}")

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.bias is not None:
            bias_info.append(f"BatchNorm bias found in {name}")

        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            bias_info.append(f"LayerNorm bias found in {name}")

        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for param_name, param in module.named_parameters():
                if 'bias' in param_name:
                    bias_info.append(f"{type(module).__name__} bias found in {name}.{param_name}")

        elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
            bias_info.append(f"Embedding bias (padding_idx) found in {name}")

    if bias_info:
        error_message = "The model contains biases:\n" + "\n".join(f"- {info}" for info in bias_info)
        raise AssertionError(error_message)


def apply_lor(x, lorl, lorr) -> torch.Tensor:
    ''' Low rank "matrix" multiplication

    args:
      x: [B, S, D]
      lorl: [B, rank, out_features]
      lorr: [B, in_features, rank]

    '''
    x = torch.einsum('bsd, bdr -> bsr', x, lorr)
    x = torch.einsum('bsr, bdr -> bsd', x, lorl)

    return x


class LORProject(nn.Module):
    '''
    LOR weights need to be projected to fit the shapes of the underlying
    Linear layers they're matching. These LORModules solve this, and there
    are 2 modules per target linear layer, (left singular values, right
    singular values). They multiply like: out=LRx, to match the usual out=Wx,
    so the new calculation becomes out=Wx + LRx.

    R are the input vectors, and L are the output vectors. The first
    dimension of the LORModules must match the embedding that we're
    projecting, so the 1st values are all `dim`. The 2nd dim of R is the
    input dimension of the matched linear weights. The 2nd dim of L is the
    output dimension of the same linear layer.

    '''
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        dim = model.model.embed_tokens.weight.shape[1]  # embedding dimension
        k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
        v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
        ff_dim = model.config.intermediate_size

        self.dim = dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.ff_dim = ff_dim

        self.input_dims = [dim] * N_METAWEIGHTS  # All inputs are 'dim'-dimensional

        n_out = 14
        self.output_dims = [
            dim, dim,  # lor_qs_l, lor_qs_r
            k_dim, dim,  # lor_ks_l, lor_ks_r
            v_dim, dim,  # lor_vs_l, lor_vs_r
            dim, dim,  # lor_os_l, lor_os_r
            ff_dim, dim,  # lor_gs_l, lor_gs_r
            ff_dim, dim,  # lor_us_l, lor_us_r
            dim, ff_dim  # lor_ds_l, lor_ds_r
        ]

        self.token_mixing_dim = 128
        self.channel_mixing_dim = 128

        # Token-mixing MLP
        self.token_mixing_mlp = nn.Sequential(
            nn.RMSNorm(N_METAWEIGHTS),
            nn.Linear(N_METAWEIGHTS, self.token_mixing_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.token_mixing_dim, n_out, bias=False),
            nn.Dropout(dropout_rate)
        )

        # Channel-mixing MLP (same for all inputs)
        self.channel_mixing_mlp = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, self.channel_mixing_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.channel_mixing_dim, dim, bias=False),
            nn.Dropout(dropout_rate)
        )

        # Final projection layers
        self.final_projections = nn.ModuleDict({
            'lor_qs_l': nn.Linear(dim, dim, bias=False),
            'lor_qs_r': nn.Linear(dim, dim, bias=False),
            'lor_ks_l': nn.Linear(dim, k_dim, bias=False),
            'lor_ks_r': nn.Linear(dim, dim, bias=False),
            'lor_vs_l': nn.Linear(dim, v_dim, bias=False),
            'lor_vs_r': nn.Linear(dim, dim, bias=False),
            'lor_os_l': nn.Linear(dim, dim, bias=False),
            'lor_os_r': nn.Linear(dim, dim, bias=False),
            'lor_gs_l': nn.Linear(dim, ff_dim, bias=False),
            'lor_gs_r': nn.Linear(dim, dim, bias=False),
            # 'lor_us_l': nn.Linear(dim, ff_dim, bias=False),
            'lor_us_r': nn.Linear(dim, dim, bias=False),
            'lor_ds_l': nn.Linear(dim, dim, bias=False),
            # 'lor_ds_r': nn.Linear(dim, ff_dim, bias=False),

            # 'lor_us_l_ds_r': nn.Linear(dim, ff_dim, bias=False),

        })


        # LORModule must play nicely in a batch situation, where some samples
        # of the batch imply lor parses and others don't. Non LoR'd samples
        # should not be affected by sharing a batch with LoR'd samples. Biases
        # corrupt this property. 0-valued lors (from samples without lor
        # parses) must produce 0-valued outputs here. Checking for biases is
        # not the whole solution, you must take care.
        #
        # This is not necessarily necessary. For instance, clever masking of
        # non-parsed samples might obviate this.
        assert_no_biases(self)


    def forward(self, x):
        '''
        x: [B, N_METAWEIGHTS, D]
        '''

        B = x.shape[0]
        device = x.device

        # Token-mixing
        residual = x
        # x_token shape: [batch, dim, 14]
        x_token = x.transpose(1, 2)
        # Normalize across token dimension
        x_token = self.token_mixing_mlp(x_token)
        x_token = x_token.transpose(1, 2)  # [batch, 14, dim]

        # x = residual + x_token  # TODO: residuals off

        # Channel-mixing
        residual = x
        x = self.channel_mixing_mlp(x)

        # x = residual + x  # TODO: residuals off

        ##########
        # Original attempt

        # # Final projections to adjust dimensions
        # outputs = []
        # for i, proj in enumerate(self.final_projections.values()):
        #     outputs.append(proj(x[:, i, :]))


        ##########
        # Tie intermediates. Adopting results from (ie, results from t14_homoiconic_llm_adding_data_to_mlp)

        # ud_intermediate = self.final_projections['lor_us_l_ds_r']


        # NOTE: statistics of randn are likely way off
        # TODO: bc of this shrink, shrink token_mixing_mlp from dim=14 to dim=12, since those outputs go unused
        ud_intermediate = torch.randn(B, self.ff_dim, device=device)

        outputs = (
            self.final_projections['lor_qs_l'](x[:, 0, :]),
            self.final_projections['lor_qs_r'](x[:, 1, :]),
            self.final_projections['lor_ks_l'](x[:, 2, :]),
            self.final_projections['lor_ks_r'](x[:, 3, :]),
            self.final_projections['lor_vs_l'](x[:, 4, :]),
            self.final_projections['lor_vs_r'](x[:, 5, :]),
            self.final_projections['lor_os_l'](x[:, 6, :]),
            self.final_projections['lor_os_r'](x[:, 7, :]),
            self.final_projections['lor_gs_l'](x[:, 8, :]),
            # ud_intermediate * -1,
            self.final_projections['lor_gs_r'](x[:, 9, :]),
            # self.final_projections['lor_us_l'](x[:, 10, :]),
            ud_intermediate,
            self.final_projections['lor_us_r'](x[:, 11, :]),
            self.final_projections['lor_ds_l'](x[:, 12, :]),
            # self.final_projections['lor_ds_r'](x[:, 13, :]),
            ud_intermediate,
        )

        return outputs

class LORNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.norm = nn.RMSNorm(out_dim)

        # NOTE: there is probably a principled way of setting this, or
        #   pre-learning this. As it stands, the norm can be initialized even
        #   to 0 which effectively removes this entire layer from the
        #   transformer stack EXCEPT residuals still pass forward. Strangely
        #   enough, the network can do ok with a layer removed. I'm thinking
        #   here that we can limit the initial impact of LoR stuff by setting
        #   this low. This has a SIDE-EFFECT though of small grads to this layer.
        with torch.no_grad():
            self.norm.weight[:] = self.norm.weight * 1e-2

        # This is akin to He initialization. TODO: worth it?
        self.scale = (2 / in_dim) ** 0.5

        # LORModule must play nicely in a batch situation, where some samples
        # of the batch imply lor parses and others don't. Non LoR'd samples
        # should not be affected by sharing a batch with LoR'd samples. Biases
        # corrupt this property. 0-valued lors (from samples without lor
        # parses) must produce 0-valued outputs here. Checking for biases is
        # not the whole solution, you must take care.
        #
        # This is not necessarily necessary. For instance, clever masking of
        # non-parsed samples might obviate this.
        assert_no_biases(self)

    def forward(self, lor_cache, original, hidden_state):
        '''This gets applied separately per QKVOGUD, and really just allows
        Normalization to be handled from this module, instead of say adding a
        new norm layer throughout the underlying model.

        The `project` function is where more magic happens; it takes in ALL QKVOGUD parses for a layer, and generates a cache for each together.

        Args:
          original: model's original values of eg QKVOGUD within this layer
          hidden_state: hidden_state at this layer, that projects through this layer's associated linear QKVOGUD block, and will be used to project through the LoR version too.

        '''

        if lor_cache is not None:
            lorl, lorr = lor_cache
            l = apply_lor(hidden_state, lorl, lorr)
            # return self.norm(original + l * self.scale)  # TODO: revisit if `scale` is good
            return self.norm(original + l)
        else:
            return self.norm(original)


##################################################
# Go


num_epochs = 50

# LOR Models: same structure as LORs
dim = model.model.embed_tokens.weight.shape[1]
k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
ff_dim = model.config.intermediate_size

# Note: there must be at least a None per each QKVOGUD block per layer
lor_models = nn.ModuleDict(
    {

        #####
        # Projection
        'lor_proj': nn.ModuleList([None] * num_layers),

        #####
        # Norms

        # low rank attention params
        "lor_qs": nn.ModuleList([None] * num_layers),
        "lor_ks": nn.ModuleList([None] * num_layers),
        "lor_vs": nn.ModuleList([None] * num_layers),
        "lor_os": nn.ModuleList([None] * num_layers),

        # low rank mlp params
        "lor_gs": nn.ModuleList([None] * num_layers),
        "lor_us": nn.ModuleList([None] * num_layers),
        "lor_ds": nn.ModuleList([None] * num_layers),
    }
)

lor_models['lor_proj'][LOR_LAYER] = LORProject()

if WHICH_LOR == 1:
    lor_models['lor_qs'][LOR_LAYER] = LORNorm(dim, dim)
    lor_models['lor_ks'][LOR_LAYER] = LORNorm(dim, k_dim)
    lor_models['lor_vs'][LOR_LAYER] = LORNorm(dim, v_dim)
    lor_models['lor_os'][LOR_LAYER] = LORNorm(dim, dim)

    lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim)
elif WHICH_LOR == 2:
    lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim)
lor_models = lor_models.to(DEVICE, dtype=model.dtype)


print_model_info(lor_models)
add_hooks(lor_models)


##########
# Load state

fast_params = [
    'lor_proj.14.token_mixing_mlp.0.weight',
    'lor_proj.14.token_mixing_mlp.1.weight',
    'lor_proj.14.token_mixing_mlp.4.weight',
    'lor_proj.14.channel_mixing_mlp.0.weight',
    'lor_proj.14.channel_mixing_mlp.1.weight',
    'lor_proj.14.channel_mixing_mlp.4.weight',
    'lor_proj.14.final_projections.lor_qs_l.weight',
    'lor_proj.14.final_projections.lor_qs_r.weight',
    'lor_proj.14.final_projections.lor_ks_l.weight',
    'lor_proj.14.final_projections.lor_ks_r.weight',
    'lor_proj.14.final_projections.lor_vs_l.weight',
    'lor_proj.14.final_projections.lor_vs_r.weight',
    'lor_proj.14.final_projections.lor_os_l.weight',
    'lor_proj.14.final_projections.lor_os_r.weight',
    'lor_proj.14.final_projections.lor_gs_l.weight',
    'lor_proj.14.final_projections.lor_gs_r.weight',
    'lor_proj.14.final_projections.lor_us_l.weight',
    'lor_proj.14.final_projections.lor_us_r.weight',
    'lor_proj.14.final_projections.lor_ds_l.weight',
    'lor_proj.14.final_projections.lor_ds_r.weight',
]

models = {
    # 'main': model,
    'lor': lor_models,
}

param_groups = [
    # {
    #     'name': 'main_model',
    #     'model': 'main',
    #     'lr': 1e-5,
    #     'weight_decay': 0,
    # },
    {
        'name': 'lor_standard',
        'model': 'lor',
        'lr': 1e-3,
        'weight_decay': 1e-2,
        'exclude': fast_params,
    },
    {
        'name': 'lor_high',
        'model': 'lor',
        'lr': 1e-3,
        'weight_decay': 1e-2,
        'include': fast_params,
    },
]

optimizer_param_groups = []

for group in param_groups:
    params = []
    for name, param in models[group['model']].named_parameters():
        if 'include' in group and name in group['include']:
            params.append(param)
        elif 'exclude' in group and name not in group['exclude']:
            params.append(param)
        elif 'include' not in group and 'exclude' not in group:
            params.append(param)

    if params:
        optimizer_param_groups.append({
            'params': params,
            'lr': group['lr'],
            'weight_decay': group.get('weight_decay', 0)
        })

optimizer = optim.AdamW(optimizer_param_groups)

'''

new_lr = 1e-2
for param_group in optimizer.param_groups: param_group['lr'] = new_lr

new_lr = 1e-3
for param_group in optimizer.param_groups: param_group['lr'] = new_lr

new_lr = 1e-4
for param_group in optimizer.param_groups: param_group['lr'] = new_lr

'''

train_losses = []
test_losses = []
best_loss = float('inf')
global_epoch = 0

model.train()
for epoch in range(num_epochs):
    global_epoch += 1
    train_loss = run_epoch(model, lor_models, train_dl, optimizer, DEVICE, train=True)
    train_losses.append(train_loss)
    writer.add_scalars('loss', {'train': train_loss}, global_epoch)

    if epoch % 1 == 0:
        test_loss = run_epoch(model, lor_models, test_dl, optimizer, DEVICE, train=False)
        # test_loss = 0
        test_losses.append(test_loss)
        writer.add_scalars('loss', {'test': test_loss}, global_epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")



if num_epochs > 0:
    epochs = list(range(len(train_losses)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='red')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
