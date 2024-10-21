'''

Verify that batches of data can be:

* processed into column blocks
* have the BLOCKS aligned
* ensure that attention masking works properly with padding
* ensure that position_ids make processing with padding, or not, equivalent
* ensure that whether or not an input is columnized, outputs are equivalent

'''

import os
import torch
import torch.nn.functional as F
import random
from transformers import AutoTokenizer
from typing import List, Dict
import t14_homoiconic_llm_model as Q
import itertools

SEED = 152
torch.manual_seed(152)
random.seed(SEED)

DEVICE = 'cuda:1'
BATCH_SIZE = 32

model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")

try:
    already_loaded
except:
    print('Loading model')
    model = Q.Qwen2ForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float32,
        # torch_dtype=torch.float64,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    already_loaded = True


##########
#

def empty_lors(num_layers):
    lors = {
        # low rank attention params
        "lor_qs": [None] * num_layers,
        "lor_ks": [None] * num_layers,
        "lor_vs": [None] * num_layers,
        "lor_os": [None] * num_layers,

        # low rank mlp params
        "lor_us": [None] * num_layers,
        "lor_gs": [None] * num_layers,
        "lor_ds": [None] * num_layers,
    }
    return lors

def forward_columns(model, col_inputs):
    '''Recurrently process column blocks of ids, concatenating attention_mask and
past_key_values across generations. '''
    col_out_logits = []
    past_key_values = None
    lors = empty_lors(model.config.num_hidden_layers)

    # Iterate over columns of the batch
    attention_mask = None
    for i, batch_column in enumerate(col_inputs):
        input_ids = batch_column['input_ids']
        new_attention_mask = batch_column['attention_mask']
        position_ids = batch_column['position_ids']

        # padded columns can have 0 elements in a given row. This shouldn't
        # affect anything if processing whole batches, but when you prepare
        # cols then process row-wise (ie when debugging) there can be issues.
        if input_ids.numel() == 0:
            continue

        # `attention_mask` will continue to grow as the entire sequence length
        # seen so far
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
        else:
            attention_mask = new_attention_mask

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    position_ids=position_ids,
                    **lors,
                    )
        col_out_logits.append(out.logits)
        past_key_values = out.past_key_values

    # # for developing in interpreter mode
    # save = ['past_key_values', 'hidden_states']
    # for name, value in locals().items():
    #     if name in save:
    #         globals()[name] = value

    return col_out_logits, attention_mask, past_key_values


##################################################


def columnized_to_row_wise(col_data: List[Dict[str, torch.Tensor]], key: str) -> List[List[torch.Tensor]]:
    """
    Convert columnized data to row-wise format.

    Args:
    col_data (List[Dict[str, torch.Tensor]]): List of dictionaries containing columnized data
    key (str): The key to access in each dictionary (e.g., 'input_ids' or 'attention_mask')

    Returns:
    List[List[torch.Tensor]]: Row-wise data
    """
    num_samples = col_data[0][key].shape[0]
    return [[col[key][i] for col in col_data] for i in range(num_samples)]

def print_grid(grid: List[List[str]]):
    """
    Print a grid stored as a list of lists, with dynamic column widths.

    Args:
    grid (List[List[str]]): The grid to print
    """
    if not grid:
        return

    # Calculate the maximum width for each column
    col_widths = [max(len(str(cell)) for row in grid for cell in row[i:i+1]) for i in range(len(grid[0]))]

    # Print the grid with dynamic column widths
    for row in grid:
        print(" ".join(f"{str(cell):^{width}}" for cell, width in zip(row, col_widths)))

def create_token_grid(row_wise_data: List[List[torch.Tensor]], tokenizer) -> List[List[str]]:
    """
    Create a token grid from row-wise data, replacing padding tokens with '_'.

    Args:
    row_wise_data (List[List[torch.Tensor]]): Row-wise input_ids data
    tokenizer: The tokenizer to decode token IDs

    Returns:
    List[List[str]]: Grid of decoded tokens
    """
    pad_token = tokenizer.pad_token
    return [
        [
            '_' if tokenizer.decode([token_id.item()]) == pad_token else tokenizer.decode([token_id.item()])
            for tensor in row for token_id in tensor
        ]
        for row in row_wise_data
    ]

def create_attention_mask_grid(row_wise_data: List[List[torch.Tensor]]) -> List[List[str]]:
    """
    Create an attention mask grid from row-wise data.

    Args:
    row_wise_data (List[List[torch.Tensor]]): Row-wise attention_mask data

    Returns:
    List[List[str]]: Grid of attention mask values
    """
    return [[str(val.item()) for tensor in row for val in tensor] for row in row_wise_data]

def print_token_grid(col_inputs: List[Dict[str, torch.Tensor]], tokenizer):
    row_wise_data = columnized_to_row_wise(col_inputs, 'input_ids')
    token_grid = create_token_grid(row_wise_data, tokenizer)
    print("Token Grid:")
    print_grid(token_grid)

def print_attention_mask_grid(col_inputs: List[Dict[str, torch.Tensor]]):
    row_wise_data = columnized_to_row_wise(col_inputs, 'attention_mask')
    mask_grid = create_attention_mask_grid(row_wise_data)
    print("Attention Mask Grid:")
    print_grid(mask_grid)

def print_position_id_grid(col_inputs: List[Dict[str, torch.Tensor]]):
    def create_position_id_grid(row_wise_data: List[List[torch.Tensor]]) -> List[List[str]]:
        """
        Create a position ID grid from row-wise data.

        Args:
        row_wise_data (List[List[torch.Tensor]]): Row-wise position_ids data

        Returns:
        List[List[str]]: Grid of position ID values
        """
        return [[str(val.item()) if val.item() != -1 else '_' for tensor in row for val in tensor] for row in row_wise_data]

    row_wise_data = columnized_to_row_wise(col_inputs, 'position_ids')
    position_id_grid = create_position_id_grid(row_wise_data)
    print("Position ID Grid:")
    print_grid(position_id_grid)


##################################################
# Data

# def create_position_ids(attention_mask):
#     """
#     Create position IDs based on the attention mask.
#     Set position to -1 for padding tokens.
#     """
#     position_ids = attention_mask.long().cumsum(-1) - 1
#     position_ids.masked_fill_(attention_mask == 0, -1)
#     return position_ids


def create_position_ids(attention_mask, start_ixs: torch.Tensor = None):
    """
    Create position IDs based on the attention mask, starting from start_ix.
    Set position to -1 for padding tokens.
    """
    S = attention_mask.shape[-1]
    position_ids = attention_mask.long().cumsum(-1) - 1
    if start_ixs is not None:
        position_ids = position_ids + start_ixs.unsqueeze(1).repeat(1, S)
    position_ids.masked_fill_(attention_mask == 0, -1)
    return position_ids


def pad_columns(xss: List[List[str]], padding_side='right'):
    ''' The input list of lists represents batches of columns:
    xss[0] is all the columns of the first row
    xss[0][1] is the second column of the first row
    This function adds new columns of empty strings to pad each row so each row has the same number of columns.
    '''
    max_cols = max(map(len, xss))

    padded_xss = []
    for row in xss:
        padding_length = max_cols - len(row)
        if padding_side == 'right':
            padded_row = row + [''] * padding_length
        else:  # padding_side == 'left'
            padded_row = [''] * padding_length + row
        padded_xss.append(padded_row)

    return padded_xss

BLOCK = '~X'

check_eq_ixs = [0, 5, 6, 7, 8, 9] # to check that block-wise padding works

# This contains 2 different tests:
#   1. is processing in columnized batches equivalent (even after padding) to processing rows?
#   2. is block padding equivalent to not using it?
prompt_chonkss = pad_columns([
    ["A~B~C~D~E", BLOCK, "~F~G~H~I~J", BLOCK, "~K~L~M~N~O"],
    ["A~B~C~D~E", BLOCK, "~F~G~H~I~J", BLOCK + "~K~L~M~N~O"],
    ["A~B~C~D~E", BLOCK, "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
    ["A~B~C~D~E", BLOCK + "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
    ["A~B~C~D~E" + BLOCK + "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
    ["", "A~B~C~D~E" + BLOCK + "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
])

# Double check block padding tests tokenize the same
ix0 = check_eq_ixs[0]
id0 = list(itertools.chain(*[tokenizer.encode(x) for x in prompt_chonkss[ix0]]))
for ix1 in check_eq_ixs[1:]:
    id1 = list(itertools.chain(*[tokenizer.encode(x) for x in prompt_chonkss[ix1]]))
    assert id0 == id1
print('block padding tests tokenize the same')


# reorient prompt_chonkss into columns, since batches of chonks will be processed column wise
col_prompt_chonkss = zip(*prompt_chonkss)


# COLUMN-WISE
# Columns are each separate, and contain a full batch
col_batch_inputs = []
batch_size = len(prompt_chonkss)
max_positions = torch.tensor([0] * batch_size, device=DEVICE)

for prompt_chonks in col_prompt_chonkss:
    inputs = tokenizer(prompt_chonks, padding=True, return_tensors="pt").to(DEVICE)
    position_ids = create_position_ids(inputs['attention_mask'], start_ixs=max_positions)
    inputs['position_ids'] = position_ids
    # increment
    max_positions = 1 + position_ids.max(dim=1).values.clip(0)  # clip -1 values
    col_batch_inputs.append(inputs)


# ROW-WISE
# Rows are each separate, and contain just that row (in columns), so no padding
# induced from columnizing over a batch
row_inputs = []
for row in prompt_chonkss:
    row_toks = [tokenizer(x, return_tensors="pt").to(DEVICE) for x in row]
    # add position_ids
    max_position = torch.tensor([0], device=DEVICE)
    for col in row_toks:
        if col['input_ids'].numel() == 0:
            position_ids = None
        else:
            position_ids = create_position_ids(col['attention_mask'], start_ixs=max_position)
            max_position = 1 + position_ids.max(dim=1).values.clip(0)  # clip -1 values
        col['position_ids'] = position_ids
    row_inputs.append(row_toks)

# # debug
# for i, (row1, row2) in enumerate(zip(row_inputs[check_eq_ixs[0]], row_inputs[check_eq_ixs[1]])):
#     print(f"Column {i}:")
#     print("Row 1 input_ids:", row1['input_ids'])
#     print("Row 2 input_ids:", row2['input_ids'])
#     print("Row 1 position_ids:", row1['position_ids'])
#     print("Row 2 position_ids:", row2['position_ids'])
#     print("Row 1 attention_mask:", row1['attention_mask'])
#     print("Row 2 attention_mask:", row2['attention_mask'])
#     print()


# # If each row in the batch is the same, we can make sure column-wise and row-wise are equivalent
# for col in range(5):
#     for row in range(2):
#         assert (
#             col_batch_inputs[col]['input_ids'][row].tolist() ==
#             row_inputs[row][col]['input_ids'].squeeze(0).tolist()
#         )
#         assert (
#             col_batch_inputs[col]['attention_mask'][row].tolist() ==
#             row_inputs[row][col]['attention_mask'].squeeze(0).tolist()
#         )
# print('column-wise and row-wise are same')


##################################################
# Print Grids

print()
print_token_grid(col_batch_inputs, tokenizer)
print_attention_mask_grid(col_batch_inputs)
print_position_id_grid(col_batch_inputs)

##################################################
# Test batch vs single-mode

# Process the entire batch
batch_logits, batch_attn, _ = forward_columns(model, col_batch_inputs)

# Process each row individually
single_logits = []
for cols in row_inputs:
    logits, _, _ = forward_columns(model, cols)
    logits = torch.cat(logits, dim=1)
    single_logits.append(logits)

for i in range(len(prompt_chonkss)):
    print()

    # collect a row from the batched version, masking out the padding tokens (where attention==0)
    b_attn = [col['attention_mask'][i] for col in col_batch_inputs]
    b_out = [col[i] for col in batch_logits]
    b = []
    for bb, bm in zip(b_out, b_attn):
        assert bb.shape[0] == bm.shape[0]
        b.append(bb[bm.bool()])
    b = torch.cat(b, dim=0)

    # a row from the single-batched version
    s = single_logits[i].squeeze(0)

    # Check if the differences are within an acceptable threshold
    threshold = 1e-3
    is_close = torch.allclose(b, s, atol=threshold)

    if is_close:
        print(f"row {i}: Test PASSED: Batch and single row processing produce equivalent results.")
    else:
        print(f"row {i}: Test FAILED: Batch and single row processing produce different results.")

        # Additional diagnostics
        different_elements = torch.abs(b - s) > threshold
        num_different = torch.sum(different_elements).item()
        total_elements = b.numel()
        print(f"Number of elements exceeding threshold: {num_different} out of {total_elements}")
        print(f"Percentage of different elements: {(num_different / total_elements) * 100:.2f}%")
        # Max diff
        max_diff = torch.max(torch.abs(b - s))
        print(f"Maximum difference between batch and single row processing: {max_diff.item()}")


# Check that block padding doesn't affect results
ix0 = check_eq_ixs[0]
logits = torch.cat(batch_logits, dim=1)
m0 = batch_attn[ix0].bool()
x0 = logits[ix0][m0]

for ix1 in check_eq_ixs[1:]:
    m1 = batch_attn[ix1].bool()
    x1 = logits[ix1][m1]
    assert x0.shape == x1.shape
    assert torch.allclose(x0, x1, atol=1e-3)

print('Block padding doesnt corrupt results')
