'''.

Librarify the previous version

0. Collect your dataset where each sample is a list of strings, each string
   representing a separate column to organize into a block with the same column
   index from other rows/samples. Strings per column can have different
   lengths, and `create_column_batch_inputs` will pad them appropriately,
   construct an appropriate attention_mask, and position_ids.

   NOTE: if your data is particularly ragged, this has a performance
   implication, since internal columns will be padded to the same length as the
   longest string in that column.

1. `create_column_batch_inputs` over your List[List[str]]

2. make sure to use the provided `position_ids` and `attention_mask`. A helper
   function (`forward_columns`) is seen in other modules that makes use of
   this:

    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                position_ids=position_ids)

'''

import torch
from typing import List, Dict

def create_position_ids(attention_mask: torch.Tensor, start_ixs: torch.Tensor = None) -> torch.Tensor:
    """
    Create position IDs based on the attention mask, starting from start_ix.
    Set position to -1 for padding tokens.

    Args:
    attention_mask (torch.Tensor): The attention mask tensor
    start_ixs (torch.Tensor, optional): Starting indices for each sequence in the batch

    Returns:
    torch.Tensor: Position IDs tensor
    """
    S = attention_mask.shape[-1]
    position_ids = attention_mask.long().cumsum(-1) - 1
    if start_ixs is not None:
        position_ids = position_ids + start_ixs.unsqueeze(1).repeat(1, S)
    position_ids.masked_fill_(attention_mask == 0, -1)
    return position_ids

def pad_columns(xss: List[List[str]], padding_side: str = 'right') -> List[List[str]]:
    """
    Pad columns in a list of lists to ensure each row has the same number of columns.

    Args:
    xss (List[List[str]]): Input list of lists representing batches of columns
    padding_side (str, optional): Side to add padding ('left' or 'right'). Defaults to 'right'.

    Returns:
    List[List[str]]: Padded list of lists
    """
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

def create_column_batch_inputs(
    prompt_chonkss: List[List[str]],
    tokenizer,
    device: str
) -> List[Dict[str, torch.Tensor]]:
    """
    Create batch inputs for column-wise processing.

    Args:
    col_prompt_chonkss (List[List[str]]): List of column prompts
    tokenizer: The tokenizer to use for encoding
    device (str): The device to place tensors on (e.g., 'cuda:0', 'cpu')

    Returns:
    List[Dict[str, torch.Tensor]]: List of input dictionaries for each column
    """
    # reorient prompt_chonkss into columns, since batches of chonks will be
    # processed column wise
    col_prompt_chonkss = list(zip(*pad_columns(prompt_chonkss)))
    col_batch_inputs = []
    batch_size = len(col_prompt_chonkss[0])
    max_positions = torch.tensor([0] * batch_size, device=device)

    for prompt_chonks in col_prompt_chonkss:
        inputs = tokenizer(prompt_chonks, padding=True, return_tensors="pt").to(device)
        position_ids = create_position_ids(inputs['attention_mask'], start_ixs=max_positions)
        inputs['position_ids'] = position_ids
        # increment
        max_positions = 1 + position_ids.max(dim=1).values.clip(0)  # clip -1 values
        col_batch_inputs.append(inputs)

    return col_batch_inputs


##################################################
# Tests

if False:
    import os
    import torch
    import itertools
    from typing import List
    from transformers import AutoTokenizer
    import t14_homoiconic_llm_model as Q

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
            if i > 0:
                assert past_key_values is not None


        return col_out_logits, attention_mask, past_key_values

    def test_equivalent_tokenization(prompt_chonkss: List[List[str]], tokenizer):
        print("Testing equivalent tokenization...")
        base_tokens = list(itertools.chain(*[tokenizer.encode(x) for x in prompt_chonkss[0]]))

        for i, row in enumerate(prompt_chonkss[1:], start=1):
            row_tokens = list(itertools.chain(*[tokenizer.encode(x) for x in row]))
            assert base_tokens == row_tokens, f"Row {i} tokenization differs from base row"

        print("All rows tokenize equivalently.")

    def test_equivalent_logits(model, tokenizer, prompt_chonkss: List[List[str]]):
        print("Testing equivalent logits...")
        batch_inputs = create_column_batch_inputs(prompt_chonkss, tokenizer, DEVICE)

        # Process the entire batch using forward_columns
        batch_logits, batch_attn, _ = forward_columns(model, batch_inputs)

        # Combine logits and mask padding
        combined_logits = torch.cat(batch_logits, dim=1)
        combined_attention_mask = torch.cat([inputs['attention_mask'] for inputs in batch_inputs], dim=1)

        # Compare logits for each row
        base_logits = combined_logits[0][combined_attention_mask[0].bool()]
        for i in range(1, len(prompt_chonkss)):
            row_logits = combined_logits[i][combined_attention_mask[i].bool()]
            assert torch.allclose(base_logits, row_logits, atol=1e-3), f"Row {i} logits differ from base row"

        print("All rows produce equivalent logits.")


    ##########
    # Go

    BLOCK = '~X'
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = os.path.expanduser("~/_/models/Qwen2-1.5B")

    model = Q.Qwen2ForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_chonkss = [
        ["A~B~C~D~E", BLOCK, "~F~G~H~I~J", BLOCK, "~K~L~M~N~O"],
        ["A~B~C~D~E", BLOCK, "~F~G~H~I~J", BLOCK + "~K~L~M~N~O"],
        ["A~B~C~D~E", BLOCK, "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
        ["A~B~C~D~E", BLOCK + "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
        ["A~B~C~D~E" + BLOCK + "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
        ["", "A~B~C~D~E" + BLOCK + "~F~G~H~I~J" + BLOCK + "~K~L~M~N~O"],
    ]

    test_equivalent_tokenization(prompt_chonkss, tokenizer)
    test_equivalent_logits(model, tokenizer, prompt_chonkss)

    print("All tests passed successfully!")
