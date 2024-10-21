'''

The previous versions worked on raw lists of data. This module develops library functions for working with torch Datasets and DataLoaders.

'''

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase

def create_position_ids(attention_mask: torch.Tensor, start_ixs: torch.Tensor = None) -> torch.Tensor:
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

class ColumnwiseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]['prepared_data']


def columnwise_collate_fn(
        batch: List[List[Dict[str, Any]]]  # rows of single-columns
) -> List[List[Dict[str, Any]]]:  # list of columns with multiple rows
    """
    Collate function for columnwise processing.
    Pads each column to have the same number of rows within a batch.
    """
    max_cols = max(len(item) for item in batch)

    # not all data has same number of blocks/single columns, this pads blocks/cols with empty data
    padded_batch = []
    for item in batch:
        padded_item = item + [{'type': 'pad_block', 'content': '', 'include_in_loss': False}] * (max_cols - len(item))
        padded_batch.append(padded_item)

    # Transpose the batch to get columns
    transposed = list(zip(*padded_batch))
    return transposed


def create_column_batch_inputs(
    batch: List[List[Dict[str, Any]]],  # list of rows of single columns/blocks
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    assert_columns_same_type: bool = True
) -> List[Dict[str, torch.Tensor]]:
    """
    Create batch inputs for column-wise processing, including a loss mask
    that considers both 'include_in_loss' and attention mask.

    Dictionaries can include 'loss_mask' if they intend to have a mixture of true/false.
    """
    col_batch = columnwise_collate_fn(batch)  # list of columns with multiple rows (dictionaries of text)
    col_batch_inputs = []
    batch_size = len(col_batch[0])
    max_positions = torch.tensor([0] * batch_size, device=device)

    for column in col_batch:

        # check that an entire column contains same type, eg they're all either
        # text or lor_blocks. Ignores padding.
        if assert_columns_same_type:
            row_type = column[0]['type']
            for row in column:
                assert row['type'] == row_type or row['type'] == 'pad_block'

        texts = [item['content'] for item in column]
        include_in_losses = [item['include_in_loss'] for item in column]

        inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        position_ids = create_position_ids(inputs['attention_mask'], start_ixs=max_positions)
        inputs['position_ids'] = position_ids

        # Create loss mask using attention mask and include_in_loss, or
        # optional `loss_mask` that much match token length of corresponding
        # text
        loss_mask = inputs['attention_mask'].bool()  # Start with attention mask
        for i, include in enumerate(include_in_losses):
            if 'loss_mask' in column[i] and column[i]['loss_mask'] is not None:
                loss_mask_row = torch.tensor(column[i]['loss_mask'], dtype=torch.bool)
                assert loss_mask_row.shape == inputs['input_ids'][i][inputs['attention_mask'][i].bool()].shape, 'loss_mask must have same number of elements as tokenized inputs'
                loss_mask[i, :] = loss_mask_row
            if not include:
                loss_mask[i, :] = False  # Mask out entire sequence if not included in loss

        inputs['loss_mask'] = loss_mask

        # increment
        max_positions = 1 + position_ids.max(dim=1).values.clip(0)  # clip -1 values
        col_batch_inputs.append(inputs)

    return col_batch_inputs


def create_dataloader(
    dataset,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    shuffle: bool = True,
    num_workers: int = 0,
    assert_columns_are_same_type: bool = False
) -> DataLoader:
    """
    Create a DataLoader for columnwise processing.
    """
    columnwise_dataset = ColumnwiseDataset(dataset)

    return DataLoader(
        columnwise_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: create_column_batch_inputs(batch, tokenizer, device, assert_columns_are_same_type)
    )


##################################################
# Sandbox

if False:
    import os
    import torch
    import itertools
    from typing import List
    from transformers import AutoTokenizer
    from torch.utils.data import Dataset, DataLoader
    import t14_homoiconic_llm_model as Q
    from datasets import Dataset as HFDataset

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

    def insert_lor_blocks(dataset: HFDataset, block_content) -> HFDataset:
        """
        Insert LOR blocks between text elements in the dataset.

        NOTE: this is not part of the "library" portion because it's an implementation detail and will be copied into modules that need it
        """
        def process_row(row):
            prepared_data = []
            for item in row['input']:
                prepared_data.append({"type": "text", "content": item})
                prepared_data.append({"type": "lor", "content": block_content})
            # Remove the last LOR block
            prepared_data = prepared_data[:-1]
            # Add the output
            prepared_data.append({"type": "text", "content": row['output']})
            return {"prepared_data": prepared_data}

        return dataset.map(process_row, remove_columns=["input", "output"])

    def test_equivalent_tokenization(dataset: HFDataset, tokenizer):
        ''' Ignores lor blocks, only tests text blocks '''
        print("Testing equivalent tokenization...")
        base_tokens = list(itertools.chain(*[tokenizer.encode(item['content']) for item in dataset[0]['prepared_data']]))

        for i in range(1, len(dataset)):
            row_tokens = list(itertools.chain(*[tokenizer.encode(item['content']) for item in dataset[i]['prepared_data']]))
            assert base_tokens == row_tokens, (
                f"Row {i} tokenization differs from base row:\n" +
                f'base: {base_tokens}\n' +
                f'row : {row_tokens}'
            )

        print("All rows tokenize equivalently.")

    def test_equivalent_logits(model, dataloader):
        print("Testing equivalent logits...")

        # Process the entire batch using forward_columns
        batch = next(iter(dataloader))
        batch_logits, batch_attn, _ = forward_columns(model, batch)

        # Combine logits and mask padding
        combined_logits = torch.cat(batch_logits, dim=1)
        combined_attention_mask = torch.cat([inputs['attention_mask'] for inputs in batch], dim=1)

        # Compare logits for each row
        base_logits = combined_logits[0][combined_attention_mask[0].bool()]
        for i in range(1, combined_logits.shape[0]):
            row_logits = combined_logits[i][combined_attention_mask[i].bool()]
            assert torch.allclose(base_logits, row_logits, atol=1e-3), f"Row {i} logits differ from base row"

        print("All rows produce equivalent logits.")

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

    # equivalent rows, but some have BLOCK hardcoded, and some will get it from
    # insert_lor_blocks. This will test that the shorter rows get padded with
    # empty blocks appropriately.
    data = [
        {"input": ["A~B~C~D~E", "~F~G~H~I~J", "~K~L~M~N~O"], "output": "Z"},
        {"input": [f"A~B~C~D~E{BLOCK}~F~G~H~I~J", "~K~L~M~N~O"], "output": "Z"},
        {"input": ["A~B~C~D~E", f"~F~G~H~I~J{BLOCK}~K~L~M~N~O"], "output": "Z"},
        {"input": [f"A~B~C~D~E{BLOCK}~F~G~H~I~J{BLOCK}~K~L~M~N~O"], "output": "Z"},

    ]

    # Create the initial dataset
    dataset = HFDataset.from_list(data)

    # Insert LOR blocks
    dataset_with_blocks = insert_lor_blocks(dataset, BLOCK)

    # Create the DataLoader
    dataloader = create_dataloader(
        dataset=dataset_with_blocks,
        batch_size=100,  # Use the full dataset as one batch for testing
        tokenizer=tokenizer,
        device=DEVICE,
        shuffle=False,
        num_workers=0
    )

    test_equivalent_tokenization(dataset_with_blocks, tokenizer)
    test_equivalent_logits(model, dataloader)

    print("All tests passed successfully!")




##################################################
# graveyard

# def create_column_batch_inputs(
#     col_batch: List[List[Dict[str, Any]]],
#     tokenizer: PreTrainedTokenizerBase,
#     device: str
# ) -> List[Dict[str, torch.Tensor]]:
#     """
#     Create batch inputs for column-wise processing.
#     """
#     col_batch_inputs = []
#     batch_size = len(col_batch[0])
#     max_positions = torch.tensor([0] * batch_size, device=device)

#     for column in col_batch:
#         texts = [item['content'] for item in column]
#         inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
#         position_ids = create_position_ids(inputs['attention_mask'], start_ixs=max_positions)
#         inputs['position_ids'] = position_ids
#         # increment
#         max_positions = 1 + position_ids.max(dim=1).values.clip(0)  # clip -1 values
#         col_batch_inputs.append(inputs)

#     return col_batch_inputs


# def create_column_batch_inputs(
#     col_batch: List[List[Dict[str, Any]]],
#     tokenizer: PreTrainedTokenizerBase,
#     device: str
# ) -> List[Dict[str, torch.Tensor]]:
#     """
#     Create batch inputs for column-wise processing, including a loss mask.
#     """
#     col_batch_inputs = []
#     batch_size = len(col_batch[0])
#     max_positions = torch.tensor([0] * batch_size, device=device)

#     for column in col_batch:
#         texts = [item['content'] for item in column]
#         include_in_loss = [item['include_in_loss'] for item in column]

#         inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
#         position_ids = create_position_ids(inputs['attention_mask'], start_ixs=max_positions)
#         inputs['position_ids'] = position_ids

#         # Create loss mask
#         loss_mask = torch.zeros_like(inputs['input_ids'], dtype=torch.bool)  # default false
#         for i, (tokens, include) in enumerate(zip(inputs['input_ids'], include_in_loss)):
#             if include:
#                 loss_mask[i, :len(tokens)] = True

#         inputs['loss_mask'] = loss_mask

#         # increment
#         max_positions = 1 + position_ids.max(dim=1).values.clip(0)  # clip -1 values
#         col_batch_inputs.append(inputs)

#     return col_batch_inputs
