'''

Add ability for each row to mention its own LOR parse indexes

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
        batch: List[List[Dict[str, Any]]],  # rows of single-columns
        lor_ix_keys: List[str]
) -> List[List[Dict[str, Any]]]:  # list of columns with multiple rows
    """
    Collate function for columnwise processing.
    Pads each column to have the same number of rows within a batch.
    """
    max_cols = max(len(item) for item in batch)

    # not all data has same number of blocks/single columns, this pads blocks/cols with empty data
    padded_batch = []
    for item in batch:
        padded_item = item + [{'type': 'pad_block',
                               'content': '',
                               'include_in_loss': False,
                               **{k: (None, None) for k in lor_ix_keys}
                               }] * (max_cols - len(item))
        padded_batch.append(padded_item)

    # Transpose the batch to get columns
    transposed = list(zip(*padded_batch))
    return transposed


def create_column_batch_inputs(
    batch: List[List[Dict[str, Any]]],  # list of rows of single columns/blocks
    lor_ix_keys: List[str],
    tokenizer: PreTrainedTokenizerBase,
    device: str
) -> List[Dict[str, torch.Tensor]]:
    """
    Create batch inputs for column-wise processing, including a loss mask
    that considers both 'include_in_loss' and attention mask.

    Dictionaries can include 'loss_mask' if they intend to have a mixture of true/false.
    """
    col_batch = columnwise_collate_fn(batch, lor_ix_keys)  # list of columns with multiple rows (dictionaries of text)
    col_batch_inputs = []
    batch_size = len(col_batch[0])
    max_positions = torch.tensor([0] * batch_size, device=device)

    for column in col_batch:

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

        # lor parsing ixs. Stack the ixs in a vector of size `[batch]` (ie has no sequence dim)
        for k in lor_ix_keys:
            left_lor_ixs = []
            right_lor_ixs = []
            for item in column:
                l = item[k][0]
                r = item[k][1]
                left_lor_ixs.append(l if l is not None else -1)
                right_lor_ixs.append(r if r is not None else -1)
            inputs[k] = (torch.tensor(left_lor_ixs), torch.tensor(right_lor_ixs))

        # increment
        max_positions = 1 + position_ids.max(dim=1).values.clip(0)  # clip -1 values
        col_batch_inputs.append(inputs)

    return col_batch_inputs


def create_dataloader(
    dataset,
    lor_ix_keys: List[str],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    shuffle: bool = True,
    num_workers: int = 0,
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
        collate_fn=lambda batch: create_column_batch_inputs(batch, lor_ix_keys, tokenizer, device)
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
    from neurallambda.lab.common import print_grid

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

    lor_block = 'Q^*K^*'  # i'm writing it this way bc it tokenizes into 6 pieces, but I dont need to add to vocabulary

    empty_lor_ixs = {
        'lor_qs_ix': (None, None),
        'lor_ks_ix': (None, None),
    }

    lor_ixs = {
        'lor_qs_ix': (0, 1),  # (left singular value, right singular value)
        'lor_ks_ix': (3, 4),
    }

    lor_ix_keys = set(lor_ixs.keys())

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = os.path.expanduser("~/_/models/Qwen2-1.5B")

    # model = Q.Qwen2ForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.float32,
    #     device_map=DEVICE,
    #     _attn_implementation='eager',
    # )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def t(x):
        return {'type': 'text', 'content': x, 'include_in_loss': False, **empty_lor_ixs}

    def w(x, lor_ixs):
        # one parse per type can be used per column/block, eg LOR Q left
        # singular value, LOR Q right singular value, LOR K etc...
        return {'type': 'lor', 'content': x, 'include_in_loss': False, **lor_ixs}

    def p(x):
        return {'type': 'pad_block', 'content': x, 'include_in_loss': False, **empty_lor_ixs}

    lor = w(lor_block, lor_ixs)

    # equivalent rows, but some have BLOCK hardcoded, and some will get it from
    # insert_lor_blocks. This will test that the shorter rows get padded with
    # empty blocks appropriately.
    data = [
        [t('a'), lor, t('b')],
        [t('a b c'), t('d')],
        [t('x')],
    ]

    # Create the initial dataset
    col_dataset = create_column_batch_inputs(data, lor_ix_keys, tokenizer, DEVICE)


    ##########
    # Print Table

    def format_inputs_for_grid(data, row_ix=0):
        formatted_data = []

        # flatten nested lists
        flat = lambda xss: list(itertools.chain(*xss))
        rmpad = lambda xs: list(map(lambda x: 'pad' if x == tokenizer.pad_token_id else x, xs))
        decode = lambda xs: list(map(lambda tokid: 'pad' if tokid == tokenizer.pad_token_id else tokenizer.decode([tokid], skip_special_tokens=False), xs))

        # Extract the relevant tensors
        toks           = flat([decode(item['input_ids'][row_ix]) + ['|'] for item in data])
        input_ids      = flat([rmpad(item['input_ids'][row_ix].tolist() + ['|']) for item in data])
        attention_mask = flat([rmpad(item['attention_mask'][row_ix].tolist() + ['|']) for item in data])
        position_ids   = flat([rmpad(item['position_ids'][row_ix].tolist() + ['|']) for item in data])

        # Create rows for the grid
        formatted_data.append(['tok'] + [x for x in toks])
        formatted_data.append(['inp'] + [str(id) for id in input_ids])
        formatted_data.append(['att'] + [str(mask) for mask in attention_mask])
        formatted_data.append(['pos'] + [str(pos) for pos in position_ids])

        return formatted_data

    print()
    B = len(data)
    for b in range(B):
        x = format_inputs_for_grid(col_dataset, row_ix=b)
        print()
        print(f'Item {b}')
        print_grid(x)

    print("All tests passed successfully!")
