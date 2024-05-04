
import torch
import random
import torch.nn.functional as F
from datasets import Dataset
from functools import partial
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from typing import List, Dict
import tokenizers


##################################################
# DATA

def print_grid(data):
    # maximum width for each column
    column_widths = [max(len(str(item)) for item in column)
                     for column in zip(*data)]
    for row in data:
        formatted_row = "  ".join(str(item).ljust(width)
                                  for item, width in zip(row, column_widths))
        print(formatted_row)


def print_model_info(model):
    print('------------')
    print('MODEL PARAMS')
    info = []
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        info.append((name,
                     param_count,
                     f'{list(param.shape)}',
                     f'grad:{param.requires_grad}'))
    print_grid(info)
    print(f'Total Parameters: {total_params:,}')


def iterator(all_datasets: List[Dataset], keys: List[str], special_tokens):
    ''' Iterate over a list of datasets and build a vocab for them. '''
    for tok in special_tokens:
        yield tok
    for dataset in all_datasets:
        for xs in dataset:
            for key in keys:
                for x in xs[key]:
                    yield x


def collate_fn(tokenizer, pad_token, batch):
    ''' Tokenize and pad batches of data. '''

    # is_pretokenized expects `isinstance(sample['inputs'], List[str])`
    input_ids = [tokenizer.encode(sample['inputs'], is_pretokenized=True).ids
                 for sample in batch]
    output_ids = [tokenizer.encode(sample['outputs'], is_pretokenized=True).ids
                  for sample in batch]

    # Get the maximum sequence length in the batch
    max_length = max(len(ids) for ids in input_ids + output_ids)

    # Pad the sequences on the left side
    input_ids = [F.pad(torch.tensor(ids), (max_length - len(ids), 0),
                       value=tokenizer.token_to_id(pad_token))
                 for ids in input_ids]
    output_ids = [F.pad(torch.tensor(ids), (max_length - len(ids), 0),
                        value=tokenizer.token_to_id(pad_token))
                  for ids in output_ids]

    # Stack the padded sequences into tensors
    input_ids = torch.stack(input_ids)
    output_ids = torch.stack(output_ids)

    return input_ids, output_ids


def dataloader_info(dataloader, tokenizer):
    ''' Print debug info about data, and get histogram of sequence length. '''
    import matplotlib.pyplot as plt

    # PRINT EXAMPLES
    for batch_idx, batch in enumerate(dataloader):
        input_ids, output_ids = batch

        # a random sample from the batch
        sample_idx = random.randint(0, input_ids.size(0) - 1)
        sample_input_ids = input_ids[sample_idx].tolist()
        sample_output_ids = output_ids[sample_idx].tolist()

        # decode
        #   NOTE: map decode since these ids are symbols in a list, not a
        #         string
        sample_input_tokens = [tokenizer.decode([x], skip_special_tokens=True)
                               for x in sample_input_ids]
        sample_output_tokens = [tokenizer.decode([x], skip_special_tokens=True)
                                for x in sample_output_ids]

        print(f"Batch {batch_idx + 1}:")
        print_grid([
            ['Input Tokens:'] + sample_input_tokens,
            ['Output Tokens:'] + sample_output_tokens,
            ['Input IDs:'] + list(map(str, sample_input_ids)),
            ['Output IDs:'] + list(map(str, sample_output_ids))
        ])
        print()
        if batch_idx >= 4:
            break

    # STATISTICS
    sequence_lengths = []
    batch_sizes = []
    for batch in dataloader:
        input_ids, _ = batch
        batch_sizes.append(input_ids.size(0))
        sequence_lengths.append(input_ids.size(1))

    print("DATALOADER STATISTICS:")
    print(f"number of batches: {len(batch_sizes)}")
    print(f"avg batch size: {sum(batch_sizes) / len(batch_sizes):.2f}")
    print(f"min padded sequence length: {min(sequence_lengths)}")
    print(f"max padded sequence length: {max(sequence_lengths)}")
    print(f"avg padded sequence length: {sum(sequence_lengths) / len(sequence_lengths):.2f}")

    # histogram of sequence lengths
    plt.figure(figsize=(8, 6))
    plt.hist(sequence_lengths, bins=20, edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sequence Lengths')
    plt.show()


def build_tokenizer_dataloader(
        raw_datasets: List[  # multiple datasets
            List[Dict[str, List[str]]]  # a single dataset
        ],
        batch_size,
        unk_token='[UNK]',
        pad_token='[PAD]',
        data_keys: List[str] = None,
):
    ''' Dataset has pretokenization applied, ie it's split into a list of str
    tokens (it has not been tokenized into ints yet). '''
    for data in raw_datasets:
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        assert 'inputs' in data[0]
        assert 'outputs' in data[0]

    # the keys to pay attention to from the dataset
    if data_keys is None:
        keys = raw_datasets[0].keys()
    else:
        keys = data_keys

    # Convert to HF Dataset
    hf_datasets = [Dataset.from_list(x) for x in raw_datasets]

    # Make Tokenizer from Data
    special_tokens = [unk_token, pad_token]
    # init tokenizer
    tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab={},
                                                      unk_token=unk_token))
    print('training tokenizer')
    tokenizer.train_from_iterator(iterator(hf_datasets, keys, special_tokens))
    tokenizer.add_special_tokens([unk_token, pad_token])
    dataloaders = [
        DataLoader(x, batch_size=batch_size,
                   collate_fn=partial(collate_fn, tokenizer, pad_token))
        for x in hf_datasets
    ]

    return tokenizer, dataloaders


##################################################
# TRAINING

def run_epoch(model, dl, optimizer, mode, device, clip=None,
              check_accuracy=False):
    ''' Run an epoch over a DataLoader, and optionally perform greedy sampling
    to check accuracy. '''
    assert mode in ['eval', 'train'], "mode must be either 'eval' or 'train'"
    if mode == 'train':
        model.train()
    else:
        model.eval()

    epoch_loss = 0

    if check_accuracy:
        n_correct = 0
        n = 0
        outputs = []

    with torch.set_grad_enabled(mode == 'train'):
        for i, (src_ids, trg_ids) in enumerate(dl):
            src_ids = src_ids.to(device)  # [batch, seq, vec_size]
            trg_ids = trg_ids.to(device)
            output = model(src_ids)  # [batch, seq, vec_size]
            loss = F.cross_entropy(output.flatten(0, 1),
                                   trg_ids.flatten(),
                                   reduction='mean')
            epoch_loss += loss.item()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            if check_accuracy:
                with torch.no_grad():
                    output_ids = output.argmax(dim=2)
                    outputs.append((src_ids, trg_ids, output_ids))
                    n += trg_ids.shape[0] * trg_ids.shape[1]  # total count
                    n_correct += (output_ids == trg_ids).sum()

    if check_accuracy:
        acc = n_correct / n
        return epoch_loss / len(dl), acc.item(), outputs
    else:
        return epoch_loss / len(dl)
