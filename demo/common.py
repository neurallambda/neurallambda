'''

* Build datasets

* Tokenizer and Embedding

* Loss

* Training

'''

import tokenizers
from tokenizers import Tokenizer, AddedToken
from typing import List, Any, Dict
import datasets
import torch
import random
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from functools import partial

def print_grid(data):
    # maximum width for each column
    column_widths = [max(len(str(item)) for item in column) for column in zip(*data)]
    for row in data:
        formatted_row = "  ".join(str(item).ljust(width) for item, width in zip(row, column_widths))
        print(formatted_row)


def iterator(all_datasets: List[Dataset], keys: List[str], special_tokens):
    ''' Iterate over a list of datasets and build a vocab for them. '''
    batch_size = 1000
    for tok in special_tokens:
        yield tok
    for dataset in all_datasets:
        for xs in dataset:
            for key in keys:
                for x in xs[key]:
                    yield x


def collate_fn(pad_token, batch):
    ''' Tokenize and pad batches of data. '''
    input_ids = [tokenizer.encode(sample['inputs'], is_pretokenized=True).ids for sample in batch]
    output_ids = [tokenizer.encode(sample['outputs'], is_pretokenized=True).ids for sample in batch]

    # Get the maximum sequence length in the batch
    max_length = max(len(ids) for ids in input_ids + output_ids)

    # Pad the sequences on the left side
    input_ids = [F.pad(torch.tensor(ids), (max_length - len(ids), 0), value=tokenizer.token_to_id(pad_token)) for ids in input_ids]
    output_ids = [F.pad(torch.tensor(ids), (max_length - len(ids), 0), value=tokenizer.token_to_id(pad_token)) for ids in output_ids]

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
        #   NOTE: map decode since these ids are symbols in a list, not a string
        sample_input_tokens  = [tokenizer.decode([x], skip_special_tokens=True) for x in sample_input_ids]
        sample_output_tokens = [tokenizer.decode([x], skip_special_tokens=True) for x in sample_output_ids]

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
        raw_datasets: List[ # multiple datasets
            List[Dict[str, List[str]]] # a single dataset
        ]):
    for data in raw_datasets:
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        assert 'inputs' in data[0]
        assert 'outputs' in data[0]


    # Convert to HF Dataset
    hf_datasets = [Dataset.from_list(x) for x in raw_datasets]

    # Make Tokenizer from Data
    UNK_TOKEN = '[UNK]'
    PAD_TOKEN = '[PAD]'
    special_tokens = [UNK_TOKEN, PAD_TOKEN]
    # init tokenizer
    tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab={}, unk_token=UNK_TOKEN))
    print('training tokenizer')
    tokenizer.train_from_iterator(iterator(hf_datasets, ['inputs', 'outputs'], special_tokens))
    tokenizer.add_special_tokens([UNK_TOKEN, PAD_TOKEN])
    # a = tokenizer.encode(['1', '2', '3'], is_pretokenized=True)
    # print_grid([['tokens:'] + a.tokens,
    #             ['ids:'] + a.ids])
    dataloaders = [
        DataLoader(
            x,
            batch_size=BATCH_SIZE,
            collate_fn=partial(collate_fn, PAD_TOKEN)
        ) for x in hf_datasets]

    return tokenizer, dataloaders



##################################################
# sandbox

REFLECT_SYMBOL = '|'
PAUSE_SYMBOL = '.'

def palindrome(num_samples, max_length, lang) -> Dict[str, List[str]]:
    data = []
    for _ in range(num_samples):
        length = random.randint(5, max_length)
        hlength = length // 2
        seed = [random.choice(lang) for _ in range(hlength)]
        # add pauses to inputs and outputs
        inputs  = seed + [REFLECT_SYMBOL] + [PAUSE_SYMBOL] * hlength
        outputs = [PAUSE_SYMBOL] * (hlength + 1) + seed[::-1]
        # convert all symbols to str
        inputs  = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'seed': seed,
            'inputs': inputs,
            'outputs': outputs,
        })
    return data

TRAIN_NUM_SAMPLES = 1000
TRAIN_MAX_SEQUENCE_LENGTH = 100

VAL_NUM_SAMPLES = 100
VAL_MAX_SEQUENCE_LENGTH = 100

BATCH_SIZE = 10


# train_lang = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
train_lang = 'a b c d e'.split(' ')
val_lang = 'f g h i j'.split(' ')

train_raw = palindrome(TRAIN_NUM_SAMPLES, TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang)
train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))

val_raw = palindrome(VAL_NUM_SAMPLES, VAL_MAX_SEQUENCE_LENGTH, lang=val_lang)
val_raw = sorted(val_raw, key=lambda x: len(x['inputs']))

tokenizer, (train_loader, val_loader) = build_tokenizer_dataloader([train_raw, val_raw])

dataloader_info(train_loader, tokenizer)
dataloader_info(val_loader, tokenizer)
