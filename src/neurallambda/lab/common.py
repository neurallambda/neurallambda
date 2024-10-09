
import torch
import random
import torch.nn.functional as F
from datasets import Dataset
from functools import partial
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import tokenizers


##################################################
# DATA

def print_grid(data: List[List[Any]]) -> None:
    """
    Prints a grid of data with evenly spaced columns.

    Args:
        data (List[List[Any]]): A 2D list containing the data to be printed.
            Each sublist represents a row of the grid.

    Returns:
        None

    Example:
        data = [
            [1, "Apple", 3.14],
            [2, "Banana", 2.71],
            [3, "Orange", 1.41]
        ]
        print_grid(data)

    Output:
        1  Apple   3.14
        2  Banana  2.71
        3  Orange  1.41
    """
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
                     f'grad:{param.requires_grad}',
                     f'device:{param.device}'),
                    )
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
    accuracy_mask = [x['accuracy_mask'] for x in batch]

    # Get the maximum sequence length in the batch
    max_length = max(len(ids) for ids in input_ids + output_ids)

    # Pad the sequences on the left side
    input_ids = [F.pad(torch.tensor(ids), (max_length - len(ids), 0),
                       value=tokenizer.token_to_id(pad_token))
                 for ids in input_ids]
    output_ids = [F.pad(torch.tensor(ids), (max_length - len(ids), 0),
                        value=tokenizer.token_to_id(pad_token))
                  for ids in output_ids]
    accuracy_mask = [F.pad(torch.tensor(x, dtype=torch.bool), (max_length - len(x), 0),
                           value=False)
                     for x in accuracy_mask]


    # Stack the padded sequences into tensors
    input_ids = torch.stack(input_ids)
    output_ids = torch.stack(output_ids)
    accuracy_mask = torch.stack(accuracy_mask)

    return input_ids, output_ids, accuracy_mask


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
            List[Dict[str, List[str]]]  # a single dataset, eg [{inputs:..., outputs:...}]
        ],
        unk_token='[UNK]',
        pad_token='[PAD]',
        data_keys: List[str] = None,
):
    '''Dataset has pretokenization applied, ie it's split into a list of str
    tokens (it has not been tokenized into ints yet).

    This helper is highly specific to toy problems, where we eliminate
    tokenization issues by pretokenizing, IE, for an input sequence of
    "abc|cba", we need to have already pretokenized it into:

    ['a', 'b', 'c', '|', 'c', 'b', 'a']

    Returns:
      DataLoader creator fn, that accepts a batch_size

    '''
    for data in raw_datasets:
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        assert 'inputs' in data[0]
        assert 'outputs' in data[0]
        assert 'accuracy_mask' in data[0]

    # the keys to pay attention to from the dataset
    if data_keys is None:
        data_keys = list(raw_datasets[0][0].keys())  # first data split, first row
        data_keys = list(filter(lambda x: x not in {'accuracy_mask'}, data_keys))

    # Convert to HF Dataset
    hf_datasets = [Dataset.from_list(x) for x in raw_datasets]

    # Make Tokenizer from Data
    special_tokens = [unk_token, pad_token]
    # init tokenizer
    tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab={},
                                                      unk_token=unk_token))
    print('training tokenizer')
    tokenizer.train_from_iterator(iterator(hf_datasets, data_keys, special_tokens))
    tokenizer.add_special_tokens([unk_token, pad_token])

    def create_dataloaders(batch_size: int) -> List[DataLoader]:
        dataloaders = [
            DataLoader(x, batch_size=batch_size,
                       collate_fn=partial(collate_fn, tokenizer, pad_token))
            for x in hf_datasets
        ]
        return dataloaders

    return tokenizer, create_dataloaders


##################################################
# TRAINING

def run_epoch(model, dl, optimizer, mode, device, loss_fn, clip=None,
              check_accuracy=False, regularization_fn=None):
    ''' Run an epoch over a DataLoader, and optionally perform greedy sampling
    to check accuracy.

    Args:
      if loss_fn == 'cosine_distance', model must return probabilities (after sigmoid activation)
      if loss_fn == 'cross_entropy', model must return unnormalized logits (ie final output comes from Linear layer)
      if loss_fn == 'nllloss', model must return logprobs, ie `F.log_softmax(..., dim=-1)`
    '''
    assert mode in ['eval', 'train'], "mode must be either 'eval' or 'train'"
    assert loss_fn in ['cosine_distance', 'nllloss', 'cross_entropy'], "loss_fn must be 'cosine_distance', 'nllloss', or 'cross_entropy'"
    assert clip is None or isinstance(clip, float)

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
        for i, (src_ids, trg_ids, acc_mask) in enumerate(dl):
            src_ids = src_ids.to(device)  # [batch, seq, vec_size]
            trg_ids = trg_ids.to(device)
            output = model(src_ids)  # [batch, seq, vec_size]

            if loss_fn == 'cosine_distance':
                # trgs = torch.stack([model.embeddings(x) for x in trg_ids], dim=1)
                trgs = model.embeddings(trg_ids)
                sim = torch.cosine_similarity(
                    output.flatten(0, 1),
                    trgs.flatten(0, 1), dim=1)
                loss = (1 - sim).mean()

            elif loss_fn == 'cross_entropy':
                loss = F.cross_entropy(output.flatten(0, 1),
                                       trg_ids.flatten(),
                                       reduction='mean')

            elif loss_fn == 'nllloss':
                # log_probs = F.log_softmax(output, dim=-1)
                loss = F.nll_loss(output.flatten(0, 1),
                                  trg_ids.flatten(),
                                  reduction='mean')

            if regularization_fn is not None:
                loss += regularization_fn(model, output)

            epoch_loss += loss.item()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                if isinstance(clip, float):  # grad_clip
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            if check_accuracy:
                with torch.no_grad():
                    if loss_fn == 'cosine_distance':
                        # find the embedding closest to the output, consider
                        # that the output_id
                        output_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0), # [1, 1, VOCAB, EMB_DIM]
                                                             output.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                                                             dim=3).argmax(dim=2)
                    else:
                        output_ids = output.argmax(dim=2)
                    outputs.append((src_ids, trg_ids, acc_mask, output_ids))
                    n += trg_ids[acc_mask].numel()  # total count
                    n_correct += (output_ids[acc_mask] == trg_ids[acc_mask]).sum()

    if check_accuracy:
        acc = n_correct / n
        return epoch_loss / len(dl), acc.item(), outputs
    else:
        return epoch_loss / len(dl)



# def run_epoch(model, dl, optimizer, mode, device, clip=None,
#               check_accuracy=False, loss_fn='nllloss', debug=False):
#     ''' Run an epoch over a DataLoader, and optionally perform greedy sampling
#     to check accuracy.

#     Args:
#       if loss_fn == 'cosine_distance', model must return probabilities (after sigmoid activation)
#       if loss_fn == 'cross_entropy', model must return unnormalized logits (ie final output comes from Linear layer)
#       if loss_fn == 'nllloss', model must return logprobs, ie `F.log_softmax(..., dim=-1)`
#     '''
#     assert mode in ['eval', 'train'], "mode must be either 'eval' or 'train'"
#     assert loss_fn in ['cosine_distance', 'nllloss', 'cross_entropy'], "loss_fn must be 'cosine_distance', 'nllloss', or 'cross_entropy'"

#     if mode == 'train':
#         model.train()
#     else:
#         model.eval()

#     epoch_loss = 0

#     if check_accuracy:
#         n_correct = 0
#         n = 0
#         outputs = []

#     with torch.set_grad_enabled(mode == 'train'):
#         for i, (src_ids, trg_ids, acc_mask) in enumerate(dl):
#             src_ids = src_ids.to(device)  # [batch, seq, vec_size]
#             trg_ids = trg_ids.to(device)
#             output = model(src_ids)  # [batch, seq, vec_size]

#             if loss_fn == 'cosine_distance':
#                 # trgs = torch.stack([model.embeddings(x) for x in trg_ids], dim=1)
#                 trgs = model.embeddings(trg_ids)
#                 sim = torch.cosine_similarity(
#                     output.flatten(0, 1),
#                     trgs.flatten(0, 1), dim=1)
#                 loss = (1-sim).mean()

#             elif loss_fn == 'cross_entropy':
#                 loss = F.cross_entropy(output.flatten(0, 1),
#                                        trg_ids.flatten(),
#                                        reduction='mean')

#             elif loss_fn == 'nllloss':
#                 # log_probs = F.log_softmax(output, dim=-1)
#                 loss = F.nll_loss(output.flatten(0, 1),
#                                   trg_ids.flatten(),
#                                   reduction='mean')

#             epoch_loss += loss.item()

#             if mode == 'train':
#                 optimizer.zero_grad()
#                 loss.backward()

#                 if debug and i==0:
#                     print(f"Loss: {loss.item()}")
#                     for name, param in model.named_parameters():
#                         if param.grad is not None:
#                             print(f"Layer: {name}, Largest Gradient: {param.grad.abs().max().item()}")

#                 if clip is not None:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#                 optimizer.step()

#             if check_accuracy:
#                 with torch.no_grad():
#                     if loss_fn == 'cosine_distance':
#                         # find the embedding closest to the output, consider
#                         # that the output_id
#                         output_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0), # [1, 1, VOCAB, EMB_DIM]
#                                                              output.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
#                                                              dim=3).argmax(dim=2)
#                     else:
#                         output_ids = output.argmax(dim=2)
#                     outputs.append((src_ids, trg_ids, acc_mask, output_ids))
#                     n += trg_ids[acc_mask].numel()  # total count
#                     n_correct += (output_ids[acc_mask] == trg_ids[acc_mask]).sum()

#     if check_accuracy:
#         acc = n_correct / n
#         return epoch_loss / len(dl), acc.item(), outputs
#     else:
#         return epoch_loss / len(dl)
