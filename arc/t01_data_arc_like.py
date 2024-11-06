import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from typing import Dict, Callable


from arc_like.puzzles import (
    Sequence,

    gen_some_blocks, gen_random_pixel_block,

    compose, translate, reflect, colorshift, shrink, expand, endpoints, swap, right_align, reflect_around_pivot, add_pivot, remove_longest_blocks, remove_shortest_blocks, sort_pixels, magnets, gen_one_block, gen_some_pixels, gen_three_blocks, invert_colors, add_bg_noise, repaint_max_block, move_to_pivot, extend_to_pivot, rotate_block_pixels, gen_n_blocks
)
from arc_like.visualization import visualize_datasets



##################################################
# Puzzles

colors = [1, 2, 3, 4, 6, 7, 8, 9]
seq_len = 24

default_train_puzzles = {

    'some translate(1)': compose([gen_some_blocks(colors, seq_len), translate(1)]),
    'some translate(2)': compose([gen_some_blocks(colors, seq_len), translate(2)]),
    'some translate(3)': compose([gen_some_blocks(colors, seq_len), translate(3)]),
    'some translate(4)': compose([gen_some_blocks(colors, seq_len), translate(4)]),

    'some translate(-1)': compose([gen_some_blocks(colors, seq_len), translate(-1)]),
    'some translate(-2)': compose([gen_some_blocks(colors, seq_len), translate(-2)]),
    'some translate(-3)': compose([gen_some_blocks(colors, seq_len), translate(-3)]),
    'some translate(-4)': compose([gen_some_blocks(colors, seq_len), translate(-4)]),


    # 'one translate(1)': compose([gen_one_block(colors, seq_len), translate(1)]),
    # 'one translate(2)': compose([gen_one_block(colors, seq_len), translate(2)]),
    # 'one translate(3)': compose([gen_one_block(colors, seq_len), translate(3)]),
    # 'one translate(4)': compose([gen_one_block(colors, seq_len), translate(4)]),

    # 'one translate(-1)': compose([gen_one_block(colors, seq_len), translate(-1)]),
    # 'one translate(-2)': compose([gen_one_block(colors, seq_len), translate(-2)]),
    # 'one translate(-3)': compose([gen_one_block(colors, seq_len), translate(-3)]),
    # 'one translate(-4)': compose([gen_one_block(colors, seq_len), translate(-4)]),


    # 'pixels translate(1)': compose([gen_some_pixels(colors, seq_len), translate(1)]),
    # 'pixels translate(2)': compose([gen_some_pixels(colors, seq_len), translate(2)]),
    # 'pixels translate(3)': compose([gen_some_pixels(colors, seq_len), translate(3)]),
    # 'pixels translate(4)': compose([gen_some_pixels(colors, seq_len), translate(4)]),

    # 'pixels translate(-1)': compose([gen_some_pixels(colors, seq_len), translate(-1)]),
    # 'pixels translate(-2)': compose([gen_some_pixels(colors, seq_len), translate(-2)]),
    # 'pixels translate(-3)': compose([gen_some_pixels(colors, seq_len), translate(-3)]),
    # 'pixels translate(-4)': compose([gen_some_pixels(colors, seq_len), translate(-4)]),


    'some endpoints': compose([gen_some_blocks(colors, seq_len), endpoints]),
    'one endpoints': compose([gen_one_block(colors, seq_len), endpoints]),
    'three endpoints': compose([gen_three_blocks(colors, seq_len), endpoints]),

    'some infill': compose([gen_some_blocks(colors, seq_len), endpoints, swap]),
    'one infill': compose([gen_one_block(colors, seq_len), endpoints, swap]),
    'three infill': compose([gen_three_blocks(colors, seq_len), endpoints, swap]),


    'some endpoints + translate(1)': compose([gen_some_blocks(colors, seq_len), endpoints, translate(1)]),
    'one endpoints + translate(1)': compose([gen_one_block(colors, seq_len), endpoints, translate(1)]),
    'three endpoints + translate(1)': compose([gen_three_blocks(colors, seq_len), endpoints, translate(1)]),
    'some endpoints + translate(-1)': compose([gen_some_blocks(colors, seq_len), endpoints, translate(1)]),
    'one endpoints + translate(-1)': compose([gen_one_block(colors, seq_len), endpoints, translate(1)]),
    'three endpoints + translate(-1)': compose([gen_three_blocks(colors, seq_len), endpoints, translate(1)]),


    'some infill + translate(1)': compose([gen_some_blocks(colors, seq_len), endpoints, swap, translate(1)]),
    'one infill + translate(1)': compose([gen_one_block(colors, seq_len), endpoints, swap, translate(1)]),
    'three infill + translate(1)': compose([gen_three_blocks(colors, seq_len), endpoints, swap, translate(1)]),
    'some infill + translate(-1)': compose([gen_some_blocks(colors, seq_len), endpoints, swap, translate(1)]),
    'one infill + translate(-1)': compose([gen_one_block(colors, seq_len), endpoints, swap, translate(1)]),
    'three infill + translate(-1)': compose([gen_three_blocks(colors, seq_len), endpoints, swap, translate(1)]),



    'some denoise (0.3)': compose([gen_some_blocks(colors, seq_len), swap, add_bg_noise(0.3, colors), swap]),
    'one denoise (0.3)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap]),
    'three denoise (0.3)': compose([gen_three_blocks(colors, seq_len), swap, add_bg_noise(0.3, colors), swap]),

    'some denoise (0.2)': compose([gen_some_blocks(colors, seq_len), swap, add_bg_noise(0.2, colors), swap]),
    'one denoise (0.2)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.2, colors), swap]),
    'three denoise (0.2)': compose([gen_three_blocks(colors, seq_len), swap, add_bg_noise(0.2, colors), swap]),

    'some denoise (0.1)': compose([gen_some_blocks(colors, seq_len), swap, add_bg_noise(0.1, colors), swap]),
    'one denoise (0.1)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.1, colors), swap]),
    'three denoise (0.1)': compose([gen_three_blocks(colors, seq_len), swap, add_bg_noise(0.1, colors), swap]),



    # 'reflect(seq_len//2)': compose([gen_one_block(colors, seq_len), reflect(24)]),
    # 'colorshift(2)': compose([gen_some_blocks(colors, seq_len), colorshift(2)]),
    # 'translate(1) + reflect(seq_len//2)': compose([gen_some_blocks(colors, seq_len), translate(1), reflect(24)]),
    # 'translate(1) + colorshift(2)': compose([gen_some_blocks(colors, seq_len), translate(1), colorshift(2)]),
    # 'expand(1)': compose([gen_some_blocks(colors, seq_len), expand(1)]),
    # 'expand(1) expand(1)': compose([gen_some_blocks(colors, seq_len), expand(1), expand(1)]),
    # 'expand(1) + colorshift(2)': compose([gen_some_blocks(colors, seq_len), expand(1), colorshift(2)]),
    # 'expand(1) + translate(1)': compose([gen_some_blocks(colors, seq_len), expand(1), translate(1)]),
    # 'shrink': compose([gen_some_blocks(colors, seq_len), shrink]),
    # 'shrink + expand(2)': compose([gen_some_blocks(colors, seq_len), shrink, expand(2)]),
    # 'endpoints': compose([gen_some_blocks(colors, seq_len), endpoints]),
    # 'infill': compose([gen_some_blocks(colors, seq_len), endpoints, swap]),
    # 'expand(1) + endpoints': compose([gen_one_block(colors, seq_len), expand(1), endpoints]),
    # 'endpoints + expand(1)': compose([gen_one_block(colors, seq_len), endpoints, expand(1)]),
    # 'endpoints + expand(4) + endpoints + expand(1)': compose([gen_one_block(colors, seq_len), endpoints, expand(4), endpoints, expand(1)]),
    # 'right_align': compose([gen_some_pixels(colors, seq_len), right_align]),
    # 'denoise': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap]),
    # 'invert_colors': compose([gen_one_block(colors, seq_len), invert_colors]),
    # 'remove_longest_blocks': compose([gen_some_blocks(colors, seq_len), remove_longest_blocks]),
    # 'remove_shortest_blocks': compose([gen_some_blocks(colors, seq_len), remove_shortest_blocks]),
    # 'remove_longest + endpoints': compose([gen_some_blocks(colors, seq_len), remove_longest_blocks, endpoints]),
    # 'reflect-pivot': compose([gen_some_blocks(list(set(colors) - {5})), add_pivot, reflect_around_pivot]),
    # 'reflect-pivot + shrink': compose([gen_one_block(list(set(colors) - {5})), add_pivot, reflect_around_pivot, shrink]),
    # 'repaint-from-max-block': compose([gen_three_blocks(colors, seq_len), repaint_max_block]),
    # 'move_to_pivot': compose([gen_one_block(list(set(colors) - {5})), add_pivot, move_to_pivot]),
    # 'extend_to_pivot': compose([gen_one_block(list(set(colors) - {5})), add_pivot, extend_to_pivot]),
    # 'rotate colored block': compose([gen_random_pixel_block(colors), rotate_block_pixels(1)]),
    # 'sort_pixels': compose([gen_some_pixels(colors[:3], p=0.1), sort_pixels()]),
    # 'magnets': compose([gen_n_blocks(colors, 2), magnets()]),
}

default_val_puzzles = {

    'denoise one translate(1)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap, translate(1)]),
    'denoise one translate(2)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap, translate(2)]),
    'denoise one translate(3)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap, translate(3)]),

    'denoise one translate(-1)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap, translate(-1)]),
    'denoise one translate(-2)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap, translate(-2)]),
    'denoise one translate(-3)': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap, translate(-3)]),


    # 'translate(5)': compose([gen_some_blocks(colors, seq_len), translate(5)]),
    # 'reflect(seq_len//2)': compose([gen_one_block(colors, seq_len), reflect(24)]),
    # 'colorshift(2)': compose([gen_some_blocks(colors, seq_len), colorshift(2)]),
    # 'translate(1) + reflect(seq_len//2)': compose([gen_some_blocks(colors, seq_len), translate(1), reflect(24)]),
    # 'translate(1) + colorshift(2)': compose([gen_some_blocks(colors, seq_len), translate(1), colorshift(2)]),
    # 'expand(1)': compose([gen_some_blocks(colors, seq_len), expand(1)]),
    # 'expand(1) expand(1)': compose([gen_some_blocks(colors, seq_len), expand(1), expand(1)]),
    # 'expand(1) + colorshift(2)': compose([gen_some_blocks(colors, seq_len), expand(1), colorshift(2)]),
    # 'expand(1) + translate(1)': compose([gen_some_blocks(colors, seq_len), expand(1), translate(1)]),
    # 'shrink': compose([gen_some_blocks(colors, seq_len), shrink]),
    # 'shrink + expand(2)': compose([gen_some_blocks(colors, seq_len), shrink, expand(2)]),
    # 'endpoints': compose([gen_some_blocks(colors, seq_len), endpoints]),
    # 'infill': compose([gen_some_blocks(colors, seq_len), endpoints, swap]),
    # 'expand(1) + endpoints': compose([gen_one_block(colors, seq_len), expand(1), endpoints]),
    # 'endpoints + expand(1)': compose([gen_one_block(colors, seq_len), endpoints, expand(1)]),
    # 'endpoints + expand(4) + endpoints + expand(1)': compose([gen_one_block(colors, seq_len), endpoints, expand(4), endpoints, expand(1)]),
    # 'right_align': compose([gen_some_pixels(colors, seq_len), right_align]),
    # 'denoise': compose([gen_one_block(colors, seq_len), swap, add_bg_noise(0.3, colors), swap]),
    # 'invert_colors': compose([gen_one_block(colors, seq_len), invert_colors]),
    # 'remove_longest_blocks': compose([gen_some_blocks(colors, seq_len), remove_longest_blocks]),
    # 'remove_shortest_blocks': compose([gen_some_blocks(colors, seq_len), remove_shortest_blocks]),
    # 'remove_longest + endpoints': compose([gen_some_blocks(colors, seq_len), remove_longest_blocks, endpoints]),
    # 'reflect-pivot': compose([gen_some_blocks(list(set(colors) - {5})), add_pivot, reflect_around_pivot]),
    # 'reflect-pivot + shrink': compose([gen_one_block(list(set(colors) - {5})), add_pivot, reflect_around_pivot, shrink]),
    # 'repaint-from-max-block': compose([gen_three_blocks(colors, seq_len), repaint_max_block]),
    # 'move_to_pivot': compose([gen_one_block(list(set(colors) - {5})), add_pivot, move_to_pivot]),
    # 'extend_to_pivot': compose([gen_one_block(list(set(colors) - {5})), add_pivot, extend_to_pivot]),
    # 'rotate colored block': compose([gen_random_pixel_block(colors), rotate_block_pixels(1)]),
    # 'sort_pixels': compose([gen_some_pixels(colors[:3], p=0.1), sort_pixels()]),
    # 'magnets': compose([gen_n_blocks(colors, 2), magnets()]),
}


##################################################
#

def padded_collate(batch,
                   tuple_keys={},
                   default_pad_value=0,
                   special_pad_value={},
                   pad_side='right',
                   device='cpu'):
    """Pad sequences to the same length using F.pad.  Works with dictionary-based
    datasets with arbitrary keys.

    Args:
    - batch: A list of dictionaries, each containing input, output, and possibly other keys
    - tuple_keys: the keys which contain tuples of data (eg context) where padding must be handled specially
    - default_pad_value: The value to use for padding (default: 0)
    - special_pad_value: dict of keys, and corresponding custom value. EG mask keys should be padded with 0, token sequences padded with <pad>
    - pad_side: 'right' or 'left', determines which side to add padding (default: 'right')
    - length_key: The key to use for determining sequence lengths (default: 'input')

    Returns:
    - A dictionary containing padded and batched tensors for each key in the input dictionaries

    """
    # Combine all keys from all dictionaries in the batch
    keys = set().union(*batch)
    B = len(batch)

    # Pad sequences
    def pad_tensor(x, max_len, pad_value):
        x = torch.tensor(x)
        pad_size = max_len - len(x)
        if pad_side == 'right':
            return F.pad(x, (0, pad_size), value=pad_value)
        elif pad_side == 'left':
            return F.pad(x, (pad_size, 0), value=pad_value)
        else:
            raise ValueError("pad_side must be either 'right' or 'left'")

    # Create a dictionary to store the padded and batched tensors
    padded_batch = {}

    for key in keys:
        pad_value = special_pad_value[key] if key in special_pad_value else default_pad_value
        if key in tuple_keys:
            # contexts: [(context 1, context 2, ...), ...]
            #   list shapes: [B, N_FEWSHOT, S]
            contexts = [item[key] for item in batch]
            n_fewshot = len(contexts[0])

            # transpose context
            transposed = [[] for _ in range(n_fewshot)]  # [N_FEWSHOT, B, S]
            for shots in contexts:
                assert len(shots) == n_fewshot
                for i, shot in enumerate(shots):
                    transposed[i].append(shot)
            assert len(transposed) == n_fewshot

            # pad contexts
            padded_context = []
            for shot in transposed:
                assert len(shot) == B
                max_len = max([len(item) for item in shot])
                p = torch.stack([pad_tensor(item, max_len, pad_value) for item in shot])
                assert p.shape[0] == B
                padded_context.append(p.to(device))
            padded_batch[key] = torch.stack(padded_context, dim=0)  # [N_FEWSHOT, B, S]
        else:
            max_len = max([len(item[key]) for item in batch])
            padded_batch[key] = torch.stack([pad_tensor(item[key], max_len, pad_value) for item in batch]).to(device)
    return padded_batch


##################################################
# Data

def token_to_id(tokenizer, tok):
    assert isinstance(tok, str)
    i = tokenizer.token_to_id(tok)
    if i is None:
        raise ValueError(f'token "{tok}" not found in tokenizer vocabulary')
    return i


def few_shot(tokenizer, puzzle: Callable[[Sequence], Sequence], n_few_shot, seq_len, eq_token_id):
    '''
    Build a dataset from a generator, returns `n_few_shot` context input-output pairs from `gen` and one final input-output pair.
    n_few_shot: number of shots
    gen: generator
    '''

    # Context
    context_inputs = ()  # empty tuple to hold each separate context
    for _ in range(n_few_shot):
        # ins, outs = gen(seq_len)
        seq = Sequence([], [], None)
        seq = puzzle(seq)
        ins = seq.inputs
        outs = seq.outputs
        context = (
            list(map(lambda x: token_to_id(tokenizer, str(x)), ins)) +  # int data -> str -> token
            [eq_token_id] +
            list(map(lambda x: token_to_id(tokenizer, str(x)), outs))  # int data -> str -> token
        )
        context_inputs += (context,)

    # Challenge
    # i, o = gen(seq_len)
    seq = Sequence([], [], None)
    seq = puzzle(seq)
    i = list(map(lambda x: token_to_id(tokenizer, str(x)), seq.inputs))
    o = list(map(lambda x: token_to_id(tokenizer, str(x)), seq.outputs))
    return {
        'context_inputs': context_inputs,  # nshot sequences, ie a list of list of ids
        'context_inputs_mask': tuple([1] * len(x) for x in context_inputs),
        'inputs': i,
        'inputs_mask': [1] * len(i),
        'outputs': o,
    }


def generate_fewshot_datasets(tokenizer, puzzles, num_samples: int, seq_len: int, eq_token_id):
    ''' make the datasets few-shot, for in-context learning '''
    n_few_shot = 2
    datasets = {}
    for (dataset_name, puzzle) in puzzles.items():
        data = []
        for _ in range(num_samples):
            d = few_shot(tokenizer, puzzle, n_few_shot, seq_len, eq_token_id)
            data.append(d)

        datasets[dataset_name] = data
    return datasets


def build_fewshot_dataloaders(tokenizer, batch_size, num_samples, seq_len, eq_token_id, device):
    ''' few-shot datsets, for in-context learning '''
    num_workers = 0  # 0 means data will be loaded in the main process
    collate_fn = lambda batch: padded_collate(batch, {'context_inputs', 'context_inputs_mask'},
                                              default_pad_value=tokenizer.pad_token_id,
                                              special_pad_value={'context_inputs_mask': 0},
                                              pad_side='right',
                                              device=device)


    train_dataset = ConcatDataset(generate_fewshot_datasets(tokenizer, default_train_puzzles, num_samples, seq_len, eq_token_id).values())
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    val_dataset = ConcatDataset(generate_fewshot_datasets(tokenizer, default_val_puzzles, num_samples, seq_len, eq_token_id).values())
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_dl, val_dl

##########

def generate_datasets(tokenizer, puzzles, num_samples, seq_len, device):
    ''' input-output only, not fewshot '''
    datasets = {}
    for dataset_name, puzzle in puzzles.items():
        data = []
        for _ in range(num_samples):
            seq = Sequence([], [], None)
            seq = puzzle(seq)
            inputs = list(map(lambda x: token_to_id(tokenizer, str(x)), seq.inputs))
            outputs = list(map(lambda x: token_to_id(tokenizer, str(x)), seq.outputs))
            data.append({
                'inputs': inputs,
                'inputs_mask': [1] * len(inputs),
                'outputs': outputs,
                'outputs_mask': [1] * len(outputs)
            })
        datasets[dataset_name] = data
    return datasets


def build_dataloaders(tokenizer, batch_size, num_samples, seq_len, device, train_puzzles=None, val_puzzles=None):
    """
    Build dataloaders without few-shot examples, just inputs and outputs.

    Args:
    - tokenizer: The tokenizer to use for encoding
    - batch_size: Batch size for the dataloaders
    - num_samples: Number of samples to generate for each puzzle
    - seq_len: Maximum sequence length
    - device: Device to load the data onto ('cpu' or 'cuda')

    Returns:
    - train_dl: DataLoader for training data
    - val_dl: DataLoader for validation data
    """
    if train_puzzles is None:
        train_puzzles = default_train_puzzles
    if val_puzzles is None:
        val_puzzles = default_val_puzzles

    num_workers = 0  # 0 means data will be loaded in the main process
    collate_fn = lambda batch: padded_collate(
        batch,
        default_pad_value=tokenizer.pad_token_id,
        special_pad_value={'inputs_mask': 0, 'outputs_mask': 0},
        pad_side='right',
        device=device
    )

    train_dataset = ConcatDataset(generate_datasets(tokenizer, train_puzzles, num_samples, seq_len, device).values())
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    val_dataset = ConcatDataset(generate_datasets(tokenizer, val_puzzles, num_samples, seq_len, device).values())
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dl, val_dl


##########
# Debug datasets


# Non-fewshot datasets
if False:
    datasets = {}
    num_samples = 10
    grid_width = 7
    grid_height = 5
    for (name, transformer) in default_train_puzzles.items():
        all_inputs, all_outputs = [], []
        for _ in range(num_samples):
            seq = Sequence([], [], None)
            seq = transformer(seq)
            all_inputs.append(seq.inputs)
            all_outputs.append(seq.outputs)
            inputs_tensor, outputs_tensor = torch.tensor(all_inputs), torch.tensor(all_outputs)
            datasets[name] = TensorDataset(inputs_tensor, outputs_tensor)

    visualize_datasets(datasets, grid_width=grid_width, grid_height=grid_height, num_samples=num_samples)

# Fewshot datasets
if False:
    import tokenizers
    from tokenizers import Tokenizer, models, pre_tokenizers

    # special tokens
    EQ_TOKEN = '='
    EOS_TOKEN = '</s>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    # custom vocabulary (to keep colors aligned with tokenizer ids)
    tokens = [str(i) for i in range(21)]  # 0 to 20
    special_tokens = [EQ_TOKEN, UNK_TOKEN, EOS_TOKEN, PAD_TOKEN]
    vocab = {token: i for i, token in enumerate(tokens + special_tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(special_tokens)

    EOS_TOKEN_ID = token_to_id(tokenizer, EOS_TOKEN)
    PAD_TOKEN_ID = token_to_id(tokenizer, PAD_TOKEN)
    UNK_TOKEN_ID = token_to_id(tokenizer, UNK_TOKEN)
    EQ_TOKEN_ID = token_to_id(tokenizer, '=')
    tokenizer.pad_token_id = PAD_TOKEN_ID

    train_dl, val_dl = build_fewshot_dataloaders(tokenizer, batch_size=10, seq_len=32, num_samples=10, eq_token_id=EQ_TOKEN_ID, device='cuda')

    # collect from batch in visualizable format
    all_context_in = []
    all_context_out = []
    all_inputs = []
    all_outputs = []
    for batch in train_dl:
        all_inputs += batch['inputs'].tolist()
        all_outputs += batch['outputs'].tolist()
        tcontext = batch['context_inputs']
        n_context = tcontext.shape[0]
        B = tcontext.shape[1]
        cis = []
        cos = []
        for b in range(B):
            ci = []
            co = []
            for c in range(n_context):
                # split on EQ token
                ix = (tcontext[c, b] == EQ_TOKEN_ID).float().argmax().item()
                ci += tcontext[c, b, :ix].tolist()
                co += tcontext[c, b, ix + 1:].tolist()
            cis.append(ci)
            cos.append(co)
        all_context_in += cis
        all_context_out += cos

    datasets = {
        'inp-out': TensorDataset(torch.tensor(all_inputs), torch.tensor(all_outputs)),
        'context': TensorDataset(torch.tensor(all_context_in), torch.tensor(all_context_out))
    }
    visualize_datasets(datasets, grid_width=2, grid_height=1, num_samples=10)
