'''

Train a Serial Transformer on the 1D ARC-like data


INTERMEDIATE GOAL:

Can we generalize to programs that are compositions of primitives?

Upgrade arc-like and create several datasets that use 1-3 combinators. Test on unique combinations.

(?) Build simple arc-like primitives and variants:
  shift LR
  mirror
  denoise
  identify start/end (move-to, flip, magnet)
(?) Pretrain on autoencoder and single combinators
(?) Fine tune just LSTM on 2 combinators and CQ/CK encoders
(?) ignore training CQ/CK encoders at some step?
(?) Test on unique combinations
(?) self-prediction (graziano paper)

TODO:

- [ ] start with just translate, and generalize to an unseen length
- [ ] cleanup
- [ ] refactor?
- [ ] improve combinator additions to arc-like
- [ ] Step 1: train and save autoencoder for decoder only, without using CQ/CK encoders
- [ ] Step 2: Finetune just CQ/CK encoders and LSTM of decoder
- [ ] Step 3: test on unique combos

'''

import os
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import t01_data_1d as data
import t02_train_1d_serial_attn as Attention
from torch.utils.data import ConcatDataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as Q

import tokenizers
from tokenizers import Tokenizer

from neurallambda.lab.common import (
    print_model_info,
    print_grid
)

import warnings

SEED = 152
torch.manual_seed(152)
random.seed(SEED)

num_samples = 400
seq_len = 32
N_SHOTS = 2

batch_size = 128

NUM_EPOCHS = 20
LR = 1e-3
WD = 0.0
GRAD_CLIP = None
LOSS_FN = 'cross_entropy'
DEVICE = 'cuda'


##################################################
# Tokenizer

EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
tokens = '= -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'.split(' ')

special_tokens = [UNK_TOKEN, EOS_TOKEN, PAD_TOKEN]
tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab={},
                                                  unk_token=UNK_TOKEN))
tokenizer.train_from_iterator(tokens + special_tokens)
tokenizer.add_special_tokens(special_tokens)

def token_to_id(tok):
    assert isinstance(tok, str)
    i = tokenizer.token_to_id(tok)
    if i is None:
        raise ValueError(f'token "{tok}" not found in tokenizer vocabulary')
    return i

EOS_TOKEN_ID = token_to_id(EOS_TOKEN)
PAD_TOKEN_ID = token_to_id(PAD_TOKEN)
UNK_TOKEN_ID = token_to_id(UNK_TOKEN)
EQ_TOKEN_ID = token_to_id('=')

tokenizer.pad_token_id = PAD_TOKEN_ID


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
                padded_context.append(p.to(DEVICE))
            padded_batch[key] = torch.stack(padded_context, dim=0)  # [N_FEWSHOT, B, S]
        else:
            max_len = max([len(item[key]) for item in batch])
            padded_batch[key] = torch.stack([pad_tensor(item[key], max_len, pad_value) for item in batch]).to(DEVICE)
    return padded_batch


def generate_datasets(num_samples: int, seq_len: int, generators: Dict[str, Callable]):
    datasets = {}
    for dataset_name, generator in generators.items():
        data = []
        for _ in range(num_samples):
            d = generator(seq_len)
            data.append(d)

        datasets[dataset_name] = data
    return datasets


##################################################
# Data

def few_shot(n, gen):
    '''
    Build a dataset from a generator, returns `n` context input-output pairs from `gen` and one final input-output pair.
    n: number of shots
    gen: generator
    '''
    def f(seq_len):
        ''' Upgraded version of `gen` that uses it to also generate context, and format things into a dict '''
        # Context
        context_inputs = ()  # empty tuple to hold each separate context
        for _ in range(n):
            ins, outs = gen(seq_len)
            context = (
                list(map(lambda x: token_to_id(str(x)), ins)) +  # int data -> str -> token
                [EQ_TOKEN_ID] +
                list(map(lambda x: token_to_id(str(x)), outs))  # int data -> str -> token
            )
            context_inputs += (context,)

        # Challenge
        i, o = gen(seq_len)
        return {
            'context_inputs': context_inputs,  # nshot sequences, ie a list of list of ids
            'context_inputs_mask': tuple([1] * len(x) for x in context_inputs),
            'inputs': i,
            'inputs_mask': [1] * len(i),
            'outputs': o,
        }
    return f

MAX_DIGITS = 10
recolor_odd_even_map = {'odd': 1, 'even': 2}
recolor_size_map = {x: x for x in range(MAX_DIGITS)}

train_generators = {
    # Pre-training

    "Random Identity": few_shot(N_SHOTS, lambda s: data.gen_random_identity(s, max_digits=MAX_DIGITS, background=0)),
    "Identity": few_shot(N_SHOTS, lambda s: data.gen_identity_block(s, min_len=2, max_len=5, max_digits=MAX_DIGITS, background=0)),
    "Identity Multi": few_shot(N_SHOTS, lambda s: data.gen_multi_identity(s, num_blocks=3, min_len=2, max_len=5, max_digits=MAX_DIGITS, background=0)),
    "Single Color Denoising": few_shot(N_SHOTS, lambda s: data.gen_single_1c_denoising(s, max_digits=MAX_DIGITS, background=0)),
    "Multi-Color Denoising": few_shot(N_SHOTS, lambda s: data.gen_single_mc_denoising(s, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),


    # Translation Tasks
    "Move 1": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=1, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    "Move 2": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=2, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    "Move 3": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=3, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    "Move 4": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=4, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    "Move 5": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=5, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    "Move 6": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=6, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move 7": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=7, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),


#     # Tests

#     "Basic Fill": few_shot(N_SHOTS, lambda s: data.gen_basic_fill(s, min_len=8, max_len=32, min_hole_len=1, max_digits=MAX_DIGITS, background=0)),
#     # "Multi Fill": few_shot(N_SHOTS, lambda s: data.gen_multi_fill(s, min_len=8, max_len=32, min_hole_len=1, max_digits=MAX_DIGITS, background=0)),
#     "Hollow": few_shot(N_SHOTS, lambda s: data.gen_hollow(s, min_len=8, max_len=32, min_hole_len=1, max_digits=MAX_DIGITS, background=0)),
#     "Single Flip": few_shot(N_SHOTS, lambda s: data.gen_single_flip(s, min_len=8, max_len=32, max_digits=MAX_DIGITS, background=0)),
#     "Mirror": few_shot(N_SHOTS, lambda s: data.gen_single_mirror(s, min_len=12, max_len=32, max_digits=MAX_DIGITS, background=0, pivot_pt=9)),
#     "Move Single Bar": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=3, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
#     "Move to Pixel": few_shot(N_SHOTS, lambda s: data.gen_move_to_pixel(s, min_len=8, max_len=32, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
#     "Move Towards Indicator": few_shot(N_SHOTS, lambda s: data.gen_move_towards_indicator(s, min_len=8, max_len=32, max_move=2, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
#     "Extend to Pixel": few_shot(N_SHOTS, lambda s: data.gen_extend_to_pixel(s, min_len=8, max_len=32, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
#     "Pattern Copy (Single Color)": few_shot(N_SHOTS, lambda s: data.gen_grow_copy_color(s, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
#     "Pattern Copy (Multi-Color)": few_shot(N_SHOTS, lambda s: data.gen_grow(s, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
#     "Recolor Odd-Even": few_shot(N_SHOTS, lambda s: data.gen_recolor_odd_even(s, recolor_odd_even_map, min_len=12, max_len=33, max_digits=MAX_DIGITS, background=0)),
#     "Recolor by Size Count": few_shot(N_SHOTS, lambda s: data.gen_recolor_size_cnt(s, recolor_size_map, min_len=12, max_len=33, max_digits=MAX_DIGITS, background=0)),
#     "Recolor by Max": few_shot(N_SHOTS, lambda s: data.gen_recolor_max(s, max_color=1, min_len=17, max_len=33, max_digits=MAX_DIGITS, background=0)),
#     "Sort Colors": few_shot(N_SHOTS, lambda s: data.gen_scattered_pixels_sorted(s, num_pixels=3, min_len=32, max_len=33, max_digits=3, background=0)),
#     "Stack Pixels": few_shot(N_SHOTS, lambda s: data.gen_stack(s, max_pixels=5, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
#     "Overlap Spread": few_shot(N_SHOTS, lambda s: data.gen_overlap_spread(s, max_blocks=3, block_len=5, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
#     "Color Rotation": few_shot(N_SHOTS, lambda s: data.gen_color_rotation(s, min_len=8, max_len=32, max_digits=MAX_DIGITS, background=0)),
#     "Magnets": few_shot(N_SHOTS, lambda s: data.gen_magnets(s, min_block_size=2, max_block_size=5, max_digits=MAX_DIGITS, background=0)),

}

val_generators = {
    # Pre-training

    # "Random Identity": few_shot(N_SHOTS, lambda s: data.gen_random_identity(s, max_digits=MAX_DIGITS, background=0)),
    # "Identity": few_shot(N_SHOTS, lambda s: data.gen_identity_block(s, min_len=2, max_len=5, max_digits=MAX_DIGITS, background=0)),
    # "Identity Multi": few_shot(N_SHOTS, lambda s: data.gen_multi_identity(s, num_blocks=3, min_len=2, max_len=5, max_digits=MAX_DIGITS, background=0)),
    # "Single Color Denoising": few_shot(N_SHOTS, lambda s: data.gen_single_1c_denoising(s, max_digits=MAX_DIGITS, background=0)),
    # "Multi-Color Denoising": few_shot(N_SHOTS, lambda s: data.gen_single_mc_denoising(s, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),


    # Translation Tasks
    # "Move 1": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=1, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move 2": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=2, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move 3": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=3, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move 4": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=4, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move 5": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=5, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move 6": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=6, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    "Move 7": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=7, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),

    # Tests

    # "Basic Fill": few_shot(N_SHOTS, lambda s: data.gen_basic_fill(s, min_len=8, max_len=32, min_hole_len=1, max_digits=MAX_DIGITS, background=0)),
    # "Multi Fill": few_shot(N_SHOTS, lambda s: data.gen_multi_fill(s, min_len=8, max_len=32, min_hole_len=1, max_digits=MAX_DIGITS, background=0)),
    # "Hollow": few_shot(N_SHOTS, lambda s: data.gen_hollow(s, min_len=8, max_len=32, min_hole_len=1, max_digits=MAX_DIGITS, background=0)),
    # "Single Flip": few_shot(N_SHOTS, lambda s: data.gen_single_flip(s, min_len=8, max_len=32, max_digits=MAX_DIGITS, background=0)),
    # "Mirror": few_shot(N_SHOTS, lambda s: data.gen_single_mirror(s, min_len=12, max_len=32, max_digits=MAX_DIGITS, background=0, pivot_pt=9)),
    # "Move Single Bar": few_shot(N_SHOTS, lambda s: data.gen_move_single_bar(s, min_len=8, max_len=32, move_len=3, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move to Pixel": few_shot(N_SHOTS, lambda s: data.gen_move_to_pixel(s, min_len=8, max_len=32, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Move Towards Indicator": few_shot(N_SHOTS, lambda s: data.gen_move_towards_indicator(s, min_len=8, max_len=32, max_move=2, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Extend to Pixel": few_shot(N_SHOTS, lambda s: data.gen_extend_to_pixel(s, min_len=8, max_len=32, min_bar_len=3, max_digits=MAX_DIGITS, background=0)),
    # "Pattern Copy (Single Color)": few_shot(N_SHOTS, lambda s: data.gen_grow_copy_color(s, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
    # "Pattern Copy (Multi-Color)": few_shot(N_SHOTS, lambda s: data.gen_grow(s, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
    # "Recolor Odd-Even": few_shot(N_SHOTS, lambda s: data.gen_recolor_odd_even(s, recolor_odd_even_map, min_len=12, max_len=33, max_digits=MAX_DIGITS, background=0)),
    # "Recolor by Size Count": few_shot(N_SHOTS, lambda s: data.gen_recolor_size_cnt(s, recolor_size_map, min_len=12, max_len=33, max_digits=MAX_DIGITS, background=0)),
    # "Recolor by Max": few_shot(N_SHOTS, lambda s: data.gen_recolor_max(s, max_color=1, min_len=17, max_len=33, max_digits=MAX_DIGITS, background=0)),
    # "Sort Colors": few_shot(N_SHOTS, lambda s: data.gen_scattered_pixels_sorted(s, num_pixels=3, min_len=32, max_len=33, max_digits=3, background=0)),
    # "Stack Pixels": few_shot(N_SHOTS, lambda s: data.gen_stack(s, max_pixels=5, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
    # "Overlap Spread": few_shot(N_SHOTS, lambda s: data.gen_overlap_spread(s, max_blocks=3, block_len=5, min_len=32, max_len=33, max_digits=MAX_DIGITS, background=0)),
    # "Color Rotation": few_shot(N_SHOTS, lambda s: data.gen_color_rotation(s, min_len=8, max_len=32, max_digits=MAX_DIGITS, background=0)),
    # "Magnets": few_shot(N_SHOTS, lambda s: data.gen_magnets(s, min_block_size=2, max_block_size=5, max_digits=MAX_DIGITS, background=0)),
}


shuffle = True
num_workers = 0  # 0 means data will be loaded in the main process
collate_fn = lambda batch: padded_collate(batch, {'context_inputs', 'context_inputs_mask'},
                                          default_pad_value=tokenizer.pad_token_id,
                                          special_pad_value={'context_inputs_mask': 0},
                                          pad_side='right',
                                          device=DEVICE)


train_dataset = ConcatDataset(generate_datasets(num_samples, seq_len, train_generators).values())
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

val_dataset = ConcatDataset(generate_datasets(num_samples, seq_len, val_generators).values())
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

# # Visualization
# data.visualize_datasets(generate_datasets(num_samples, seq_len, train_generators), grid_width=3, grid_height=2, num_samples=10)

# combined_dataset = ConcatDataset(datasets.values())

# train_ratio = 0.8
# val_ratio = 1 - train_ratio

# # Calculate the number of samples for each split
# num_samples = len(combined_dataset)
# num_train = int(num_samples * train_ratio)
# num_val = num_samples - num_train

# # Create the splits
# train_dataset, val_dataset = random_split(combined_dataset, [num_train, num_val])

# Create DataLoaders for each split


##################################################
#

def run_epoch(model,
              hq_model,
              cq_model,
              hk_model,
              ck_model,
              dl, optimizer, mode, device, clip=None,
              check_accuracy=False, loss_fn='cross_entropy', regularization_fn=None):
    ''' Run an epoch over a DataLoader, and optionally perform greedy sampling
    to check accuracy.

    Args:
      if loss_fn == 'cosine_distance', model must return probabilities (after sigmoid activation)
      if loss_fn == 'cross_entropy', model must return unnormalized logits (ie final output comes from Linear layer)
      if loss_fn == 'nllloss', model must return logprobs, ie `F.log_softmax(..., dim=-1)`
    '''
    assert mode in ['eval', 'train'], "mode must be either 'eval' or 'train'"
    assert loss_fn in ['cosine_distance', 'nllloss', 'cross_entropy'], "loss_fn must be 'cosine_distance', 'nllloss', or 'cross_entropy'"

    if mode == 'train':
        model.train()
    else:
        model.eval()

    epoch_loss = 0

    if check_accuracy:
        outputs = []
        total_correct = 0
        total_samples = 0

    with torch.set_grad_enabled(mode == 'train'):
        for i, batch in enumerate(dl):
            input_ids = batch['inputs']  # [B, S]
            attention_mask = batch['inputs_mask']  # [B, S]
            target_ids = batch['outputs']  # [B, S]
            context_input_ids = batch['context_inputs']  # [NUM_FEWSHOT, B, S]
            context_mask = batch['context_inputs_mask']  # [NUM_FEWSHOT, B, S]
            assert context_input_ids.shape == context_mask.shape
            B, S = input_ids.shape
            D = model.model.embed_tokens.weight.shape[1]
            n_few_shot = context_input_ids.shape[0]

            ##########
            # Starting state for LSTM
            hq = None
            cq = None
            hk = None
            ck = None

            for i in range(n_few_shot):
                hq = torch.zeros((B, D), device=DEVICE)
                # hq_ = hq_model(context_input_ids[i], attention_mask=context_mask[i]).last_hidden_state[:, -1, :]  # [B, S, D] -> [B, D]
                # hq = hq_ if hq is None else hq + hq_
                cq_ = cq_model(context_input_ids[i], attention_mask=context_mask[i]).last_hidden_state[:, -1, :]  # [B, S, D] -> [B, D]
                cq = cq_ if cq is None else cq + cq_

                hk = torch.zeros((B, D), device=DEVICE)
                # hk_ = hk_model(context_input_ids[i], attention_mask=context_mask[i]).last_hidden_state[:, -1, :]  # [B, S, D] -> [B, S]
                # hk = hk_ if hk is None else hk + hk_
                ck_ = ck_model(context_input_ids[i], attention_mask=context_mask[i]).last_hidden_state[:, -1, :]  # [B, S, D] -> [B, S]
                ck = ck_ if ck is None else ck + ck_

            cq /= n_few_shot
            ck /= n_few_shot

            # initial hidden states sent to LSTM. Each key contains a list, one
            # item per layer. Here we clone the encoded context across all
            # layers
            n_layers = len(model.model.layers)
            init_backpack = [{
                'hq': hq,
                'cq': cq,
                'hk': hk,
                'ck': ck,
            }] * n_layers


            ##########
            # Main Model
            outs = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         init_backpack=init_backpack)  # [B, S, DIM]

            if loss_fn == 'cosine_distance':

                # trg_tags = model.embeddings(trg_tags_ids)
                # trg_col1 = model.embeddings(trg_col1_ids)
                # trg_col2 = model.embeddings(trg_col2_ids)

                # loss = 0
                # for t, o in zip([trg_tags, trg_col1, trg_col2],
                #                 [out_tags, out_col1, out_col2]):
                #     # t: [B, S, D]
                #     # o: [B, S, D]
                #     assert o.shape[0] == t.shape[0]
                #     assert o.shape[1] == t.shape[1]
                #     assert o.shape[2] == t.shape[2]
                #     loss += (1 - torch.cosine_similarity(o, t, dim=2).mean())
                raise Exception('cosine_distance not implemented')

            elif loss_fn == 'cross_entropy':
                loss = 0

                # target_ids: [B, S]
                # outs      : [B, S, Vocab]
                assert outs.logits.shape[0] == target_ids.shape[0]
                assert outs.logits.shape[1] == target_ids.shape[1]
                assert outs.logits.shape[2] == model.model.embed_tokens.weight.shape[0]
                loss += F.cross_entropy(outs.logits.flatten(0, 1),
                                        target_ids.flatten(),
                                        reduction='mean')


            elif loss_fn == 'nllloss':
                # trg_ids = torch.cat([trg_tags_ids, trg_col1_ids, trg_col2_ids], dim=1)
                # # log_probs = F.log_softmax(output, dim=-1)
                # loss = F.nll_loss(output.flatten(0, 1),
                #                   trg_ids.flatten(),
                #                   reduction='mean')
                pass

            # if regularization_fn is not None:
            #     loss += regularization_fn(model, output)

            epoch_loss += loss.item()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            if check_accuracy:
                with torch.no_grad():
                    if loss_fn == 'cosine_distance':
                        # # find the embedding closest to the output, consider
                        # # that the output_id
                        # out_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                        #                                   outs.logits.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                        #                                   dim=3).argmax(dim=2)
                        pass
                    else:
                        out_ids = outs.logits.argmax(dim=2)

                    correct_outs = (out_ids == target_ids).sum().item()

                    total_correct += correct_outs
                    total_samples += out_ids.numel()
                    outputs.append((out_ids))

    if check_accuracy:
        accuracy = total_correct / total_samples
        # scores: (precision, recall, f1_score, support)
        # f1_tags = precision_recall_fscore_support(y_true_tags, y_pred_tags, average='weighted', zero_division=torch.nan)
        # f1_col1 = precision_recall_fscore_support(y_true_col1, y_pred_col1, average='weighted', zero_division=torch.nan)
        # f1_col2 = precision_recall_fscore_support(y_true_col2, y_pred_col2, average='weighted', zero_division=torch.nan)

        return epoch_loss / len(dl), {
            # 'f1_tags': f1_tags,
            # 'f1_col1': f1_col1,
            # 'f1_col2': f1_col2,
            'accuracy': accuracy,
        }, outputs

    else:
        return epoch_loss / len(dl)


##################################################
# Model

hidden_size = 64
intermediate_size = 64  # transformer mlp block
num_hidden_layers = 3
num_attention_heads = 4
num_key_value_heads = 4
rope_theta = 10000
attention_dropout = 0.2
head_dim = hidden_size // num_attention_heads
q_dim = head_dim * num_attention_heads
k_dim = head_dim * num_key_value_heads

config = Q.Qwen2Config(
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
    rope_theta=rope_theta,
    attention_dropout=attention_dropout,
    num_hidden_layers=num_hidden_layers,
    _attn_implementation='eager',
    vocab_size=tokenizer.get_vocab_size(),
    # vocab_size (`int`, *optional *, defaults to 151936):
    # hidden_size (`int`, *optional *, defaults to 4096):
    # intermediate_size (`int`, *optional *, defaults to 22016):
    # num_hidden_layers (`int`, *optional *, defaults to 32):
    # num_attention_heads (`int`, *optional *, defaults to 32):
    # num_key_value_heads (`int`, *optional *, defaults to 32):
    # hidden_act (`str` or `function`, *optional *, defaults to `"silu"`):
    # max_position_embeddings (`int`, *optional *, defaults to 32768):
    # initializer_range (`float`, *optional *, defaults to 0.02):
    # rms_norm_eps (`float`, *optional *, defaults to 1e-06):
    # use_cache (`bool`, *optional *, defaults to `True`):
    # tie_word_embeddings (`bool`, *optional *, defaults to `False`):
    # rope_theta (`float`, *optional *, defaults to 10000.0):
    # use_sliding_window (`bool`, *optional *, defaults to `False`):
    # sliding_window (`int`, *optional *, defaults to 4096):
    # max_window_layers (`int`, *optional *, defaults to 28):
    # attention_dropout (`float`, *optional *, defaults to 0.0):
)

lstm_config = {
    'input_dim': hidden_size,
    'q_dim': q_dim,
    'k_dim': k_dim,
    'dropout_p': 0.0
}

# lstm_config = None  # dont use lstm on QK

# model_hq = Attention.Qwen2Model(config, causal_decoders=False, lstm_config=lstm_config).to(DEVICE)
model_hq = None
model_cq = Attention.Qwen2Model(config, causal_decoders=False, lstm_config=lstm_config).to(DEVICE)
# model_hk = Attention.Qwen2Model(config, causal_decoders=False, lstm_config=lstm_config).to(DEVICE)
model_hk = None
model_ck = Attention.Qwen2Model(config, causal_decoders=False, lstm_config=lstm_config).to(DEVICE)

model = Attention.Qwen2ForCausalLM(config, causal_decoders=False, lstm_config=lstm_config)
model = model.to(DEVICE)


print('skipping training embeddings')
no_trains = [model.model.embed_tokens,
             # model_hq.embed_tokens,
             model_cq.embed_tokens,
             # model_hk.embed_tokens,
             model_ck.embed_tokens,
             ]
for no_train in no_trains:
    for p in no_train.parameters():
        p.requires_grad = False

params = (
    [x for x in model.parameters() if x.requires_grad] +
    # [x for x in model_hq.parameters() if x.requires_grad] +
    [x for x in model_cq.parameters() if x.requires_grad] +
    # [x for x in model_hk.parameters() if x.requires_grad] +
    [x for x in model_ck.parameters() if x.requires_grad]
)
print_model_info(model)
optimizer = optim.Adam(params, lr=LR, weight_decay=WD)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


for epoch in tqdm(range(NUM_EPOCHS)):
    # Training
    train_loss, t_log, _ = run_epoch(
        model,
        model_hq,
        model_cq,
        model_hk,
        model_ck,
        train_dl, optimizer, 'train', DEVICE, GRAD_CLIP,
        check_accuracy=True, loss_fn=LOSS_FN)
    tacc = t_log['accuracy']

    train_losses.append(train_loss)
    train_accuracies.append(tacc)

    # Validation
    val_loss, v_log, _ = run_epoch(
        model,
        model_hq,
        model_cq,
        model_hk,
        model_ck,
        val_dl, None, 'eval', DEVICE,
        check_accuracy=True, loss_fn=LOSS_FN)
    vacc = v_log['accuracy']

    val_losses.append(val_loss)
    val_accuracies.append(vacc)

    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.9f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')


    # # Log parameter histograms
    # try:
    #     for name, param in model.named_parameters():
    #         writer.add_histogram(name, param, epoch)
    # except:
    #     print('couldnt log to tensorboard')



##########
# Visualization

# Set the style to dark background
plt.style.use('dark_background')

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Define cyberpunk colors
cyberpunk_blue = '#00ffff'
cyberpunk_pink = '#ff00ff'

# Plot losses
epochs = range(1, len(train_losses) + 1)
ax1.plot(epochs, train_losses, color=cyberpunk_blue, label='Training Loss', linewidth=2)
ax1.plot(epochs, val_losses, color=cyberpunk_pink, label='Validation Loss', linewidth=2)
ax1.set_title('Loss', color='white')
ax1.set_xlabel('Epochs', color='white')
ax1.set_ylabel('Loss', color='white')
ax1.legend()
ax1.grid(True, color='white', alpha=0.3)

# Plot accuracies
ax2.plot(epochs, train_accuracies, color=cyberpunk_blue, label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_accuracies, color=cyberpunk_pink, label='Validation Accuracy', linewidth=2)
ax2.set_title('Accuracy', color='white')
ax2.set_xlabel('Epochs', color='white')
ax2.set_ylabel('Accuracy', color='white')
ax2.legend()
ax2.grid(True, color='white', alpha=0.3)

# Customize the appearance
for ax in [ax1, ax2]:
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
