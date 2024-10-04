'''.

Apply Metalearning-Hypernet stuff to Transformer arch

This version gives control over LoR calculation to a custom module. The
previous version used a Model that was updated to add in the LoR calculation
directly. This version allows the custom module to cache LoR values itself, and
apply LayerNorm after it has been added in with the original computation.


PROVENANCE:
- t14_homoiconic_llm_04.py

NaN CAUSES:
- cross_entropy loss where everything is masked out
- ragged batch sizes
- bfloat16?
- clip grads?
- layer norm?
- lor module init?
- if loss_mask is all False = NaN (ie cross_entropy gets empty tensors)


TODO: STABILIZE TRAINING
- [ ] curriculum learning (simpler data (fewer variable indirections), for starters)
- [ ] teacher forcing
- [ ] test time training
- [ ] init lormodules to output top eigenvectors of matched linear layers?
- [X] increasing dataset size helped, there's seems to be a minimal count necessary
- [X] adding problem text to loss seems to stabilize training, and it appears
      to still be able to overfit training. I suspect an issue with this though,
      that problem syntax might be predictable, but not specifics, so this might
      hurt the quality

NOTES:
- LayerNorm on end of LORModule encouraged weights to move more, without, they imperceptibly moved (also used high lr (1e-2))
- Biases in LORModule can cause non-lor layers, which receive 0 values, to suddenly have values, and corrupt performance. LORModules should have the property that they cannot alter non-LOR'd samples.
- swiglu lor modules seem more stable to train than linear-only, and better val loss
- sensitive to lr. 1e-5 seems ok, 1e-4 even might be too high. Even with lor-module params only training, 1e-3 diverges.
- sensitive to initialization


TODO FIX:



- [X] need to iterate over list of per-layer ixs
- [X] later, how to connect lor_cache with lor_model? in the model module, its now just passing in the module instead of the cache
- [X] replace `lor` dep in model
- [X] add changes to `generate`
- [X] general clean up
- [ ] add correctness tests for lor stuff
  - [X] batch mode gives same output as non-batched
  - [X] parse ixs don't cause weirdness in non-parsed samples)
  - [ ] ensure cache looks good (with test)
- [ ] `generate` should use `forward_columns`
- [ ] train RMSNorm separately, first?
- [ ] trained model gets low loss, then outputs "^@Q" as next token?!
- [ ] get test loss down
- [ ] double check that lormodule init produces LRx=0 before any training, make it a test



- [ ] loss_fn is currently unbatched
- [X] smaller dataset (fewer vars)
- [X] only train gud for now
- [X] overfit a single sample. If this doesn't work, test time training won't, nor teacher forcing.
- [ ] LORModule weights not updating much
- [ ] causal masking?
- [ ] causal mask LOR blocks?
- [ ] OFF BY ONE ERROR IN LOSS?!
- [ ] new tokens (meta tokens) not updating, should they?
- [ ] when training loss hits 0, @ inference it gets training probs wrong
- [ ] lor_models weights not updating, but loss decreases (and they're the only params!?)
  - lor modules in ParameterDict?
  - forward_lor mutates dict?
- [ ] dbl check loss_mask, and off-by-one, and include_in_loss
- [ ] dbl check off-by-one shifts in data/parsing lors/loss
- [X] right singular vectors not updating
- [X] make use of lor_qkvo in model
- [X] replace F.interpolate with a Linear Forward in LORModule
- [X] i flipped lor_i and lor_o from lor_gud, fix
- [X] make sure to properly initialize params in LORModule


TODO:
- [ ] with torch.autograd.detect_anomaly():
- [ ] teacher forcing by taking preceding clause, ex x=1, and setting QKVOGUD to relevant portions of that
- [ ] teacher forcing (via training lor weights one step, then using those)
- [X] aggregate LORs instead of replace
- [X] if not training text tokens, consider tracing what the outputs actually are
- [ ] implement LOR parsing during inference's autoregressive portion
- [ ] lor_ixs makes lor_mask irrelevant? maybe not...


OPTIM:
- [ ] batch size QUICKLY eats up VRAM
- [ ] is the way I'm unsqueezing/stacking dimensions of left/right lors ideal?
- [ ] LOR projections (LORModule) could be cached (should be cached?)
- [ ] multi-gpu


IDEAS:

- [ ] after a LOR block, forbid attention to preceding tokens (or just make
      RNN). This could provide an info bottleneck, and force lors to be highly
      relevant.
- [ ] instead of interpretting hidden state as metaweights, interpret traces of Q/K/V/etc projections?
- [ ] query back out matrix->vector. We have vec->mat, but not the reverse.

'''

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # let cuda give debug info

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from datasets import load_dataset

from typing import Optional, Tuple, Union, List, Dict, Any
import warnings
import math

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
import warnings
import itertools

from torch.utils.tensorboard import SummaryWriter

from neurallambda.lab.common import print_model_info
from neurallambda.torch import get_last_attended_token

import t14_homoiconic_llm_add_tokens as AT

from datasets import Dataset as HFDataset

from dataclasses import dataclass
import importlib
from functools import partial

from datetime import datetime
current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = SummaryWriter(f'log/t14_homoiconic_llm_05/{current_time}')
LOG = True


try:
    importlib.reload(C)
    print('RELOADING C')
except NameError:
    import t14_homoiconic_llm_columnize_05 as C

try:
    importlib.reload(Q)
    print('RELOADING Q')
except NameError:
    import t14_homoiconic_llm_model_02 as Q


SEED = 152
torch.manual_seed(152)
random.seed(SEED)

DEVICE = 'cuda:1'


##################################################

# Ex:
#
# "Hi there^Q^1||^K^1|| how are you?"
#          ^     ^
#          |     |
#          |     At K weights of layer 1, 2 pipes define the L and R singular vectors
#          At Q weights of layer 1, 2 pipes define the L and R singular values
#

# Pairs of new tokens with their initialization
new_token_pairs = [
    # Attention
    ("^@Q", "&nbsp"),  # Q proj
    ("^@K", "&nbsp"),  # K proj
    ("^@V", "&nbsp"),  # V proj
    ("^@O", "&nbsp"),  # O proj

    # MLP
    ("^@G", "&nbsp"),  # Gate proj
    ("^@U", "&nbsp"),  # Up proj
    ("^@D", "&nbsp"),  # Down proj

    # Layer identifier
    #   Qwen1.5B has 28 layers, for testing we'll target
    ("^@3", "&nbsp"),
    ("^@24", "&nbsp"),

    # Dummy weight
    ("^@|", "&nbsp"),
]


##################################################
# Lor stuff

def create_position_ids(attention_mask):
    """
    Create position IDs based on the attention mask.
    Set position to -1 for padding tokens.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, -1)
    return position_ids


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
    for k in lor_models.keys():
        fs[k] = []
        for m, c in zip(lor_models[k], lor_cache[k]):
            if m is not None:
                f = partial(m, c)
            else:
                f = None
            fs[k].append(f)
    return fs


def forward_columns(model, lor_models, lor_cache, input_idss, attention_masks, position_idss, lor_ixss):
    '''Recurrently process column blocks of ids, concatenating attention_mask and
past_key_values across generations. '''
    col_out_logits = []
    past_key_values = None
    attention_mask = None

    # Iterate over each column in the batch
    for i, (input_ids, new_attention_mask, position_ids, lor_ixs) in enumerate(zip(input_idss, attention_masks, position_idss, lor_ixss)):

        # skip empty batches. this can happen because of padded blocks.
        if input_ids.numel() == 0:
            continue

        # `attention_mask` will continue to grow as the entire sequence length
        # seen so far (even though input_ids will only be new inputs, not past)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
        else:
            attention_mask = new_attention_mask

        # give the lor models the current cache
        lorm = partially_apply_models(lor_models, lor_cache)

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,  # OPTIM: we don't need this for non-lor blocks
                    **lorm,
                    )
        col_out_logits.append(out.logits)

        past_key_values = out.past_key_values
        assert past_key_values is not None, "past_key_values cannot be None. Typically `transformers` models won't return past_key_values during training, so, you need to modify the underlying model class."

        lor_cache = update_lors(lor_models, lor_cache, lor_ixs, out.hidden_states)

    return col_out_logits, attention_mask, past_key_values, lor_cache


##################################################
# Training Fns


def unfreeze_embeddings(grads, ixs):
    """
    Zero out gradients for all embeddings except those specified by ixs.

    Args:
    grads (torch.Tensor): Gradients of the embedding layer
    ixs (list or torch.Tensor): Indices of the embeddings to keep unfrozen

    Returns:
    torch.Tensor: Modified gradients with appropriate values zeroed out
    """
    # Create a mask of zeros with the same shape as grads
    mask = torch.zeros_like(grads)

    # Set the mask to 1 for the indices we want to keep unfrozen
    mask[ixs] = 1

    # Multiply the original gradients by the mask
    # This zeroes out gradients for all embeddings except those in ixs
    out = grads * mask
    return out


def loss_fn(
    col_batch_in: List[torch.Tensor],  # list of blocks of TOKENS IDS (dtype=int)
    col_batch_out: List[torch.Tensor],  # list of blocks of EMBEDDINGS (dtype=float)
    loss_mask: List[torch.Tensor]  # list of blocks of masks (dtype=bool)
):
    vocab_size = col_batch_out[0].shape[2]

    # Concatenate, shift, flatten
    labels = torch.cat(col_batch_in, dim=1)[..., 1:].contiguous().view(-1)  # [B, S-1] -> [B * (S-1)]
    logits = torch.cat(col_batch_out, dim=1)[..., :-1, :].contiguous().view(-1, vocab_size)  # [B, S-1, D] -> [B * (S-1), D]
    # m = torch.cat(loss_mask, dim=1)[..., :-1].contiguous().view(-1)  # TODO: is this right?
    m = torch.cat(loss_mask, dim=1)[..., 1:].contiguous().view(-1)

    # Calculate masked loss
    loss = F.cross_entropy(logits[m], labels[m], reduction='mean')
    breakpoint()
    return loss



def loss_fn(
    col_batch_in: List[torch.Tensor],  # list of blocks of TOKENS IDS (dtype=int)
    col_batch_out: List[torch.Tensor],  # list of blocks of EMBEDDINGS (dtype=float)
    loss_mask: List[torch.Tensor]  # list of blocks of masks (dtype=bool)
):
    ''' TODO DEBUG for loop to make sure view isn't messing things up '''

    # Concatenate, shift, flatten
    labels = torch.cat(col_batch_in, dim=1)[..., 1:].contiguous()  # [B, S-1]
    logits = torch.cat(col_batch_out, dim=1)[..., :-1, :].contiguous()  # [B, S-1, D]
    m = torch.cat(loss_mask, dim=1)[..., 1:].contiguous()

    B = labels.shape[0]

    loss = 0
    for b in range(B):
        loss = loss + F.cross_entropy(logits[b][m[b]], labels[b][m[b]], reduction='mean')

    loss = loss / B
    return loss



def run_epoch(model, lor_models, dataloader, optimizer, device, train=True, debug=False):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch_cols in tqdm(dataloader, desc="Training" if train else "Evaluating"):
            B = batch_cols[0]['input_ids'].shape[0]
            device = model.device

            # list of columns
            input_idss = [x['input_ids'].to(device) for x in batch_cols]
            attention_masks = [x['attention_mask'].to(device) for x in batch_cols]
            position_idss = [x['position_ids'].to(device) for x in batch_cols]
            loss_masks = [x['loss_mask'].to(device) for x in batch_cols]

            # lor_ixs per column
            lor_ixss = []
            for col in batch_cols:
                lor_ixs = {k: (v[0].to(device), v[1].to(device)) for k, v in col.items() if k in lor_ix_keys}
                # v[0]: [layer, batch_size]; parse ixs, 1 (optional) index per layer for each sample
                # v[1]: [layer, batch_size]
                lor_ixss.append(lor_ixs)

            lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch
            out_logits, cat_attention_mask, _, lor_cache = forward_columns(model, lor_models, lor_cache, input_idss, attention_masks, position_idss, lor_ixss)

            loss = loss_fn(input_idss,
                           out_logits,
                           loss_masks)

            if torch.isnan(loss):
                print('NaN encountered:')
                print(f'  all loss was masked out?: {sum([x.to(dtype=torch.long).sum() for x in loss_masks]) == 0}')
                print(f'  nan in out_logits?: {torch.cat(out_logits, dim=1).isnan().sum() > 0}')
                print('  ragged batch sizes? (double check by hand if necessary)')
                breakpoint()

            if train:
                optimizer.zero_grad()
                loss.backward()

                # mask out all tokens except the new meta tokens
                if model.model.embed_tokens.weight.grad is not None:
                    warnings.warn('freezing all token embeddings except new meta tokens')
                    with torch.no_grad():
                        model.model.embed_tokens.weight.grad[:] = unfreeze_embeddings(model.model.embed_tokens.weight.grad, new_token_ids)

                # MAX_GRAD_NORM = 1.0
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)  # NOTE: nan fix?
                # nn.utils.clip_grad_norm_(lor_models.parameters(), max_norm=MAX_GRAD_NORM)  # NOTE: nan fix?

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
        return avg_loss, out_logits, cat_attention_mask, lor_cache
    else:
        return avg_loss

##################################################

model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-7B")


def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"NaN or inf detected in {module.__class__.__name__}")
            print(f"Module: {module}")
            print(f"Module name: {module_names.get(module, 'Unknown')}")
            print(f"Input shape: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}")
            print(f"Output shape: {output.shape}")
            raise RuntimeError("NaN or inf detected")

module_names = {}

def add_hooks(model):
    for name, module in model.named_modules():
        module_names[module] = name
        module.register_forward_hook(hook_fn)

try:
    fail
    already_loaded
except:
    print('Loading model')
    model = Q.Qwen2ForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    num_layers = model.config.num_hidden_layers

    # add new tokens
    if False:
        AT.assert_unique_tokens(tokenizer, [x[0] for x in new_token_pairs])

    AT.add_and_initialize_tokens(model, tokenizer, new_token_pairs)

    new_token_ids = [tokenizer.convert_tokens_to_ids(x[0]) for x in new_token_pairs]

    if False:
        AT.test_token_behavior(model, tokenizer, new_token_pairs)


    # fuzz new tokens
    with torch.no_grad():
        for i, t in enumerate(new_token_ids):

            # random fuzz
            # model.model.embed_tokens.weight[t, :] += torch.randn_like(model.model.embed_tokens.weight[t]) * 1e-2

            # # fuzz by rolling; preserves statistics, but eliminates semantixs
            # model.model.embed_tokens.weight[t, :] += torch.roll(model.model.embed_tokens.weight[t], shifts=i)

            # ignore inits, make random
            model.model.embed_tokens.weight[t, :] += torch.randn_like(model.model.embed_tokens.weight[t]) * 1e-3


    # ensure parsability
    if False:
        print('Testing that new tokens are parsable')
        toks = [x[0] for x in new_token_pairs]
        AT.assert_parsability_1(tokenizer, toks)
        AT.assert_parsability_2(tokenizer, toks)
        AT.assert_parsability_3(tokenizer, toks)

    if False:
        print_model_info(model)

    add_hooks(model)

    already_loaded = True


##########
# LOR stuff

# Layer index to target, hard coded
LOR_LAYER = 14  # must be positive num (can't index bwd)

WHICH_LOR = 2

##########
# Do QKVOGUD Lors

if WHICH_LOR == 1:

    # These are metatokens that get interjected throughout training data
    #
    # Q|| K|| V|| O|| G|| U|| D||
    lor_block = "^@Q^@|^@|^@K^@|^@|^@V^@|^@|^@O^@|^@|^@G^@|^@|^@U^@|^@|^@D^@|^@|"

    # metatoken mask, for masking loss. metatokens should be predicted, thus make
    # it to loss. meta weights should not be affected directly by the cross entropy
    # loss.

    # TODO: i think this needs to be shifted one left, into the preceding block (??)

    # lor_mask = [1, 0, 0] * 7  # note: should be list, not tensor. `datasets` converts it back to a list (then tensor again) anyway
    lor_mask = [0, 0, 0] * 7  # TODO: rm, this mask does NOT learn output of metatokens

    # for blocks that don't contain any lors
    #   note: right now this assumes only a single layer is getting targeted, but someday could have separate values per index
    empty_lor_ixs = {
        'lor_qs': (None, None),
        'lor_ks': (None, None),
        'lor_vs': (None, None),
        'lor_os': (None, None),
        'lor_gs': (None, None),
        'lor_us': (None, None),
        'lor_ds': (None, None),
    }

    # where to parse to collect metaweights, tied to `lor_block`'s implementation.
    #
    # note: this might look shifted left, but consider "Q||". The token emitted
    #       after Q will be in location 0 and represent the first predicted |.
    lor_ixs = {
        'lor_qs': (0, 1),  # (left singular value, right singular value)
        'lor_ks': (3, 4),
        'lor_vs': (6, 7),
        'lor_os': (9, 10),
        'lor_gs': (12, 13),
        'lor_us': (15, 16),
        'lor_ds': (18, 19),
    }

    lor_ix_keys = set(lor_ixs.keys())


##########
# Just do GUD Lors

elif WHICH_LOR == 2:

    # These are metatokens that get interjected throughout training data
    #
    # G|| U|| D||
    lor_block = "^@G^@|^@|^@U^@|^@|^@D^@|^@|"

    # metatoken mask, for masking loss. metatokens should be predicted, thus make
    # it to loss. meta weights should not be affected directly by the cross entropy
    # loss.

    # TODO: i think this needs to be shifted one left, into the preceding block (??)

    # lor_mask = [1, 0, 0] * 3  # note: should be list, not tensor. `datasets` converts it back to a list (then tensor again) anyway
    lor_mask = [0, 0, 0] * 3  # TODO: rm, this mask does NOT learn output of metatokens

    # for blocks that don't contain any lors
    #   note: right now this assumes only a single layer is getting targeted, but someday could have separate values per index
    empty_lor_ixs = {
        'lor_gs': [None] * num_layers,
        'lor_us': [None] * num_layers,
        'lor_ds': [None] * num_layers,
    }

    # where to parse to collect metaweights, tied to `lor_block`'s implementation.
    #
    # note: this might look shifted left, but consider "Q||". The token emitted
    #       after Q will be in location 0 and represent the first predicted |.

    parse_lor_ixs = {
        'lor_gs': [None] * num_layers,
        'lor_us': [None] * num_layers,
        'lor_ds': [None] * num_layers,
    }
    parse_lor_ixs['lor_gs'][LOR_LAYER] = (0, 1)  # (left singular value, right singular value)
    parse_lor_ixs['lor_us'][LOR_LAYER] = (3, 4)
    parse_lor_ixs['lor_ds'][LOR_LAYER] = (6, 7)

    lor_ix_keys = set(parse_lor_ixs.keys())


####################
# Data

BATCH_SIZE = 64
split = '_4'  # 4 variable version of dataset

warnings.warn('training data is severely truncated during r&d')
train_small = load_dataset("neurallambda/arithmetic_dataset", split=f"train{split}").select(range(BATCH_SIZE * 10))  # todo: rm, this truncates training
test_small = load_dataset("neurallambda/arithmetic_dataset", split=f"test{split}").select(range(BATCH_SIZE * 1))

def process_row(row, lor_block, lor_mask):
    prepared_data = []
    for item in row['input']:

        # # warnings.warn('including whole sample in loss mask')
        # prepared_data.append({"type": "text", "content": item, 'include_in_loss': True, **empty_lor_ixs})  # TODO: including text in loss

        prepared_data.append({"type": "text", "content": item, 'include_in_loss': False, **empty_lor_ixs})  # TODO: including text in loss

        prepared_data.append({"type": "lor", "content": lor_block, 'include_in_loss': True, 'loss_mask': lor_mask, **parse_lor_ixs})

    # Add the output
    prepared_data.append({"type": "text", "content": row['output'], 'include_in_loss': True, **empty_lor_ixs})

    return {"prepared_data": prepared_data}


def insert_lor_blocks(dataset: HFDataset, lor_block, lor_mask) -> HFDataset:
    """
    Insert LOR blocks between text elements in the dataset.
    """
    # NOTE: the `dataset.map` process injects missing keys found among any rows, ie loss_mask
    out = dataset.map(
        lambda row: process_row(row, lor_block, lor_mask),
        remove_columns=["input", "output"])
    return out


# Apply the preparation to a specific split
train_small = insert_lor_blocks(train_small, lor_block, lor_mask)

'''

# Sample
for example in train_small.select(range(2)):  # Show first 5 examples
    print(example)
    print()

'''

train_dl = C.create_dataloader(
    dataset=train_small,
    num_layers=num_layers,
    lor_ix_keys=lor_ix_keys,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    device='cpu',
    shuffle=False,
    num_workers=0
)


# Apply the preparation to a specific split
test_small = insert_lor_blocks(test_small, lor_block, lor_mask)
# # Sample
# for example in test_small.select(range(5)):  # Show first 5 examples
#     print(example)
#     print()
test_dl = C.create_dataloader(
    dataset=test_small,
    num_layers=num_layers,
    lor_ix_keys=lor_ix_keys,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    device='cpu',
    shuffle=False,
    num_workers=0
)


##################################################
# LoR Models

@dataclass
class LoR:
    ''' this isn't used yet, cache/parse ixs/lor-modules are still a dict of this shape '''
    lor_qs: List[Any]  # For Q blocks, a per-layer list of things (parse indexes, LoRModule, or lor cache)
    lor_ks: List[Any]
    lor_vs: List[Any]
    lor_os: List[Any]
    lor_gs: List[Any]
    lor_us: List[Any]
    lor_ds: List[Any]


def assert_lor_shape(x):
    pass


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


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim, bias=False)
        self.w2 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return (
            F.silu(self.w1(x)) *  # silu = x * sigmoid(x)
            self.w2(x)
        )


class LORProjection(nn.Module):
    '''Project the LOR weights before using them.

    NOTE: this module should probably never have bias parameters. If it has
    bias parameters it can corrupt 0-vector lors, ie non-lor'd samples.'''
    def __init__(self, in_dim, out_dim, is_left_singular_value):
        super().__init__()
        self.is_left_singular_value = is_left_singular_value
        self.f = nn.Sequential(
            # nn.LayerNorm(in_dim, bias=False),
            SwiGLU(in_dim, in_dim),
            nn.LayerNorm(in_dim, bias=False),
            nn.Linear(in_dim, out_dim, bias=False),
            # nn.LayerNorm(out_dim, bias=False),
        )
        self.initialize_parameters()

    def forward(self, x):
        return self.f(x)

    def initialize_parameters(self):
        # linear layer
        with torch.no_grad():
            swiglu_ix = 0
            linear_ix = 1

            # linear
            self.f[linear_ix].weight[:] = torch.randn_like(self.f[linear_ix].weight) * 1e-3
            # self.f[linear_ix].weight[:] = torch.zeros_like(self.f[linear_ix].weight)


            # swiglu
            self.f[swiglu_ix].w1.weight[:] = torch.randn_like(self.f[swiglu_ix].w1.weight) * 1e-3
            self.f[swiglu_ix].w2.weight[:] = torch.randn_like(self.f[swiglu_ix].w2.weight) * 1e-3


            if self.is_left_singular_value:
                # # swiglu
                # self.f[swiglu_ix].w1.weight[:] = torch.randn_like(self.f[swiglu_ix].w1.weight) * 1e-3
                # self.f[swiglu_ix].w2.weight[:] = torch.randn_like(self.f[swiglu_ix].w2.weight) * 1e-3

                # linear
                # self.f[linear_ix].weight[:] = torch.randn_like(self.f[linear_ix].weight) * 1e-4
                # self.f[linear_ix].weight[:] = torch.zeros_like(self.f[linear_ix].weight)
                pass


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


class LORModule(nn.Module):
    '''.

    WARNING: batch sensitivity. If a given sample's column of data indicates a
      parse should be performed, EVERY sample in that column will have a
      "parse" performed, but it will return just 0s for the LoR values. This
      means that, while a sample did not intend to have any LoR parses
      performed, it will have them done anyway, and the parsed LoR values will
      continue to get applied in perpetuity. Thus, the LORModule MUST have the
      propertly that 0-valued LoR vectors do not affect the output, ie, are a
      null op. This property would be violated if for instance LORModule added
      biases. Certain nonlinearities, or perhaps other things could also break
      this property, TAKE CARE.

    '''

    def __init__(self,
                 left_in_dim, left_out_dim,
                 right_in_dim, right_out_dim):
        super().__init__()

        self.left_in_dim = left_in_dim
        self.left_out_dim = left_out_dim
        self.right_in_dim = right_in_dim
        self.right_out_dim = right_out_dim

        # TODO: these could be heads over a common hypernet backbone
        self.left_proj = LORProjection(left_in_dim, left_out_dim, is_left_singular_value=True)  # left singular value
        self.right_proj = LORProjection(right_in_dim, right_out_dim, is_left_singular_value=False)  # right singular value


        # TODO: test WITHOUT LoR that this is compatible with default model
        self.norm = nn.RMSNorm(left_out_dim)

        # TODO: initialize correctly
        # with torch.no_grad():
        #     self.norm.weight[:] = torch.zeros_like(self.norm.weight) + ?

        # LORModule must play nicely in a batch situation, where some samples
        # of the batch imply lor parses and others don't. Non LoR'd samples
        # should not be affected by sharing a batch with LoR'd samples. Biases
        # corrupt this property. 0-valued lors (from samples without lor
        # parses) must produce 0-valued outputs here. Checking for biases is
        # not the whole solution, you must take care.
        assert_no_biases(self)

    def forward(self, lor_cache, original, hidden_state):
        '''
        Args:
          original: model's original values of eg QKVOGUD within this layer
          hidden_state: hidden_state at this layer, that projects through this layer's associated linear QKVOGUD block, and will be used to project through the LoR version too.
        '''

        # warnings.warn('LOR IS TURNED OFF')
        # return self.norm(original)

        if lor_cache is not None:
            lorl, lorr = lor_cache
            l = apply_lor(hidden_state, lorl, lorr)
            return self.norm(original + l)
        else:
            return self.norm(original)

    def left_project(self, emb):
        '''
        Args:
          emb: embedding parsed from the appropriate index from output hidden_state
        '''
        return self.left_proj(emb)

    def right_project(self, emb):
        '''
        Args:
          emb: embedding parsed from the appropriate index from output hidden_state
        '''
        return self.right_proj(emb)


def select_ixs(x, indices):
    """Selects values from a 3D tensor (`[batch, seq, dim]`) along dim=1 using
    provided indices.

    Perform indexed selection on a 3D tensor. If `indices` contains -1, that
    location will be filled with 0s.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, xs, ys] where:

        indices (torch.Tensor): 1D tensor of indices for selection. Its length
            should match the batch size of x, and it's max value must be
            smaller than the length of xs. Negative values in indices indicate
            that the location should be filled with default 0 values, ie no
            parsing performed.

    Returns:
        torch.Tensor: [batch, dim]

    Example:
        >>> x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...                   [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
        >>> indices = torch.tensor([1, -1])
        >>> select_ixs(x, indices, dim=1)
        tensor([[[ 4,  5,  6]],
                [[ 0,  0,  0]]])

    """

    B = x.shape[0]  # batch

    # a mask for valid indices (non-negative)
    mask = indices >= 0

    # replace negative indices (indicating default 0 values) with 0
    safe_indices = torch.where(mask, indices, torch.zeros_like(indices))

    # gather the embeddings along the sequence dimension
    gathered_values = x[torch.arange(B), safe_indices]

    # a zero tensor, for default values where index == -1
    zeros = torch.zeros_like(gathered_values)

    # combine selected indices with defaults
    return torch.where(mask.unsqueeze(1), gathered_values, zeros)


def update_lors(
        lor_models,  # shaped like `empty_lors`
        lor_cache,  # shaped like `empty_lors`
        lor_ixs: Dict[str, List[Tuple[int, int]]],  # column ixs, eg: {'lor_qs': [None, None, (left_ix, right_ix), None, ...n_layers]}
        hidden_states,  # from out.hidden_states, so, for all layers
):
    ''' Update the LORs by interpreting hidden states as new lor blocks '''

    for k in lor_ixs.keys():
        assert (
            len(lor_models[k]) ==  # one (optional) lor module per layer
            len(lor_cache[k]) ==  # cache per layer
            len(lor_ixs[k][0]) ==  # left parse ixs
            len(lor_ixs[k][1])  # right parse ixs
        ), (f'''
{len(lor_models[k])=} ==  # one (optional) lor module per layer
{len(lor_cache[k])=} ==  # cache per layer
{len(lor_ixs[k][0])=} ==  # left parse ixs
{len(lor_ixs[k][1])=}  # right parse ixs
''')

    h_emb = hidden_states[-1]  # final layer states

    for lor_key, (l_layer_batch_ixs, r_layer_batch_ixs) in lor_ixs.items():
        num_layers = l_layer_batch_ixs.shape[0]
        for layer_ix in range(num_layers):

            #####
            # Assertions and Prep

            # skip layers that aren't targeted by LoR modules
            if lor_models[lor_key][layer_ix] is None:
                continue
            lor_model = lor_models[lor_key][layer_ix]

            # no parse ixs identified
            l_empty = (l_layer_batch_ixs[layer_ix] >= 0)
            r_empty = (r_layer_batch_ixs[layer_ix] >= 0)
            assert (l_empty == r_empty).all().item(), 'left parse indexes and right parse indexes should always accompany each other'
            if l_empty.sum() == 0:
                continue  # no parses in any rows in this column

            is_first_pass = lor_cache[lor_key][layer_ix] is None

            #####
            # Parse ixs, apply LoR projections to parsed embeddings, cache

            # left singular values
            left_lors = lor_model.left_project(select_ixs(h_emb, l_layer_batch_ixs[layer_ix]))
            if is_first_pass:
                new_l = left_lors.unsqueeze(2)  # [B, DIM, RANK]
            else:
                new_l = torch.cat([lor_cache[lor_key][layer_ix][0], left_lors.unsqueeze(2)], dim=2)

            # right singular values
            right_lors = lor_model.right_project(select_ixs(h_emb, r_layer_batch_ixs[layer_ix]))
            if is_first_pass:
                new_r = right_lors.unsqueeze(2)  # [B, DIM, RANK]
            else:
                new_r = torch.cat([lor_cache[lor_key][layer_ix][1], right_lors.unsqueeze(2)], dim=2)

            lor_cache[lor_key][layer_ix] = (new_l, new_r)

    return lor_cache


##########
#

if True:
    num_epochs = 100
    lr = 1e-4

    # wd = 0

    warnings.warn('WEIGHT DECAY turned on')
    wd = 1e-2

    # LOR Models: same structure as LORs
    dim = model.model.embed_tokens.weight.shape[1]
    k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
    v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
    ff_dim = model.config.intermediate_size

    # Note: there must be at least a None per each QKVOGUD block per layer
    lor_models = nn.ModuleDict(
        {
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

    # for k in lor_models.keys():
    #     lor_models[k] = nn.ModuleList(lor_models[k])

    # LOR weights need to be projected to fit the shapes of the underlying
    # Linear layers they're matching. These LORModules solve this, and there
    # are 2 modules per target linear layer, (left singular values, right
    # singular values). They multiply like: out=LRx, to match the usual out=Wx,
    # so the new calculation becomes out=Wx + LRx.
    #
    # R are the input vectors, and L are the output vectors. The first
    # dimension of the LORModules must match the embedding that we're
    # projecting, so the 1st values are all `dim`. The 2nd dim of R is the
    # input dimension of the matched linear weights. The 2nd dim of L is the
    # output dimension of the same linear layer.

    # lor_models['lor_qs'][LOR_LAYER] = LORModule(dim, dim, dim, dim)
    # lor_models['lor_ks'][LOR_LAYER] = LORModule(dim, k_dim, dim, k_dim)
    # lor_models['lor_vs'][LOR_LAYER] = LORModule(dim, v_dim, dim, v_dim)
    # lor_models['lor_os'][LOR_LAYER] = LORModule(dim, dim, dim, dim)
    lor_models['lor_gs'][LOR_LAYER] = LORModule(dim, ff_dim, dim, dim)
    lor_models['lor_us'][LOR_LAYER] = LORModule(dim, ff_dim, dim, dim)
    lor_models['lor_ds'][LOR_LAYER] = LORModule(dim, dim, dim, ff_dim)
    lor_models = lor_models.to(DEVICE, dtype=model.dtype)

    # # All Params
    # parameters = list(itertools.chain(model.parameters(), lor_models.parameters()))

    # LOR Params Only
    parameters = list(lor_models.parameters())
    # Embeddings. Add all, and then zero out grads (in run_epoch) to target just the new tokens
    parameters = parameters + list(model.model.embed_tokens.parameters())

    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=wd)

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

    global_epoch = 0

    # START_BLOCK_2

    model.train()
    for epoch in range(num_epochs):
        global_epoch += 1
        train_loss = run_epoch(model, lor_models, train_dl, optimizer, DEVICE, train=True)
        train_losses.append(train_loss)
        writer.add_scalars('loss', {'train': train_loss}, global_epoch)

        if epoch % 5 == 0:
            test_loss = run_epoch(model, lor_models, test_dl, optimizer, DEVICE, train=False)
            # test_loss = 0
            test_losses.append(test_loss)
            writer.add_scalars('loss', {'test': test_loss}, global_epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")


    # END_BLOCK_2

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


####################################################################################################
####################################################################################################
####################################################################################################


# Inference, ie Autoregressive Continuation

# START_BLOCK_1

def generate(model, lor_models, col_inputs, max_new_tokens):
    '''Recurrently process column blocks of ids, concatenating attention_mask and
    past_key_values across generations. Can generate new tokens if max_new_tokens > 0.'''
    col_out_logits = []
    past_key_values = None
    attention_mask = None
    all_output_ids = []
    tokens_generated = 0

    lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch

    # We need to keep the max position id because short batch items end with
    # `-1` padding, but during inference we'll need the correct starting
    # position. We also need to collect the last output token which will be the
    # start of generation. Again, for short inputs, their last token would
    # otherwise be predicted off a padding token.
    max_position_ids = None  # shape=[batch]
    last_token = None

    # Iterate over each column in the batch
    for i, batch_column in enumerate(col_inputs):
        input_ids = batch_column['input_ids']
        new_attention_mask = batch_column['attention_mask']
        position_ids = batch_column['position_ids']
        device = input_ids.device

        lor_ixs = {k: (v[0].to(device), v[1].to(device)) for k, v in batch_column.items() if k in lor_ix_keys}

        # skip empty batches. this can happen because of padded blocks.
        if input_ids.numel() == 0:
            continue

        # `attention_mask` will continue to grow as the entire sequence length
        # seen so far (even though input_ids will only be new inputs, not past)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
        else:
            attention_mask = new_attention_mask

        lorm = partially_apply_models(lor_models, lor_cache)

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,  # OPTIM: we don't need this for non-lor blocks
                    **lorm,
                    )
        col_out_logits.append(out.logits)
        # all_output_ids.append(input_ids)

        past_key_values = out.past_key_values
        assert past_key_values is not None, "past_key_values cannot be None. Typically `transformers` models won't return past_key_values during training, so, you need to modify the underlying model class."

        lor_cache = update_lors(lor_models, lor_cache, lor_ixs, out.hidden_states)

        # Handle max_position_id and last_token, to be used by the
        # autoregressive loop following

        # max positions
        if max_position_ids is None:
            max_position_ids = position_ids.max(dim=1).values
        else:
            max_position_ids = torch.cat([position_ids, max_position_ids.unsqueeze(1)], dim=1).max(dim=1).values

        # last token
        out_ids = out.logits.argmax(dim=-1)
        last_token_ = get_last_attended_token(out_ids, attention_mask[:, -out_ids.shape[1]:])
        if last_token is None:
            last_token = last_token_
        else:
            last_token = torch.stack([last_token, last_token_], dim=1).max(dim=1).values  # if no token was seen yet, it'll be a -1


    # Sample final prediction as first output token
    input_ids = last_token.unsqueeze(1)
    all_output_ids.append(input_ids)
    tokens_generated += 1

    # If max_new_tokens > 0, continue generating new tokens
    position_ids = max_position_ids.unsqueeze(1)
    while tokens_generated < max_new_tokens:

        # Update position_ids
        position_ids = position_ids + 1

        # Update attention_mask
        attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)

        lorm = partially_apply_models(lor_models, lor_cache)

        # Generate the next token
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    **lorm,
                    )

        # TODO: lor updates aren't implemented during autoregression
        col_out_logits.append(out.logits)
        past_key_values = out.past_key_values

        # sample the next token
        input_ids = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        all_output_ids.append(input_ids)

        tokens_generated += 1

    return col_out_logits, attention_mask, past_key_values, torch.cat(all_output_ids, dim=-1), lor_cache, position_ids, attention_mask


def t(x):
    return {'type': 'text', 'content': x, 'include_in_loss': False, **empty_lor_ixs}

def w(x, lor_ixs):
    # one parse per type can be used per column/block, eg LOR Q left
    # singular value, LOR Q right singular value, LOR K etc...
    return {'type': 'lor', 'content': x, 'include_in_loss': False, **lor_ixs}

def p(x):
    return {'type': 'pad_block', 'content': x, 'include_in_loss': False, **empty_lor_ixs}


model.eval()
with torch.no_grad():

    lor_ = w(lor_block, parse_lor_ixs)

    if WHICH_LOR == 1:
        tlor = {'content': lor_block,
                'include_in_loss': True,
                'loss_mask': [0, 0, 0] * 7,
                'type': 'lor',
                **parse_lor_ixs}

    elif WHICH_LOR == 2:
        tlor = {'content': lor_block,
                'include_in_loss': True,
                'loss_mask': [0, 0, 0] * 3,
                'type': 'lor',
                **parse_lor_ixs}

    te = {'lor_ds': [None, None], 'lor_gs': [None, None], 'lor_ks': [None, None], 'lor_os': [None, None], 'lor_qs': [None, None], 'lor_us': [None, None], 'lor_vs': [None, None]}

    training_prob = [
        {'content': 'var_0=-2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_1=var_0', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_2=var_1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_3=var_2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_4=var_3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_5=4 - 3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_6=1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_7=var_2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_8=-4', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_9=var_0 * 8', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'solve(var_7)=', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        # {'content': '-', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **te}
        # {'content': '-2', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **te}
    ]

    training_prob_2 = [
        {'content': 'var_0=9', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_1=var_0', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_2=1 - var_1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_3=-1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_4=-3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_5=-7', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_6=7', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_7=-6 * var_4', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_8=var_1 + var_3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'var_9=var_8', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        {'content': 'solve(var_5)=', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **te}, tlor,
        # {'content': '-', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **te}
        # {'content': '-7', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **te}
    ]

    test_prompts = [
        training_prob,
        training_prob_2,
        [t("var_0=1"), lor_, t("var_1=2"), lor_, t("var_3=var_0 + var_1"), lor_, t("solve(var_3)=")],  # 3
        [t("var_0=3"), lor_, t("var_1=5"), lor_, t("var_3=var_0 - var_1"), lor_, t("solve(var_3)=")],  # -2
        [t("var_0=2"), lor_, t("var_1=3"), lor_, t("var_3=var_0 * var_1"), lor_, t("solve(var_3)=")],  # 6

        [t("Once when")],
        [t("Let me tell you about the time")],
        [t("Let me tell you about the time I was golfing outside one sunny afternoon when suddenly")],
    ]

    inputs = C.create_column_batch_inputs(test_prompts, num_layers, lor_ix_keys, tokenizer, device=DEVICE)

    output_logits, col_attention_mask, col_past_key_values, output_ids, elor_cache, position_ids, attention_mask = generate(model, lor_models, inputs, max_new_tokens=10)
    # for display add in col outputs
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    print('---------- gen results')
    for o in outputs:
        print(f'"""{o}"""')

# END_BLOCK_1



##################################################
# Assert that batch mode and single-row mode are equivalent
#
#   NOTE: batch vs single is incredibly sensitive to dtype. float32 is ok. bfloat16 has wild discrepancies.

# START_BLOCK_4

def remove_zero_columns(matrix):
    column_sums = matrix.abs().sum(dim=0)
    non_zero_mask = column_sums >= 1e-8
    result = matrix[:, non_zero_mask]
    return result

def get_single_sample(inputs, ix):
    ''' out of a list of columns of batches '''
    one_sample = []
    for column in inputs:
        new_column = {}
        for k in column.keys():
            if isinstance(column[k], tuple):  # eg parse lor ixs
                tup = []
                for t in column[k]:
                    tup.append(t[:, ix:ix + 1])  # shape=[layer, batch]
                new_column[k] = tuple(tup)
            else:
                new_column[k] = column[k][ix:ix + 1]  # preserve shape
        one_sample.append(new_column)
    return one_sample

inputs = C.create_column_batch_inputs(test_prompts, num_layers, lor_ix_keys, tokenizer, device=DEVICE)
for inp in inputs:
    inp['loss_mask'] = torch.ones_like(inp['loss_mask'])


# Run batch mode
batch_avg_loss, batch_out_logits, batch_cat_attention_mask, batch_lors = run_epoch(model, lor_models, [inputs], optimizer=None, device=DEVICE, train=False, debug=True)
batch_out_logits = torch.cat(batch_out_logits, dim=1)

# Collect results for each sample individually
all_single_avg_loss = []
all_single_out_logits = []
all_single_cat_attention_mask = []
all_single_lors = []
B = inputs[0]['input_ids'].shape[0]  # batch size
for ix in range(B):
    one_sample = get_single_sample(inputs, ix)
    single_avg_loss, single_out_logits, single_cat_attention_mask, single_lors = run_epoch(model, lor_models, [one_sample], optimizer=None, device=DEVICE, train=False, debug=True)
    all_single_avg_loss.append(single_avg_loss)
    all_single_out_logits.append(single_out_logits)
    all_single_cat_attention_mask.append(single_cat_attention_mask)
    all_single_lors.append(single_lors)

# Check Losses
bl = torch.tensor(batch_avg_loss)
sl = torch.tensor(all_single_avg_loss).mean()
assert torch.allclose(bl, sl, atol=1e-1), f'batch loss = {bl:>3f}, single loss = {sl:>3f}'

# Check attention masks
sa = torch.cat(all_single_cat_attention_mask, dim=0)
assert torch.allclose(batch_cat_attention_mask, sa)

# Check LoRs
lors_same = True
for batch_ix in range(B):
    for k in lor_ix_keys:
        for layer_ix in range(num_layers):
            blor = batch_lors[k][layer_ix]
            slor = all_single_lors[batch_ix][k][layer_ix]
            if blor is None:
                assert slor is None
            else:
                bllor = blor[0][batch_ix]  # left singular value cache
                brlor = blor[1][batch_ix]  # right singular value cache
                is_zero = bllor.abs().sum() < 1e-8
                if is_zero:
                    # right values are zero too
                    assert brlor.abs().sum() < 1e-8
                    assert slor is None
                else:
                    bllor_rm = remove_zero_columns(bllor)  # bllor might be bigger than sllor, bc of "non-parse" partners in batch
                    sllor = slor[0][0]  # batch of 1, left singular value cache
                    assert bllor_rm.shape == sllor.shape
                    assert torch.allclose(bllor_rm, sllor, atol=1e-3)
                    brlor_rm = remove_zero_columns(brlor)  # bllor might be bigger than sllor, bc of "non-parse" partners in batch
                    srlor = slor[1][0]  # batch of 1, right singular value cache
                    assert brlor_rm.shape == srlor.shape
                    assert torch.allclose(brlor_rm, srlor, atol=1e-3)

# Check Logits
# reorient to shape like batch version

# len(all_single_out_logits)
# 8  # batch size
# >>> len(all_single_out_logits[0])
# 22  # column count
# >>> all_single_out_logits[0][0].shape
# torch.Size([1, 17, 151936])  # [1, sequence, dim]

# transposed = zip(*all_single_out_logits)
# so = torch.cat([  # cat along sequence dim
#     torch.cat(x, dim=0)  # cat along batch dim
#     for x in transposed
# ], dim=1)

so = []
for sample in all_single_out_logits:
    so.append(torch.cat([column.squeeze(0) for column in sample], dim=0))
so = torch.stack(so, dim=0)
assert torch.allclose(batch_out_logits, so, atol=1e-3), f'{(batch_out_logits - so).abs().max().item()=}'

print('assertions passed')


# END_BLOCK_4
