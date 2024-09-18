'''

Apply Metalearning-Hypernet stuff to Transformer arch

This version uses teacher forcing on the lor metatokens. The previous version
didn't include lor metatokens in the loss, and likely suffered from vanishing
gradients. Also, lor weights *replaced* each other in that version, so I want
to experiment with just accumulating them.


PROVENANCE:
- t14_homoiconic_llm_04.py

NaN CAUSES:
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

- [ ] SMALLER DATASET (fewer vars)
- [ ] ONLY TRAIN GUD for now
- [ ] OVERFIT A SINGLE SAMPLE. If this doesn't work, test time training won't, nor teacher forcing.
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

import importlib

from datetime import datetime
current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = SummaryWriter(f'log/t14_homoiconic_llm_05/{current_time}')
LOG = True


try:
    importlib.reload(C)
    print('RELOADING C')
except NameError:
    import t14_homoiconic_llm_columnize_04 as C

try:
    importlib.reload(Q)
    print('RELOADING Q')
except NameError:
    import t14_homoiconic_llm_model as Q


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
            that the location should be filled with default 0 values.

    Returns:
        torch.Tensor: [batch, ys]

    Raises:
        ValueError: If dim is not 1 or 2.

    Note:
        - The function uses torch.where to handle negative indices, replacing them with zeros.
        - Gradients can flow through this operation, making it suitable for use in
          differentiable computations.

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


def update_lors(
        lors,  # shaped like `empty_lors`
        lor_ixs: Dict[str, List[Tuple[Union[int, None], Union[int, None]]]],  # {'lor_qs': [(left_ix, right_ix), ...n_layers]}
        hidden_states,  # from out.hidden_states
        lor_layer
):
    ''' Update the LORs by interpreting hidden states as new lor blocks '''

    warnings.warn(f'taking final layer of hidden states for projection to lor_layer={lor_layer}') # TODO
    # h_emb = hidden_states[lor_layer]
    h_emb = hidden_states[-1]

    for lor_key, (lor_l, lor_r) in lor_ixs.items():
        # skip parsing lors if there aren't any to do
        if (lor_l >= 0).sum() == 0:
            assert (lor_r >= 0).sum() == 0, "lor_l does not indicate any parses to be done, but lor_r does"
            continue

        is_first_pass = lors[lor_key][lor_layer] is None

        # left singular values
        left_lors = select_ixs(h_emb, lor_l)
        if is_first_pass:
            new_l = left_lors.unsqueeze(2)
        else:
            new_l = torch.cat([lors[lor_key][lor_layer][0], left_lors.unsqueeze(2)], dim=2)

        # right singular values
        right_lors = select_ixs(h_emb, lor_r)
        if is_first_pass:
            new_r = right_lors.unsqueeze(1)
        else:
            new_r = torch.cat([lors[lor_key][lor_layer][1], right_lors.unsqueeze(1)], dim=1)

        lors[lor_key][lor_layer] = (new_l, new_r)

    return lors

def forward_lor_models(lors, lor_models, num_layers):
    ''' Zip up lor models with lors to do the forward passes '''
    out = empty_lors(num_layers)

    for k in lor_models.keys():
        for l_ix in range(num_layers):
            if lor_models[k][l_ix] is None:
                # skip layers that we're not targeting with LOR stuff
                assert lors[k][l_ix] is None, f'lor_model exists for for {k}:{l_ix}, but the corresponding lor weights are present'
                continue
            if lors[k][l_ix] is not None:
                # on the first pass, the corresponding lor weights don't exist yet
                out[k][l_ix] = (
                    lor_models[k][l_ix][0](lors[k][l_ix][0].permute(0, 2, 1)).permute(0, 2, 1),  # left singular value
                    lor_models[k][l_ix][1](lors[k][l_ix][1]),  # right singular value
                )
    return out


def forward_columns(model, lor_models, input_idss, attention_masks, position_idss, lors, lor_ixss):
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

        # forward versions of lors
        lorf = forward_lor_models(lors, lor_models, model.config.num_hidden_layers)

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,  # OPTIM: we don't need this for non-lor blocks
                    **lorf,
                    )
        col_out_logits.append(out.logits)

        past_key_values = out.past_key_values
        assert past_key_values is not None, "past_key_values cannot be None. Typically `transformers` models won't return past_key_values during training, so, you need to modify the underlying model class."

        lors = update_lors(lors, lor_ixs, out.hidden_states, LOR_LAYER)

    return col_out_logits, attention_mask, past_key_values, lors


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
                lor_ixss.append(lor_ixs)

            lors = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch
            out_logits, cat_attention_mask, _ , lors = forward_columns(model, lor_models, input_idss, attention_masks, position_idss, lors, lor_ixss)

            loss = loss_fn(input_idss,
                           out_logits,
                           loss_masks)

            if torch.isnan(loss):
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
        return avg_loss, out_logits, cat_attention_mask, lors
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
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float32,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

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
        'lor_gs': (None, None),
        'lor_us': (None, None),
        'lor_ds': (None, None),
    }

    # where to parse to collect metaweights, tied to `lor_block`'s implementation.
    #
    # note: this might look shifted left, but consider "Q||". The token emitted
    #       after Q will be in location 0 and represent the first predicted |.

    lor_ixs = {
        'lor_gs': (0, 1),  # (left singular value, right singular value)
        'lor_us': (3, 4),
        'lor_ds': (6, 7),
    }

    # warnings.warn('TODO: is this off by one?')
    # lor_ixs = {
    #     'lor_gs': (1, 2),  # (left singular value, right singular value)
    #     'lor_us': (4, 5),
    #     'lor_ds': (7, 8),
    # }

    lor_ix_keys = set(lor_ixs.keys())


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

        prepared_data.append({"type": "lor", "content": lor_block, 'include_in_loss': True, 'loss_mask': lor_mask, **lor_ixs})

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
    lor_ix_keys=lor_ix_keys,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    device='cpu',
    shuffle=False,
    num_workers=0
)

##########
#

### IDENTITY

# class LORModule(nn.Module):
#     def __init__(self, dim):
#         super().__init__()

#     def forward(self, x):
#         return x


### Linear

# class LORModule(nn.Module):
#     def __init__(self, in_dim, out_dim, is_left_singular_value):
#         super().__init__()
#         self.is_left_singular_value = is_left_singular_value
#         self.f = nn.Linear(in_dim, out_dim, bias=False)
#         # self.norm = nn.LayerNorm(out_dim)  # NOTE: nan fix?
#         self.initialize_parameters()

#     def forward(self, x):
#         # return self.norm(self.f(x))
#         return self.f(x)

#     def initialize_parameters(self) -> None:
#         if self.is_left_singular_value:
#             nn.init.zeros_(self.f.weight)
#             # nn.init.normal_(self.f.weight, mean=0.0, std=1e-2)
#         else:
#             # nn.init.zeros_(self.f.weight)
#             nn.init.normal_(self.f.weight, mean=0.0, std=1e-4)
#             # pass


### SWIGLU MLP

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

class LORModule(nn.Module):
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




##########
#

if True:
    LOR_LAYER = 14  # must be positive num (can't index bwd)
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
    num_layers = model.config.num_hidden_layers

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
    for k in lor_models.keys():
        lor_models[k] = nn.ModuleList(lor_models[k])

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
    lor_models['lor_qs'][LOR_LAYER] = nn.ModuleList([LORModule(dim, dim, True), LORModule(dim, dim, False)])  # (left singular value (outputs), right singular value (inputs))
    lor_models['lor_ks'][LOR_LAYER] = nn.ModuleList([LORModule(dim, k_dim, True), LORModule(dim, k_dim, False)])
    lor_models['lor_vs'][LOR_LAYER] = nn.ModuleList([LORModule(dim, v_dim, True), LORModule(dim, v_dim, False)])
    lor_models['lor_os'][LOR_LAYER] = nn.ModuleList([LORModule(dim, dim, True), LORModule(dim, dim, False)])

    lor_models['lor_gs'][LOR_LAYER] = nn.ModuleList([LORModule(dim, ff_dim, True), LORModule(dim, dim, False)])
    lor_models['lor_us'][LOR_LAYER] = nn.ModuleList([LORModule(dim, ff_dim, True), LORModule(dim, dim, False)])
    lor_models['lor_ds'][LOR_LAYER] = nn.ModuleList([LORModule(dim, dim, True), LORModule(dim, ff_dim, False)])
    lor_models = lor_models.to(DEVICE, dtype=model.dtype)

    # All Params
    parameters = list(itertools.chain(model.parameters(), lor_models.parameters()))

    # # LOR Params Only
    # parameters = list(lor_models.parameters())

    # # Embeddings. Add all, and then zero out grads to target just the new tokens
    # parameters = parameters + list(model.model.embed_tokens.parameters())

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
            test_loss = run_epoch(model, lor_models, test_dl, optimizer, DEVICE, train=False)  # TODO: turned off testing
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

def generate(model, lor_models, col_inputs, lors, max_new_tokens):
    '''Recurrently process column blocks of ids, concatenating attention_mask and
    past_key_values across generations. Can generate new tokens if max_new_tokens > 0.'''
    col_out_logits = []
    past_key_values = None
    attention_mask = None
    all_output_ids = []
    tokens_generated = 0

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

        # forward versions of lors
        lorf = forward_lor_models(lors, lor_models, model.config.num_hidden_layers)

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,  # OPTIM: we don't need this for non-lor blocks
                    **lorf,
                    )
        col_out_logits.append(out.logits)
        # all_output_ids.append(input_ids)

        past_key_values = out.past_key_values
        assert past_key_values is not None, "past_key_values cannot be None. Typically `transformers` models won't return past_key_values during training, so, you need to modify the underlying model class."

        lors = update_lors(lors, lor_ixs, out.hidden_states, LOR_LAYER)

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

        # Generate the next token
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    **lorf,
                    )

        # TODO: lor updates aren't implemented during autoregression
        col_out_logits.append(out.logits)
        past_key_values = out.past_key_values

        # sample the next token
        input_ids = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        all_output_ids.append(input_ids)

        tokens_generated += 1

    return col_out_logits, attention_mask, past_key_values, torch.cat(all_output_ids, dim=-1), lors, position_ids, attention_mask


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

    lor = w(lor_block, lor_ixs)

    tlor = {'content': '^@Q^@|^@|^@K^@|^@|^@V^@|^@|^@O^@|^@|^@G^@|^@|^@U^@|^@|^@D^@|^@|',
            'include_in_loss': True,
            'lor_ds': [18, 19], 'lor_gs': [12, 13], 'lor_ks': [3, 4], 'lor_os': [9, 10], 'lor_qs': [0, 1], 'lor_us': [15, 16], 'lor_vs': [6, 7],
            'loss_mask': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'type': 'lor'}
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
        [t("var_0=1"), lor, t("var_1=2"), lor, t("var_3=var_0 + var_1"), lor, t("solve(var_3)=")],  # 3
        [t("var_0=3"), lor, t("var_1=5"), lor, t("var_3=var_0 - var_1"), lor, t("solve(var_3)=")],  # -2
        [t("var_0=2"), lor, t("var_1=3"), lor, t("var_3=var_0 * var_1"), lor, t("solve(var_3)=")],  # 6

        [t("Once when")],
        [t("Let me tell you about the time")],
        [t("Let me tell you about the time I was golfing outside one sunny afternoon when suddenly")],
    ]

    inputs = C.create_column_batch_inputs(test_prompts, lor_ix_keys, tokenizer, device=DEVICE)
    lors = empty_lors(num_layers)

    output_logits, col_attention_mask, col_past_key_values, output_ids, lors, position_ids, attention_mask = generate(model, lor_models, inputs, lors, max_new_tokens=10)
    # for display add in col outputs
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    print('---------- gen results')
    for o in outputs:
        print(f'"""{o}"""')

# END_BLOCK_1



##################################################


# START_BLOCK_4

def clean(tensor, pad_token_id=151643):
    # Move tensor to CPU if it's on CUDA
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert tensor to a list and filter out pad tokens
    cleaned_list = [item.item() for item in tensor if item.item() != pad_token_id]

    # Convert back to tensor
    return torch.tensor(cleaned_list)


batch0 = next(iter(train_dl))
a = clean(torch.cat([x['input_ids'][0] for x in batch0], dim=0)[:-2])  # trim off answer
b = clean(torch.cat([x['input_ids'][0] for x in inputs], dim=0))

ix = 0
inp = [{
    'input_ids': x['input_ids'][ix:ix+1],
    'attention_mask': x['attention_mask'][ix:ix+1],
    'position_ids': x['position_ids'][ix:ix+1],
    'loss_mask': x['loss_mask'][ix:ix+1],
    'lor_qs': (x['lor_qs'][0][ix:ix+1], x['lor_qs'][1][ix:ix+1]),
    'lor_ks': (x['lor_ks'][0][ix:ix+1], x['lor_ks'][1][ix:ix+1]),
    'lor_vs': (x['lor_vs'][0][ix:ix+1], x['lor_vs'][1][ix:ix+1]),
    'lor_os': (x['lor_os'][0][ix:ix+1], x['lor_os'][1][ix:ix+1]),
    'lor_gs': (x['lor_gs'][0][ix:ix+1], x['lor_gs'][1][ix:ix+1]),
    'lor_us': (x['lor_us'][0][ix:ix+1], x['lor_us'][1][ix:ix+1]),
    'lor_ds': (x['lor_ds'][0][ix:ix+1], x['lor_ds'][1][ix:ix+1]),
} for x in inputs]

avg_loss, out_logits, cat_attention_mask, lors = run_epoch(model, lor_models, [inp], optimizer=None, device=DEVICE, train=False, debug=True)
out_logits = torch.cat(out_logits, dim=1)

print('loss:', avg_loss)
print(tokenizer.decode(out_logits[0].argmax(dim=1)))

# output_logits, col_attention_mask, col_past_key_values, output_ids, lors, position_ids, attention_mask = generate(model, lor_models, inputs, lors, max_new_tokens=10)

# END_BLOCK_4
