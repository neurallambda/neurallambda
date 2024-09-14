'''Apply Metalearning-Hypernet stuff to Transformer arch

This version uses teacher forcing on the lor metatokens. The previous version
didn't include lor metatokens in the loss, and likely suffered from vanishing
gradients. Also, lor weights *replaced* each other in that version, so I want
to experiment with just accumulating them.


PROVENANCE:
- t14_homoiconic_llm_04.py

TODO FIX:

- [X] make use of lor_qkvo in model
- [X] replace F.interpolate with a Linear Forward in LORModule
- [X] i flipped lor_i and lor_o from lor_gud, fix
- [X] make sure to properly initialize params in LORModule


TODO:
- [ ] is loss masking shifted?
- [ ] teacher forcing by taking preceding clause, ex x=1, and setting QKVOGUD to relevant portions of that
- [ ] aggregate LORs instead of replace
- [ ] teacher forcing (via training lor weights one step, then using those)
- [ ] if not training text tokens, consider tracing what the outputs actually are
- [ ] implement LOR parsing during inference's autoregressive portion
- [ ] lor_ixs makes lor_mask irrelevant? maybe not...

IDEAS:

- [ ] after a LOR block, forbid attention to preceding tokens (or just make
      RNN). This could provide an info bottleneck, and force lors to be highly
      relevant.

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

from neurallambda.lab.common import print_model_info
from neurallambda.torch import get_last_attended_token

import t14_homoiconic_llm_add_tokens as AT

from datasets import Dataset as HFDataset

import importlib

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

DEVICE = 'cuda:0'


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
        "lor_us": [None] * num_layers,
        "lor_gs": [None] * num_layers,
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
    # alpha = 1e-3  # EXPERIMENT: shrink hidden values
    alpha = 1

    h_emb = hidden_states[lor_layer]

    for lor_key, (lor_l, lor_r) in lor_ixs.items():
        # skip parsing lors if there aren't any to do
        if (lor_l >= 0).sum() == 0:
            assert (lor_r >= 0).sum() == 0, "lor_l does not indicate any parses to be done, but lor_r does"
            continue

        is_first_pass = lors[lor_key][lor_layer] is None
        if is_first_pass:
            lors[lor_key][lor_layer] = (None, None)

        # left singular values
        left_lors = select_ixs(h_emb, lor_l)
        if is_first_pass:
            # lors[lor_key][lor_layer][0] = h_emb[:, lor_i].unsqueeze(2) * alpha
            # lors[lor_key][lor_layer][0] = left_lors.unsqueeze(2) * alpha
            new_l = left_lors.unsqueeze(2) * alpha
        else:
            # aggregate all lor weights
            # lors[lor_key][lor_layer][0] = torch.cat([lors[lor_key][lor_layer][0], h_emb[:, lor_i].unsqueeze(2) * alpha], dim=2)
            # lors[lor_key][lor_layer][0] = torch.cat([lors[lor_key][lor_layer][0], left_lors.unsqueeze(2) * alpha], dim=2)
            new_l = torch.cat([lors[lor_key][lor_layer][0], left_lors.unsqueeze(2) * alpha], dim=2)

        right_lors = select_ixs(h_emb, lor_l)
        if is_first_pass:
            # lors[lor_key][lor_layer][1] = h_emb[:, lor_i].unsqueeze(1) * alpha
            # lors[lor_key][lor_layer][1] = left_lors.unsqueeze(1) * alpha
            new_r = left_lors.unsqueeze(1) * alpha
        else:
            # aggregate all lor weights
            # lors[lor_key][lor_layer][1] = torch.cat([lors[lor_key][lor_layer][1], h_emb[:, lor_i].unsqueeze(1) * alpha], dim=2)
            # lors[lor_key][lor_layer][1] = torch.cat([lors[lor_key][lor_layer][1], left_lors.unsqueeze(1) * alpha], dim=2)
            new_r = torch.cat([lors[lor_key][lor_layer][1], left_lors.unsqueeze(1) * alpha], dim=1)

        lors[lor_key][lor_layer] = (new_l, new_r)


USE_LORS = True
def forward_lor_models(lors, lor_models, num_layers):
    ''' Zip up lor models with lors to do the forward passes '''
    if not USE_LORS:
        return lors
    else:
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

    # for i, batch_column in enumerate(col_inputs):
        # input_ids = batch_column['input_ids']
        # new_attention_mask = batch_column['attention_mask']
        # position_ids = batch_column['position_ids']

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
        if USE_LORS:
            lorf = forward_lor_models(lors, lor_models, model.config.num_hidden_layers)
        else:
            lorf = lors

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

        update_lors(lors, lor_ixs, out.hidden_states, LOR_LAYER)

        # if i % 2 == 1 and USE_LORS:
        #     # EXPERIMENT: TODO fix: for R&D, odd columns are the lor_block
        #     warnings.warn('hardcoded lor block positions')
        #     update_lors(lors, out.hidden_states, LOR_LAYER)

    return col_out_logits, attention_mask, past_key_values


##########
# Training Fns

def loss_fn(
    col_batch_in: List[torch.Tensor],  # list of blocks of TOKENS IDS (dtype=int)
    col_batch_out: List[torch.Tensor],  # list of blocks of EMBEDDINGS (dtype=float)
    loss_mask: List[torch.Tensor]  # list of blocks of masks (dtype=bool)
):
    vocab_size = col_batch_out[0].shape[2]

    # Concatenate, shift, flatten
    labels = torch.cat(col_batch_in, dim=1)[..., 1:].contiguous().view(-1)
    logits = torch.cat(col_batch_out, dim=1)[..., :-1, :].contiguous().view(-1, vocab_size)
    m = torch.cat(loss_mask, dim=1)[..., :-1].contiguous().view(-1)

    # Calculate masked loss
    loss = F.cross_entropy(logits[m], labels[m], reduction='mean')
    return loss


def run_epoch(model, lor_models, dataloader, optimizer, device, train=True):
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

            # batch_cols = [x.to(device) for x in batch_cols]
            lors = empty_lors(model.config.num_hidden_layers)  # init lors
            # out_logits, _, _  = forward_columns(model, lor_models, batch_cols, lors)
            out_logits, _, _  = forward_columns(model, lor_models, input_idss, attention_masks, position_idss, lors, lor_ixss)
            # loss = loss_fn([x['input_ids'] for x in batch_cols],
            #                out_logits,
            #                [x['loss_mask'] for x in batch_cols])
            loss = loss_fn(input_idss,
                           out_logits,
                           loss_masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples
    return avg_loss


##################################################

# model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
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

    if False:
        AT.test_token_behavior(model, tokenizer, new_token_pairs)

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

# hard code the layer these apply to
# lor_layer = 25  # qwen1.5B has 28 layers
# Q|| K|| V|| O|| G|| U|| D||
lor_block = "^@Q^@|^@|^@K^@|^@|^@V^@|^@|^@O^@|^@|^@G^@|^@|^@U^@|^@|^@D^@|^@|"

# metatoken mask, for masking loss. metatokens should be predicted, thus make
# it to loss. meta weights should not be affected directly by the cross entropy
# loss.

# TODO: i think this needs to be shifted one left, into the preceding block (??)

lor_mask = [1, 0, 0] * 7  # note: should be list, not tensor. `datasets` converts it back to a list (then tensor again) anyway
lor_mask = [0, 0, 0] * 7  # TODO: rm, this mask does NOT learn output of metatokens

# NOTE: lor_metatokens and lor_parse are offset by 1, because they parse out of
# the outputs. The metatoken ^@Q in the input gets converted into the first
# metaweight; the next input token, a dummy weight, gets converted into the
# second metaweight.

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


####################
# Data

BATCH_SIZE = 8

warnings.warn('training data is severely truncated during r&d')
train_small = load_dataset("neurallambda/arithmetic_dataset", split="train_small").select(range(100))  # todo: rm, this truncates training
test_small = load_dataset("neurallambda/arithmetic_dataset", split="test_small").select(range(100))

def process_row(row):
    prepared_data = []
    for item in row['input']:
        prepared_data.append({"type": "text", "content": item, 'include_in_loss': False, **empty_lor_ixs})
        prepared_data.append({"type": "lor", "content": block_content, 'include_in_loss': True, 'loss_mask': lor_mask, **lor_ixs})
    # Add the output
    prepared_data.append({"type": "text", "content": row['output'], 'include_in_loss': True, **empty_lor_ixs})
    return {"prepared_data": prepared_data}


def insert_lor_blocks(dataset: HFDataset, block_content) -> HFDataset:
    """
    Insert LOR blocks between text elements in the dataset.
    """
    # NOTE: the `dataset.map` process injects missing keys found among any rows, ie loss_mask
    out = dataset.map(process_row, remove_columns=["input", "output"])
    return out


# Apply the preparation to a specific split
train_small = insert_lor_blocks(train_small, lor_block)
# # Sample
# for example in train_small.select(range(2)):  # Show first 5 examples
#     print(example)
#     print()
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
test_small = insert_lor_blocks(test_small, lor_block)
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

class LORModule(nn.Module):
    def __init__(self, in_dim, out_dim, is_left_singular_value):
        super().__init__()
        self.is_left_singular_value = is_left_singular_value
        self.f = nn.Linear(in_dim, out_dim, bias=False)
        self.initialize_parameters()

    def forward(self, x):
        return self.f(x)

    def initialize_parameters(self) -> None:
        ''' This is how LoRAs are initialized, roughly '''
        if self.is_left_singular_value:
            nn.init.zeros_(self.f.weight)
        else:
            nn.init.normal_(self.f.weight, mean=0.0, std=1e-6)



### SWIGLU MLP
# class SwiGLU(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.w1 = nn.Linear(dim, dim)
#         self.w2 = nn.Linear(dim, dim)

#     def forward(self, x):
#         return F.silu(self.w1(x)) * self.w2(x)

# class LORModule(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.f = nn.Sequential(
#             SwiGLU(dim),
#             nn.Linear(dim, dim, bias=False)
#         )
#         # output layer will be initialized to be very small, yielding
#         # negligible impact at the start of training
#         self.init_gain = 0

#     def forward(self, x):
#         return self.f(x)

#     def initialize_parameters(self):
#         nn.init.xavier_normal_(self.f[1].weight, gain=self.init_gain)
#         if self.f[1].bias is not None:
#             nn.init.zeros_(self.f[1].bias)

##########
#

if True:
    LOR_LAYER = -2
    num_epochs = 10
    lr = 1e-6
    wd = 0.0


    # LOR Models: same structure as LORs
    dim = model.model.embed_tokens.weight.shape[1]
    k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
    v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
    ff_dim = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers
    lor_models = empty_lors(num_layers)
    for k in lor_models.keys():
        lor_models[k] = nn.ParameterList(lor_models[k])

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
    lor_models['lor_qs'][LOR_LAYER] = nn.ParameterList([LORModule(dim, dim, True), LORModule(dim, dim, False)])  # (left singular value (outputs), right singular value (inputs))
    lor_models['lor_ks'][LOR_LAYER] = nn.ParameterList([LORModule(dim, k_dim, True), LORModule(dim, k_dim, False)])
    lor_models['lor_vs'][LOR_LAYER] = nn.ParameterList([LORModule(dim, v_dim, True), LORModule(dim, v_dim, False)])
    lor_models['lor_os'][LOR_LAYER] = nn.ParameterList([LORModule(dim, dim, True), LORModule(dim, dim, False)])

    lor_models['lor_gs'][LOR_LAYER] = nn.ParameterList([LORModule(dim, ff_dim, True), LORModule(dim, dim, False)])
    lor_models['lor_us'][LOR_LAYER] = nn.ParameterList([LORModule(dim, ff_dim, True), LORModule(dim, dim, False)])
    lor_models['lor_ds'][LOR_LAYER] = nn.ParameterList([LORModule(dim, dim, True), LORModule(dim, ff_dim, False)])
    lor_models = nn.ParameterDict(lor_models)
    lor_models = lor_models.to(DEVICE, dtype=model.dtype)

    # All Params
    parameters = itertools.chain(model.parameters(), lor_models.parameters())

    # # LOR Params Only
    # parameters = lor_models.parameters()

    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=wd)


    # START_BLOCK_2

    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(num_epochs):
        train_loss = run_epoch(model, lor_models, train_dl, optimizer, DEVICE, train=True)
        train_losses.append(train_loss)
        test_loss = run_epoch(model, lor_models, test_dl, optimizer, DEVICE, train=False)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

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


##########

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
        if USE_LORS:
            lorf = forward_lor_models(lors, lor_models, model.config.num_hidden_layers)
        else:
            lorf = lors

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

        update_lors(lors, lor_ixs, out.hidden_states, LOR_LAYER)

        # if i % 2 == 1 and USE_LORS:
        #     # EXPERIMENT: TODO fix: for R&D, odd columns are the lor_block
        #     warnings.warn('hardcoded lor block positions')
        #     update_lors(lors, out.hidden_states, LOR_LAYER)


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

    tlor = {'content': '^@Q^@|^@|^@K^@|^@|^@V^@|^@|^@O^@|^@|^@G^@|^@|^@U^@|^@|^@D^@|^@|', 'include_in_loss': True, 'lor_ds': [18, 19], 'lor_gs': [12, 13], 'lor_ks': [3, 4], 'lor_os': [9, 10], 'lor_qs': [0, 1], 'lor_us': [15, 16], 'lor_vs': [6, 7], 'loss_mask': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'type': 'lor'}
    tempty_lor = {'lor_ds': [None, None], 'lor_gs': [None, None], 'lor_ks': [None, None], 'lor_os': [None, None], 'lor_qs': [None, None], 'lor_us': [None, None], 'lor_vs': [None, None]}
    training_prob = [
        {'content': 'var_0=-2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_1=var_0', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_2=var_1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_3=var_2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_4=var_3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_5=4 - 3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_6=1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_7=var_2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_8=-4', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'var_9=var_0 * 8', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        {'content': 'solve(var_7)=', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **tempty_lor}, tlor,
        # {'content': '-2', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **tempty_lor}
    ]


    test_prompts = [
        training_prob,
        [t("var_0=1"), lor, t("var_1=2"), lor, t("var_3=var_0 + var_1"), lor, t("solve(var_3)=")],  # 3
        [t("var_0=3"), lor, t("var_1=5"), lor, t("var_3=var_0 - var_1"), lor, t("solve(var_3)=")],  # -2
        [t("var_0=2"), lor, t("var_1=3"), lor, t("var_3=var_0 * var_1"), lor, t("solve(var_3)=")],  # 6

        # [t("hi there"), lor, t('how'), lor],
        # [t("hi there"), lor, t('how')],
        # [t("hi there"), lor],

        [t("Once when")],
        [t("Let me tell you about the time")],
        [t("Let me tell you about the time I was golfing outside one sunny afternoon when suddenly")],
    ]

    inputs = C.create_column_batch_inputs(test_prompts, lor_ix_keys, tokenizer, device='cuda')
    lors = empty_lors(num_layers)

    output_logits, col_attention_mask, col_past_key_values, output_ids, lors, position_ids, attention_mask = generate(model, lor_models, inputs, lors, max_new_tokens=40)
    # for display add in col outputs
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    print('---------- gen results')
    for o in outputs:
        print(f'"{o}"')

# END_BLOCK_1


##################################################
# sandbox
if False:
    def select_ixs(x, indices):
        batch_size, sequence_length, embedding_dim = x.shape

        # Create a mask for valid indices (non-negative)
        mask = indices >= 0

        # Use torch.where to replace negative indices (indicating default 0 values) with 0
        safe_indices = torch.where(mask, indices, torch.zeros_like(indices))

        # Gather the embeddings along the sequence dimension
        gathered_values = x[torch.arange(batch_size), safe_indices]

        # Create a zero tensor of the same shape as the gathered values
        zeros = torch.zeros_like(gathered_values)

        # Apply the mask: keep the gathered values where mask is True, otherwise use zeros
        out = torch.where(mask.unsqueeze(1), gathered_values, zeros)

        return out

    # Example usage
    x = torch.tensor([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]],

                      [[10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0],
                       [16.0, 17.0, 18.0]],

                      [[19.0, 20.0, 21.0],
                       [22.0, 23.0, 24.0],
                       [25.0, 26.0, 27.0]]], requires_grad=True)  # Enable gradient tracking

    indices = torch.tensor([
        1,  # from first sequence select index 1
        2,  # from second sequence select index 2
        -1  # for third sequence, it should default to 0
    ])

    out = select_ixs(x, indices)

    # Define a simple loss function (e.g., sum of all elements in the output)
    loss = out.sum()

    # Perform backpropagation
    loss.backward()

    # Print the gradients of x
    print(x.grad)
