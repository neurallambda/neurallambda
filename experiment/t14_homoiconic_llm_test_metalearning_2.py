'''.

Continue the t13 series, and do N-way k-shot learning task, where a transformer
backbone outputs LOR updates to itself. Trained with "train-time-data", so
hopefully, at inference we won't even need metalearning!

----------
Model/Data Design

- N*k support images embedded continuously as inputs
- N*k labels, tokenized and embedded, as standard
- N*q query images
- N*q query labels
- After all support images, randn LoR tokens are appended
- Things I might do:
    - These tokens get trained via metalearning at train time
    - The resulting lors from metalearning could be trained as targets for the outer loop ("train-time-data")
    - Do we need metalearning at inference time?

----------
N-way k-shot learning task:
    1. N classes: The task tests classification between N different classes that the model has not seen during training.
    2. k examples per class: For each of the N classes, the model is given only k labeled examples (usually k is a small number like 1 or 5).
    3. Support set: The N*k labeled examples make up the "support set" that the model can use to learn about the new classes.
    4. Query set: The model is then asked to classify new unlabeled examples (the "query set") from the N classes.
    5. Meta-learning: Models are typically trained on many different N-way k-shot tasks so they can learn to adapt quickly to new tasks at test time.
    6. Evaluation: Performance is measured by classification accuracy on the query set.

Metalearning with pretraining on Omniglot, Testing on Mini-Imagenet Test


PROVENANCE:
- t13_metalearning_hypernet_03


TODO:
- [ ] fix run_epoch
  - [ ] update for new dataset
  - [ ] add image projector
  - [ ] tokenize labels
  - [ ] append metatokens
  - [ ] metalearn them
  - [ ] prove loss can decrease for supports before worrying about queries
  - [ ] test on queries

- options
  1. [ ] keep model frozen, only train LORProject
  2. [ ] use metalearned tokens in outerloop, and train entire model

'''

import os
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
from functools import partial
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

from t13_metalearning_hypernet_data import omniglot_n_way_k_shot, ALPHABET_DICT
from neurallambda.lab.common import print_model_info

try:
    importlib.reload(Q)
    print('RELOADING Q')
except NameError:
    import t14_homoiconic_llm_model_03 as Q


current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = SummaryWriter(f'log/t14_homoiconic_llm_test_metalearning_2/{current_time}')
LOG = True

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'
WHICH_LOR = 1
N_METAWEIGHTS = 14  # QKVOGUD * 2
LOR_LAYER = 14  # half wayish


##################################################
# Load Model

model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-7B")

def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        isnan = torch.isnan(output).any()
        isinf = torch.isinf(output).any()
        if isnan or isinf:
            print(f"NaN or inf detected in {module.__class__.__name__}: {isnan=}, {isinf=}")
            print(f"Module: {module}")
            print(f"Module name: {module_names.get(module, 'Unknown')}")
            print(f"Input shape: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}")
            print(f"Output shape: {output.shape}")
            breakpoint()
            # raise RuntimeError("NaN or inf detected: {isnan=}, {isinf=}")

module_names = {}

def add_hooks(model):
    for name, module in model.named_modules():
        module_names[module] = name
        module.register_forward_hook(hook_fn)

try:
    already_loaded
except:
    print('Loading model')
    model = Q.Qwen2ForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.bfloat16,  # HERE BE DRAGONS
        torch_dtype=torch.float32,
        # torch_dtype=torch.float64,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    num_layers = model.config.num_hidden_layers

    add_hooks(model)
    already_loaded = True



##################################################
# Functional Optimizers
#
#   Differentiable variants of: SGD, SGD-Momentum, RMSProp, Adam


##########
# SGD

def sgd(params, grads, lr=0.01):

    # # structured dict variant
    # return {k: p - grads[k] * lr for k, p in params.items()}

    return [p - g * lr for p, g in zip(params, grads)]


##########
# SGD Momentum

def sgd_momentum_init(params):
    return {k: torch.zeros_like(p) for k, p in params.items()}

def sgd_momentum(params, grads, velocity, lr=0.01, momentum=0.9):
    updated_params = {}
    updated_velocity = {}
    for k in params.keys():
        v_new = momentum * velocity[k] + lr * grads[k]
        updated_params[k] = params[k] - v_new
        updated_velocity[k] = v_new
    return updated_params, updated_velocity


##########
# RMSProp

def rmsprop_init(params):
    return {k: torch.zeros_like(p) for k, p in params.items()}  # square averages

def rmsprop(params, grads, square_avg, lr=0.01, alpha=0.99, eps=1e-8):
    updated_params = {}
    updated_square_avg = {}
    for k in params.keys():
        avg_new = alpha * square_avg[k] + (1 - alpha) * grads[k].pow(2)
        updated_params[k] = params[k] - lr * grads[k] / (avg_new.sqrt() + eps)
        updated_square_avg[k] = avg_new
    return updated_params, updated_square_avg


##########
# ADAM

def adam_init(params):
    return (
        {k: torch.zeros_like(p) for k, p in params.items()},  # m
        {k: torch.zeros_like(p) for k, p in params.items()},  # v
        0  # t
    )

def adam(params, grads, m, v, t, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
    updated_params = {}
    updated_m = {}
    updated_v = {}
    t += 1
    for k in params.keys():
        m_new = betas[0] * m[k] + (1 - betas[0]) * grads[k]
        v_new = betas[1] * v[k] + (1 - betas[1]) * grads[k].pow(2)
        m_hat = m_new / (1 - betas[0]**t)
        v_hat = v_new / (1 - betas[1]**t)
        updated_params[k] = params[k] - lr * m_hat / (v_hat.sqrt() + eps)
        updated_m[k] = m_new
        updated_v[k] = v_new
    return updated_params, updated_m, updated_v, t


##################################################
# LOR stuff


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
    lor_ix_keys = lor_cache.keys()
    for k in lor_ix_keys:
        fs[k] = []
        for m, c in zip(lor_models[k], lor_cache[k]):
            if m is not None:
                f = partial(m, c)
            else:
                f = None
            fs[k].append(f)
    return fs


def update_lors(
        lor_models,  # shaped like `empty_lors`. Contains lor_proj, and per-head models
        lor_cache,  # shaped like `empty_lors`. Contains previously parsed lors
        lor_ixs_per_layer: List[torch.Tensor],  # List[spans], where a span is [batch, (start_ix, end_ix)]
        hidden_states,  # from out.hidden_states, so, contains all layers
        num_layers,
):
    ''' Update the LORs by interpreting hidden states as new lor blocks '''

    # check that lor_models and lor_cache are defined (at least None values)
    # for each layer, and that there's a model for every cache key
    lor_keys = lor_cache.keys()
    for k in lor_keys:
        assert (
            len(lor_models[k]) ==  # one (optional) lor module per layer
            len(lor_cache[k]) ==  # cache per layer
            num_layers
        ), (f'''
{len(lor_models[k])=} ==  # one (optional) lor module per layer
{len(lor_cache[k])=} ==  # cache per layer
{num_layers=}
''')

    h_emb = hidden_states[-1]  # final layer states

    # iterate over all layers
    for layer_ix in range(num_layers):
        lor_ix_spans = lor_ixs_per_layer[layer_ix]

        # skip non-lor'd layers
        if lor_models['lor_proj'][layer_ix] is None:
            assert lor_ix_spans is None, f'lor_proj is not defined for layer {layer_ix}, but there are lor_ix_spans is defined for this layer'
            continue

        # check that spans are within bounds
        assert isinstance(lor_ix_spans, torch.Tensor)
        assert lor_ix_spans.min() >= -1
        assert lor_ix_spans.max() <= hidden_states[-1].shape[1]


        parses = select_spans(h_emb, lor_ix_spans)

        # no parses implied anywhere
        if (parses > -1).sum() == 0:
            continue

        # run lor_proj. Returns tuple of L and R singular values, per key, eg: (lor_qs_l, lor_qs_r, ...)
        projs = lor_models['lor_proj'][layer_ix](parses)
        proj_pairs = zip(projs[::2], projs[1::2])

        # update cache
        for k, (l, r) in zip(lor_keys, proj_pairs):
            # TODO: prob shouldnt require grads here
            l = l.requires_grad_()
            r = r.requires_grad_()

            if lor_cache[k][layer_ix] is None:  # is first pass, no cache yet
                lor_cache[k][layer_ix] = (l.unsqueeze(2), r.unsqueeze(2))  # [B, DIM, RANK]
            else:
                lor_cache[k][layer_ix] = (torch.cat([lor_cache[k][layer_ix][0], l.unsqueeze(2)], dim=2),
                                          torch.cat([lor_cache[k][layer_ix][1], r.unsqueeze(2)], dim=2))  # [B, DIM, RANK]

    return lor_cache


def select_spans(x, indices):
    """Selects spans from a 3D tensor (`[batch, seq, dim]`) along dim=1 using
    provided start and end indices.

    Perform span selection on a 3D tensor. If `indices` contains [-1, -1] for a
    batch, that location will be filled with 0s.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, seq, dim] where:
            batch: number of sequences in the batch
            seq: length of each sequence
            dim: dimensionality of each token representation

        indices (torch.Tensor): 2D tensor of indices for span selection, with shape
            [batch, 2]. Each row contains [start, end] indices for the span.
            Start and end are inclusive. If a row is [-1, -1], the corresponding
            output will be filled with 0s.

    Returns:
        torch.Tensor: Output tensor of shape [batch, max_span_length, dim]

    Example:
        >>> x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        ...                   [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]])
        >>> indices = torch.tensor([[1, 2], [-1, -1]])
        >>> select_spans(x, indices)
        tensor([[[ 4,  5,  6],
                 [ 7,  8,  9],
                 [ 0,  0,  0]],
                [[ 0,  0,  0],
                 [ 0,  0,  0],
                 [ 0,  0,  0]]])
    """
    B, S, D = x.shape  # batch, sequence length, dimension

    # Create a mask for valid spans (not [-1, -1])
    mask = (indices[:, 0] != -1)  # we assume -1s are paired correctly with another -1

    # Calculate span lengths
    span_lengths = torch.where(mask, indices[:, 1] - indices[:, 0] + 1, torch.zeros_like(indices[:, 0]))
    max_span_length = span_lengths.max().item()

    # Create position indices for each element in the max span
    positions = torch.arange(max_span_length, device=x.device).unsqueeze(0).expand(B, -1)

    # Calculate absolute indices for each position
    abs_indices = indices[:, 0].unsqueeze(1) + positions

    # Create a mask for valid positions within each span
    valid_positions = positions < span_lengths.unsqueeze(1)

    # Combine the span mask and position mask
    final_mask = mask.unsqueeze(1) & valid_positions

    # Create batch indices
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, max_span_length)

    # Gather values using absolute indices, with out-of-bounds handling
    gathered_values = torch.zeros((B, max_span_length, D), device=x.device, dtype=x.dtype)
    valid_abs_indices = abs_indices[final_mask]
    valid_batch_indices = batch_indices[final_mask]
    gathered_values[final_mask] = x[valid_batch_indices, valid_abs_indices]

    return gathered_values



##################################################
# LOR Models: LORProject + LORNorm


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


class LORProject(nn.Module):
    '''
    LOR weights need to be projected to fit the shapes of the underlying
    Linear layers they're matching. These LORModules solve this, and there
    are 2 modules per target linear layer, (left singular values, right
    singular values). They multiply like: out=LRx, to match the usual out=Wx,
    so the new calculation becomes out=Wx + LRx.

    R are the input vectors, and L are the output vectors. The first
    dimension of the LORModules must match the embedding that we're
    projecting, so the 1st values are all `dim`. The 2nd dim of R is the
    input dimension of the matched linear weights. The 2nd dim of L is the
    output dimension of the same linear layer.

    '''

    def __init__(self, dropout_rate=0.2):
        super().__init__()

        dim = model.model.embed_tokens.weight.shape[1]  # embedding dimension
        k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
        v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
        ff_dim = model.config.intermediate_size

        self.dim = dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.ff_dim = ff_dim

        self.input_dims = [dim] * N_METAWEIGHTS  # All inputs are 'dim'-dimensional

        n_out = 14
        self.output_dims = [
            dim, dim,  # lor_qs_l, lor_qs_r
            k_dim, dim,  # lor_ks_l, lor_ks_r
            v_dim, dim,  # lor_vs_l, lor_vs_r
            dim, dim,  # lor_os_l, lor_os_r
            ff_dim, dim,  # lor_gs_l, lor_gs_r
            ff_dim, dim,  # lor_us_l, lor_us_r
            dim, ff_dim  # lor_ds_l, lor_ds_r
        ]

        self.token_mixing_dim = 128
        self.channel_mixing_dim = 128

        # Token-mixing MLP
        self.token_mixing_mlp = nn.Sequential(
            nn.RMSNorm(N_METAWEIGHTS),
            nn.Linear(N_METAWEIGHTS, self.token_mixing_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.token_mixing_dim, n_out, bias=False),
            nn.Dropout(dropout_rate)
        )

        # Channel-mixing MLP (same for all inputs)
        self.channel_mixing_mlp = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, self.channel_mixing_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.channel_mixing_dim, dim, bias=False),
            nn.Dropout(dropout_rate)
        )

        # Final projection layers
        self.final_projections = nn.ModuleDict({
            'lor_qs_l': nn.Linear(dim, dim, bias=False),
            'lor_qs_r': nn.Linear(dim, dim, bias=False),
            'lor_ks_l': nn.Linear(dim, k_dim, bias=False),
            'lor_ks_r': nn.Linear(dim, dim, bias=False),
            'lor_vs_l': nn.Linear(dim, v_dim, bias=False),
            'lor_vs_r': nn.Linear(dim, dim, bias=False),
            'lor_os_l': nn.Linear(dim, dim, bias=False),
            'lor_os_r': nn.Linear(dim, dim, bias=False),
            'lor_gs_l': nn.Linear(dim, ff_dim, bias=False),
            'lor_gs_r': nn.Linear(dim, dim, bias=False),
            # 'lor_us_l': nn.Linear(dim, ff_dim, bias=False),
            'lor_us_r': nn.Linear(dim, dim, bias=False),
            'lor_ds_l': nn.Linear(dim, dim, bias=False),
            # 'lor_ds_r': nn.Linear(dim, ff_dim, bias=False),

            # 'lor_us_l_ds_r': nn.Linear(dim, ff_dim, bias=False),

        })


        # LORModule must play nicely in a batch situation, where some samples
        # of the batch imply lor parses and others don't. Non LoR'd samples
        # should not be affected by sharing a batch with LoR'd samples. Biases
        # corrupt this property. 0-valued lors (from samples without lor
        # parses) must produce 0-valued outputs here. Checking for biases is
        # not the whole solution, you must take care.
        #
        # This is not necessarily necessary. For instance, clever masking of
        # non-parsed samples might obviate this.
        assert_no_biases(self)


    def forward(self, x):
        '''
        x: [B, N_METAWEIGHTS, D]
        '''

        B = x.shape[0]
        device = x.device

        # Token-mixing
        residual = x
        # x_token shape: [batch, dim, 14]
        x_token = x.transpose(1, 2)
        # Normalize across token dimension
        x_token = self.token_mixing_mlp(x_token)
        x_token = x_token.transpose(1, 2)  # [batch, 14, dim]

        # x = residual + x_token  # TODO: residuals off

        # Channel-mixing
        residual = x
        x = self.channel_mixing_mlp(x)

        # x = residual + x  # TODO: residuals off

        ##########
        # Original attempt

        # # Final projections to adjust dimensions
        # outputs = []
        # for i, proj in enumerate(self.final_projections.values()):
        #     outputs.append(proj(x[:, i, :]))


        ##########
        # Tie intermediates. Adopting results from (ie, results from t14_homoiconic_llm_adding_data_to_mlp)

        # ud_intermediate = self.final_projections['lor_us_l_ds_r']


        # NOTE: statistics of randn are likely way off
        # TODO: bc of this shrink, shrink token_mixing_mlp from dim=14 to dim=12, since those outputs go unused
        ud_intermediate = torch.randn(B, self.ff_dim, device=device)

        outputs = (
            self.final_projections['lor_qs_l'](x[:, 0, :]),
            self.final_projections['lor_qs_r'](x[:, 1, :]),
            self.final_projections['lor_ks_l'](x[:, 2, :]),
            self.final_projections['lor_ks_r'](x[:, 3, :]),
            self.final_projections['lor_vs_l'](x[:, 4, :]),
            self.final_projections['lor_vs_r'](x[:, 5, :]),
            self.final_projections['lor_os_l'](x[:, 6, :]),
            self.final_projections['lor_os_r'](x[:, 7, :]),
            self.final_projections['lor_gs_l'](x[:, 8, :]),
            # ud_intermediate * -1,
            self.final_projections['lor_gs_r'](x[:, 9, :]),
            # self.final_projections['lor_us_l'](x[:, 10, :]),
            ud_intermediate,
            self.final_projections['lor_us_r'](x[:, 11, :]),
            self.final_projections['lor_ds_l'](x[:, 12, :]),
            # self.final_projections['lor_ds_r'](x[:, 13, :]),
            ud_intermediate,
        )

        return outputs

class LORNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.norm = nn.RMSNorm(out_dim)

        # NOTE: there is probably a principled way of setting this, or
        #   pre-learning this. As it stands, the norm can be initialized even
        #   to 0 which effectively removes this entire layer from the
        #   transformer stack EXCEPT residuals still pass forward. Strangely
        #   enough, the network can do ok with a layer removed. I'm thinking
        #   here that we can limit the initial impact of LoR stuff by setting
        #   this low. This has a SIDE-EFFECT though of small grads to this layer.
        with torch.no_grad():
            self.norm.weight[:] = self.norm.weight * 1e-2

        # This is akin to He initialization. TODO: worth it?
        self.scale = (2 / in_dim) ** 0.5

        # LORModule must play nicely in a batch situation, where some samples
        # of the batch imply lor parses and others don't. Non LoR'd samples
        # should not be affected by sharing a batch with LoR'd samples. Biases
        # corrupt this property. 0-valued lors (from samples without lor
        # parses) must produce 0-valued outputs here. Checking for biases is
        # not the whole solution, you must take care.
        #
        # This is not necessarily necessary. For instance, clever masking of
        # non-parsed samples might obviate this.
        assert_no_biases(self)

    def forward(self, lor_cache, original, hidden_state):
        '''This gets applied separately per QKVOGUD, and really just allows
        Normalization to be handled from this module, instead of say adding a
        new norm layer throughout the underlying model.

        The `project` function is where more magic happens; it takes in ALL QKVOGUD parses for a layer, and generates a cache for each together.

        Args:
          original: model's original values of eg QKVOGUD within this layer
          hidden_state: hidden_state at this layer, that projects through this layer's associated linear QKVOGUD block, and will be used to project through the LoR version too.

        '''

        if lor_cache is not None:
            lorl, lorr = lor_cache
            l = apply_lor(hidden_state, lorl, lorr)
            # return self.norm(original + l * self.scale)  # TODO: revisit if `scale` is good
            return self.norm(original + l)
        else:
            return self.norm(original)



##################################################
# Training

def run_epoch(model, dataloader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):

            # batch (Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]):
            #     A batch containing a single task, where:
            #     - The first element is the support set: a list of N*k tuples, each containing
            #       (batched_image, batched_label) for support examples.
            #     - The second element is the query set: a list of N*q tuples, each containing
            #       (batched_image, batched_label) for query examples.

            # (Pdb) type(batch)
            # <class 'list'>
            # (Pdb) len(batch)
            # 2
            # (Pdb) type(batch[0])
            # <class 'list'>
            # (Pdb) len(batch[0])
            # 5
            # (Pdb) type(batch[0][0])
            # <class 'list'>
            # (Pdb) len(batch[0][0])
            # 2
            # (Pdb) batch[0][0][0].shape
            # torch.Size([32, 28, 28])
            # (Pdb) batch[0][0][1].shape
            # torch.Size([32])


            # supports: N*k tuples
            # queries: N queries (or N*q if multiple queries)
            supports, queries = batch
            support_imgs = [x[0].to(device).unsqueeze(1) for x in supports]  # N*k tensors, shape=[B, 1, IMG_SIZE, IMG_SIZE]
            support_labels = [x[1].to(device) for x in supports]  # N*k tensors, shape=[B]
            query_imgs = [x[0].to(device).unsqueeze(1) for x in queries]  # N*k tensors, shape=[B, 1, IMG_SIZE, IMG_SIZE]
            query_labels = [x[1].to(device) for x in queries]  # N*k tensors, shape=[B]
            B = query_labels[0].shape[0]  # batch size

            #####
            # Go

            loss = 0
            for img, target_label in zip(support_imgs + query_imgs, support_labels + query_labels):
                output_labels = model(img)
                loss = loss + F.cross_entropy(output_labels, target_label)

            loss = loss / (len(support_imgs) + len(query_imgs))

            if train:
                optimizer.zero_grad()
                loss.backward()

                # Grad clip
                MAX_GRAD_NORM = 1.0
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples

    # Log weight histogram
    if LOG and train:
        try:
            # for name, param in itertools.chain(model.named_parameters(), lor_models.named_parameters()):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'weights/{name}', param.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'grads/{name}', param.grad.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
        except Exception as e:
            warnings.warn(f'Failed to write to tensorboard: {e}')

    return avg_loss


def run_epoch(model, img_proj, lor_models, inner_lr, dataloader, optimizer, device, train=True, debug=False):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    num_layers = model.config.num_hidden_layers
    D = model.config.hidden_size
    vocab_size = model.model.embed_tokens.weight.shape[0]

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):

            # batch (Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]):
            #     A batch containing a single task, where:
            #     - The first element is the support set: a list of N*k tuples, each containing
            #       (batched_image, batched_label) for support examples.
            #     - The second element is the query set: a list of N*q tuples, each containing
            #       (batched_image, batched_label) for query examples.

            # (Pdb) type(batch)
            # <class 'list'>
            # (Pdb) len(batch)
            # 2
            # (Pdb) type(batch[0])
            # <class 'list'>
            # (Pdb) len(batch[0])
            # 5
            # (Pdb) type(batch[0][0])
            # <class 'list'>
            # (Pdb) len(batch[0][0])
            # 2
            # (Pdb) batch[0][0][0].shape
            # torch.Size([32, 28, 28])
            # (Pdb) batch[0][0][1].shape
            # torch.Size([32])


            # supports: N*k tuples of batched images and labels
            # queries: N tuples (or N*q if multiple queries) of batched images and labels
            supports, queries = batch

            # Move to device, flatten and project images into embedding dim
            support_imgs = img_proj(torch.stack([x[0].to(device).flatten(start_dim=1, end_dim=2) for x in supports], dim=1))  # N*k tensors [B, IMG, IMG] -> [B, N*k, D]
            support_labels = torch.stack([x[1].to(device) for x in supports], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
            query_imgs = img_proj(torch.stack([x[0].to(device).flatten(start_dim=1, end_dim=2) for x in queries], dim=1))  # N*k tensors [B, IMG, IMG] -> [B, N*k, D]
            query_labels = torch.stack([x[1].to(device) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
            B, S = query_labels.shape  # batch size, sequence (N*k)

            empty_lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch

            metaweights = torch.randn(B, N_METAWEIGHTS, D, device=device) * 1e-3  # TODO: what should this init be?
            inputs_embeds = torch.cat([
                support_imgs,
                metaweights], dim=1)

            img_attention_mask = torch.ones((B, S), device=device, dtype=torch.long)
            meta_attention_mask = torch.ones((B, N_METAWEIGHTS), device=device, dtype=torch.long)
            attention_mask = torch.cat([
                img_attention_mask,
                meta_attention_mask
            ], dim=1)

            uncausal_mask = torch.cat([
                torch.ones_like(img_attention_mask, dtype=torch.bool),  # NOTE: whole sequence is uncausally masked
                torch.ones_like(meta_attention_mask, dtype=torch.bool)
            ], dim=1)


            # span ixs of metaweights to parse out (inclusive span). currently
            # lor_ixs is only defined for LOR_LAYER, other spans are None.
            lor_ixs = torch.zeros(B, 2, dtype=torch.long, device=device)
            with torch.no_grad():
                lor_ixs[:, 0] = S
                lor_ixs[:, 1] = S + N_METAWEIGHTS - 1

            lor_ixs_per_layer = (
                [None] * LOR_LAYER +
                [lor_ixs]  +
                [None] * (num_layers - LOR_LAYER - 1)
            )

            #####
            # Run supports and metaweights to populate lor_cache
            lor_cache = empty_lor_cache

            # iterate over supports and do SGD on LoRs
            with torch.enable_grad():
                for i in range(2):

                    if i > 0:
                        lorm = partially_apply_models(lor_models, lor_cache)
                    else:
                        lorm = lor_cache  # ie empty

                    out = model(inputs_embeds=inputs_embeds,  # note: input_embeds, not input_ids
                                attention_mask=attention_mask,
                                return_dict=True,
                                output_hidden_states=True,
                                uncausal_mask=uncausal_mask,
                                **lorm,
                                )

                    # calculate TRANSDUCTIVE loss, ie not autoregressive, ie don't offset logits/targets (and don't causal mask)
                    logits = out.logits[:, :S].contiguous().view(-1, vocab_size)  # note: not offset
                    target = support_labels[:, :S].view(-1)
                    loss = F.cross_entropy(logits, target)

                    # Metalearn
                    if i > 0:
                        key_ixs = []  # [(lor_key, layer_ix)]
                        params = []
                        # flatten params to be used in optim
                        for k in lor_cache.keys():
                            for lix in range(len(lor_cache[k])):
                                if lor_cache[k][lix] is not None:
                                    for tuple_ix in range(2):
                                        key_ixs.append((k, lix, tuple_ix))
                                        params.append(lor_cache[k][lix][tuple_ix].requires_grad_())

                        grads = torch.autograd.grad(loss, params, create_graph=True)

                        new_params = sgd(params, grads, lr=inner_lr)

                        lor_cache = empty_lors(model.config.num_hidden_layers)
                        for (k, lix, tuple_ix), p in zip(key_ixs, new_params):
                            if lor_cache[k][lix] is None:
                                lor_cache[k][lix] = [None, None]  # these will both be filled in
                            lor_cache[k][lix][tuple_ix] = p

                    # replace current lors with new lors by updating over a fresh empty cache, instead of appending successively higher ranks
                    lor_cache = update_lors(lor_models, empty_lor_cache, lor_ixs_per_layer, out.hidden_states, num_layers)



            # #####
            # # Run Challenge with lor_cache, but no attention to original inputs (answers must live in weights ie lor_cache)
            # breakpoint()
            # lorm = partially_apply_models(lor_models, lor_cache)

            # challenge_out = model(
            #     input_ids=challenge_input_ids,
            #     attention_mask=challenge_attention_mask,
            #     **lorm,
            # )


            # m = challenge_loss_mask[..., 1:].contiguous().view(-1)

            # vocab_size = challenge_out.logits.shape[2]
            # logits = challenge_out.logits[:, :-1].contiguous().view(-1, vocab_size)  # [B, S-1, D] -> [B * (S-1), D]
            # target = challenge_input_ids[:, 1:].contiguous().view(-1)  # [B, S-1] -> [B * (S-1)]
            # loss = F.cross_entropy(logits[m], target[m])

            if torch.isnan(loss):
                print('NaN encountered:')
                # print(f'  all loss was masked out?: {sum([x.to(dtype=torch.long).sum() for x in loss_masks]) == 0}')
                # print(f'  nan in out_logits?: {torch.cat(out_logits, dim=1).isnan().sum() > 0}')
                # print('  ragged batch sizes? (double check by hand if necessary)')
                breakpoint()

            if train:
                optimizer.zero_grad()
                loss.backward()

                MAX_GRAD_NORM = 1.0
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(lor_models.parameters(), max_norm=MAX_GRAD_NORM)

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
        return avg_loss, out, challenge_out, lor_cache
    else:
        return avg_loss




##################################################
# Go

global_epoch = 0

# data
train_alphabets = ["Latin", "Greek"]
test_alphabets = ["Mongolian"]
img_size = 28
n_way = 5  # N-way classification
k_shot = 1  # k-shot learning
q_query = 1  # query examples per class
num_tasks = 100  # number of tasks per epoch

# training
num_epochs = 100
batch_size = 32
lr = 1e-3
wd = 1e-2

inner_lr = 1e-2


##########
# LoR Models

dim = model.model.embed_tokens.weight.shape[1]
k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
ff_dim = model.config.intermediate_size

# Note: there must be at least a None per each QKVOGUD block per layer
lor_models = nn.ModuleDict(
    {

        #####
        # Projection
        'lor_proj': nn.ModuleList([None] * num_layers),

        #####
        # Norms

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

lor_models['lor_proj'][LOR_LAYER] = LORProject()

if WHICH_LOR == 1:
    lor_models['lor_qs'][LOR_LAYER] = LORNorm(dim, dim)
    lor_models['lor_ks'][LOR_LAYER] = LORNorm(dim, k_dim)
    lor_models['lor_vs'][LOR_LAYER] = LORNorm(dim, v_dim)
    lor_models['lor_os'][LOR_LAYER] = LORNorm(dim, dim)

    lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim)
elif WHICH_LOR == 2:
    lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim)
    lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim)
lor_models = lor_models.to(DEVICE, dtype=model.dtype)


print_model_info(lor_models)
add_hooks(lor_models)


##########
# Image Projection

img_proj = nn.Linear(img_size ** 2, dim)
img_proj.to(DEVICE)

##########
# Dataset

train_dl, test_dl = omniglot_n_way_k_shot(
    train_alphabets,
    test_alphabets,
    n_way,
    k_shot,
    q_query,
    num_tasks,
    img_size,
    batch_size,
)

##########
# Optimizer

parameters = [{
    'params': model.parameters(),
    'lr': 0.0,
    'wd': 0.0
},
    {
    'params': img_proj.parameters(),
    'lr': lr,
    'wd': wd
},
    {
    'params': lor_models.parameters(),
    'lr': lr,
    'wd': wd
}
]

optimizer = optim.AdamW(parameters)

####################

train_losses = []
test_losses = []
best_loss = float('inf')

model.train()
for epoch in range(num_epochs):
    global_epoch += 1
    train_loss = run_epoch(model, img_proj, lor_models, inner_lr, train_dl, optimizer, DEVICE, train=True)
    train_losses.append(train_loss)
    writer.add_scalars('loss', {'train': train_loss}, global_epoch)

    if epoch % 1 == 0:
        test_loss = run_epoch(model, img_proj, lor_models, inner_lr, test_dl, optimizer, DEVICE, train=False)
        # test_loss = 0
        test_losses.append(test_loss)
        writer.add_scalars('loss', {'test': test_loss}, global_epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")



if num_epochs > 0:
    l = len(test_losses)
    epochs = list(range(l))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses[:l], label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='red')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
