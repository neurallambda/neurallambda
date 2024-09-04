'''

Add Binding Attention to LLMs

1. Run a dataset across the original transformer. Collect QK states at each layer to build a new dataset out of.
2. For each LSTM you'll insert, train it separately to work as an id function on the QK states
3. insert the "identity-initialized" LSTMs into the model
4. continue training


DEBUG:
- Cache issues, how to use with LSTM?
  - cache is RoPE specific, takes cos and sin
- Is attention mask getting set correctly?
- Double check loss, why is it integers?
- Double check that LSTMs train right (why do layer idxs > 0 train so fast?!)
- Confirm trained LSTM has low loss
- Try a single LSTM swapped in
- Train proj without/before RoPE

TRY:
- use RoPE, don't skip it?


TODO:
I need to get a backpack of hidden/cell state into FullLSTM
To do that I need to use a custom Cache
Theres a complexity around legacy (tuples) vs new (DynamicCache)



NOTE:

pre and post kv_cache shape of keys during inference:

    # encoder pass
    PRE : torch.Size([B, N_KV_HEAD, 52, D])
    POST: torch.Size([B, N_KV_HEAD, 52, D])

    # gen token 2
    PRE : torch.Size([B, N_KV_HEAD, 1, D])
    POST: torch.Size([B, N_KV_HEAD, 53, D])

    # gen token 3
    PRE : torch.Size([B, N_KV_HEAD, 1, D])
    POST: torch.Size([B, N_KV_HEAD, 54, D])

'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
import transformers.models.qwen2.modeling_qwen2 as Q
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from typing import Optional, Tuple, Union, List, Dict, Any
import warnings
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt

import t11_llm_binding_attention_data as Data
import t11_llm_binding_attention_log_qk as Log

import os
import json
import random

SEED = 152
torch.manual_seed(152)
random.seed(SEED)

DEVICE = 'cuda'
BATCH_SIZE = 32

TRAIN_LSTMS = False
USE_TRAINED_LSTM = True
LSTM_DIR = 't11_llm_binding_attention_lstms'
os.makedirs(LSTM_DIR, exist_ok=True)

# model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")


##################################################
# Original

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)  # time * different inv_freq per head

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def shift_halves(x):
    """Permute x in half for use in RoPE calculation"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (shift_halves(q) * sin)
    k_embed = (k * cos) + (shift_halves(k) * sin)
    return q_embed, k_embed


##################################################
#

class FullLSTMQK(nn.Module):
    '''Transductive LLM, records and stacks each output, so, same sequence length as input.

    This QK variant is intended to match the input-output pattern of
    hidden_state -> QK in grouped query attention. '''
    def __init__(self, input_dim, q_dim, k_dim, dropout_p):
        super(FullLSTMQK, self).__init__()
        self.input_dim = input_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.dropout_p = dropout_p

        # inputs-hidden_states: torch.Tensor,  # [B, S, D]
        # query_states: torch.Tensor,  # [B, NUM_HEADS, S, D // NUM_HEADS]
        # key_states: torch.Tensor,  # [B, NUM_KV_HEADS, S, D // NUM_HEADS] (Group Query Attention)

        # self.norm = nn.LayerNorm(input_dim)
        # in_h = 64
        # self.in_h = nn.Linear(input_dim, input_dim)
        # self.in_h = nn.Sequential(nn.Linear(input_dim, in_h), nn.ReLU(), nn.Linear(in_h, input_dim))

        # One Cell
        self.lstm_q = nn.LSTMCell(input_dim, input_dim)
        self.lstm_k = nn.LSTMCell(input_dim, input_dim)

        # # Two Cell
        # self.lstm_q1 = nn.LSTMCell(input_dim, input_dim)
        # self.lstm_q2 = nn.LSTMCell(input_dim, input_dim)
        # self.lstm_k1 = nn.LSTMCell(input_dim, input_dim)
        # self.lstm_k2 = nn.LSTMCell(input_dim, input_dim)

        # # Linear
        # self.out_q = nn.Linear(input_dim, input_dim)
        # self.out_k = nn.Linear(input_dim, input_dim // num_q_heads * num_k_heads)

        # More Complex
        out_h = 64
        self.out_q = nn.Sequential(nn.Linear(input_dim, out_h), nn.ReLU(), nn.Linear(out_h, q_dim))
        self.out_k = nn.Sequential(nn.Linear(input_dim, out_h), nn.ReLU(), nn.Linear(out_h, k_dim))

        self.dropout = nn.Dropout(dropout_p)


    def forward(self, x, backpack):
        B, S, _ = x.size()
        dtype = x.dtype

        # (Pdb) backpack['hq'].shape
        # torch.Size([1, 1536])

        # Assertions
        backpack_is_occupied = backpack['hq'] is not None
        if backpack_is_occupied:
            assert (
                backpack['hq'] is not None and
                backpack['cq'] is not None and
                backpack['hk'] is not None and
                backpack['ck'] is not None
            )
        else:
            assert (
                backpack['hq'] is None and
                backpack['cq'] is None and
                backpack['hk'] is None and
                backpack['ck'] is None
            )

        # Init Hidden State
        if backpack_is_occupied:
            hq = backpack['hq']
            cq = backpack['cq']
            hk = backpack['hk']
            ck = backpack['ck']
        else:
            hq = torch.zeros(B, self.input_dim, dtype=dtype).to(x.device)
            cq = torch.zeros(B, self.input_dim, dtype=dtype).to(x.device)
            hk = torch.zeros(B, self.input_dim, dtype=dtype).to(x.device)
            ck = torch.zeros(B, self.input_dim, dtype=dtype).to(x.device)

        # Run LSTM
        out_q = []
        out_k = []

        # x = self.norm(x)
        # x = self.in_h(x)
        for t in range(S):

            # ONE CELL
            # Q
            hq, cq = self.lstm_q(x[:, t, :], (hq, cq))
            # hq = hq + x[:, t, :]  # residual
            out_q.append(hq)

            # K
            hk, ck = self.lstm_k(x[:, t, :], (hk, ck))
            # hk = hk + x[:, t, :]  # residual
            out_k.append(hk)

            # # TWO CELL
            # # Q
            # hq, cq = self.lstm_q1(x[:, t, :], (hq, cq))
            # hq, cq = self.lstm_q2(x[:, t, :], (hq, cq))
            # out_q.append(hq)

            # # K
            # hk, ck = self.lstm_k1(x[:, t, :], (hk, ck))
            # hk, ck = self.lstm_k2(x[:, t, :], (hk, ck))
            # out_k.append(hk)


        out_q = self.out_q(torch.stack(out_q, dim=1))
        out_k = self.out_k(torch.stack(out_k, dim=1))
        backpack = {
                'hq': hq,
                'cq': cq,
                'hk': hk,
                'ck': ck,
            }
        return (
            out_q,
            out_k,
            backpack
        )

# class FullLSTMQK(nn.Module):
#     ''' linear only version to debug some issues '''
#     def __init__(self, input_dim, q_dim, k_dim, dropout_p):
#         super(FullLSTMQK, self).__init__()
#         self.input_dim = input_dim
#         self.q_dim = q_dim
#         self.k_dim = k_dim
#         self.dropout_p = dropout_p

#         self.out_q = nn.Linear(input_dim, q_dim, bias=True)
#         self.out_k = nn.Linear(input_dim, k_dim, bias=True)

#         self.dropout = nn.Dropout(dropout_p)


#     def forward(self, x, backpack):
#         B, S, _ = x.size()
#         dtype = x.dtype

#         out_q = self.out_q(x)
#         out_k = self.out_k(x)
#         backpack = {}
#         return (
#             out_q,
#             out_k,
#             backpack
#         )



class MyAttention(nn.Module):
    """
    Copied from: Qwen2Attention

    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Q.Qwen2Config, lstm_config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            Q.logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.should_use_lstm = lstm_config != None

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Should we use the LSTM variant?
        if lstm_config is not None:
            self.qk_proj = FullLSTMQK(**lstm_config)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # pass value just for dtype/device

        if self.should_use_lstm:
            if len(past_key_value) <= self.layer_idx:
                backpack = {
                    'hq': None,
                    'cq': None,
                    'hk': None,
                    'ck': None
                }
            else:
                _, _, backpack = past_key_value[self.layer_idx]
            query_states, key_states, backpack = self.qk_proj(hidden_states, backpack)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            # query_states = query_states.view(B, self.num_q_heads, S, self.input_dim // self.num_q_heads)

            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # key_states = key_states.view(B, self.num_k_heads, S, self.input_dim // self.num_q_heads),  # //nq looks weird but is right

            # TODO: RoPE Cheat?
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            key_states = self.k_proj(hidden_states)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            backpack = None

        if past_key_value is not None:
            key_states, value_states, backpack = past_key_value.update(key_states, value_states, self.layer_idx, backpack=backpack)

        # repeat k/v heads if n_kv_heads < n_heads (Grouped Query Attn)
        key_states = Q.repeat_kv(key_states, self.num_key_value_groups)
        value_states = Q.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @classmethod
    def from_qwen2_attention(cls, qwen2_attention, randomize_lstm):
        with torch.no_grad():
            config = qwen2_attention.config
            layer_idx = qwen2_attention.layer_idx

            lstm_to_use = set(range(0, 28))
            # lstm_to_use = set(range(5, 28))
            # lstm_to_use = set(range(22, 28))
            # lstm_to_use = {}
            if USE_TRAINED_LSTM and layer_idx in lstm_to_use:

                ##########
                # Read trained LSTM from disk
                if not randomize_lstm:

                    print(f'ONLY USING LSTM idx in {lstm_to_use}')
                    lstm_path = f'{LSTM_DIR}/lstm_{layer_idx}.pth'
                    print(f'loading lstm: {lstm_path}')
                    with open(f'{LSTM_DIR}/lstm_{layer_idx}.json', 'r') as f:
                        lstm_config = json.load(f)
                    custom_attn = cls(config, lstm_config, layer_idx)
                    with open(lstm_path, 'rb') as f:
                        lstm_weight = torch.load(f)
                    custom_attn.qk_proj.load_state_dict(lstm_weight)

                ##########
                # Init random LSTM
                else:
                    print(f'USING RANDOMIZED LSTM idx in {lstm_to_use}')
                    input_dim = config.hidden_size
                    num_q_heads = config.num_attention_heads
                    num_k_heads = config.num_key_value_heads
                    lstm_config = {
                        'input_dim': input_dim,
                        'num_q_heads': num_q_heads,
                        'num_k_heads': num_k_heads,
                        'dropout_p': 0.0
                    }
                    custom_attn = cls(config, lstm_config, layer_idx)

                    # ##########
                    # # Extreme initialization

                    # # LSTM Q
                    # custom_attn.qk_proj.lstm_q.weight_ih[:] = torch.randn_like(custom_attn.qk_proj.lstm_q.weight_ih) * 10
                    # custom_attn.qk_proj.lstm_q.weight_hh[:] = torch.randn_like(custom_attn.qk_proj.lstm_q.weight_hh) * 10
                    # custom_attn.qk_proj.lstm_q.bias_ih[:] = torch.randn_like(custom_attn.qk_proj.lstm_q.bias_ih) * 10
                    # custom_attn.qk_proj.lstm_q.bias_hh[:] = torch.randn_like(custom_attn.qk_proj.lstm_q.bias_hh) * 10
                    # # LSTM K
                    # custom_attn.qk_proj.lstm_k.weight_ih[:] = torch.randn_like(custom_attn.qk_proj.lstm_k.weight_ih) * 10
                    # custom_attn.qk_proj.lstm_k.weight_hh[:] = torch.randn_like(custom_attn.qk_proj.lstm_k.weight_hh) * 10
                    # custom_attn.qk_proj.lstm_k.bias_ih[:] = torch.randn_like(custom_attn.qk_proj.lstm_k.bias_ih) * 10
                    # custom_attn.qk_proj.lstm_k.bias_hh[:] = torch.randn_like(custom_attn.qk_proj.lstm_k.bias_hh) * 10
                    # # MLP Out
                    # custom_attn.qk_proj.out_q[0].weight[:] = torch.randn_like(custom_attn.qk_proj.out_q[0].weight) * 10
                    # custom_attn.qk_proj.out_q[0].bias[:] = torch.randn_like(custom_attn.qk_proj.out_q[0].bias) * 10
                    # custom_attn.qk_proj.out_q[2].weight[:] = torch.randn_like(custom_attn.qk_proj.out_q[2].weight) * 10
                    # custom_attn.qk_proj.out_q[2].bias[:] = torch.randn_like(custom_attn.qk_proj.out_q[2].bias) * 10
                    # custom_attn.qk_proj.out_k[0].weight[:] = torch.randn_like(custom_attn.qk_proj.out_k[0].weight) * 10
                    # custom_attn.qk_proj.out_k[0].bias[:] = torch.randn_like(custom_attn.qk_proj.out_k[0].bias) * 10
                    # custom_attn.qk_proj.out_k[2].weight[:] = torch.randn_like(custom_attn.qk_proj.out_k[2].weight) * 10
                    # custom_attn.qk_proj.out_k[2].bias[:] = torch.randn_like(custom_attn.qk_proj.out_k[2].bias) * 10

            else:
                # use original QK, not LSTM
                lstm_config = None
                custom_attn = cls(config, lstm_config, layer_idx)

                custom_attn.q_proj.weight.data = qwen2_attention.q_proj.weight.data.clone()
                custom_attn.q_proj.bias.data = qwen2_attention.q_proj.bias.data.clone()

                custom_attn.k_proj.weight.data = qwen2_attention.k_proj.weight.data.clone()
                custom_attn.k_proj.bias.data = qwen2_attention.k_proj.bias.data.clone()

                print(f'NOT USING TRAINED LSTMs on {layer_idx=}')

            custom_attn.v_proj.weight.data = qwen2_attention.v_proj.weight.data.clone()
            custom_attn.v_proj.bias.data = qwen2_attention.v_proj.bias.data.clone()
            custom_attn.o_proj.weight.data = qwen2_attention.o_proj.weight.data.clone()

            # Ensure same device and dtype
            device = qwen2_attention.q_proj.weight.device
            dtype = qwen2_attention.q_proj.weight.dtype
            custom_attn = custom_attn.to(device=device, dtype=dtype)

            return custom_attn

    def state_dict(self, *args, **kwargs):
        # Customize state_dict if needed
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        # Customize load_state_dict if needed
        return super().load_state_dict(state_dict, strict)


############################################################
##################################################
# Add Custom Cache to Model. This is otherwise unchanged.

class MyCache(Cache):
    """
    Adapted from DynamicCache, adds a `backpack` where you can store anything!

    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.backpack = []  # store anything here per layer
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.backpack[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        backpack: List[Optional[Dict[str, Any]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.backpack.append(backpack)
            old_backpack_at_ix = None
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            old_backpack_at_ix = self.backpack[layer_idx]
            self.backpack[layer_idx] = backpack

        return self.key_cache[layer_idx], self.value_cache[layer_idx], old_backpack_at_ix

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        raise Exception('dont use to_legacy_cache')

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        raise Exception('dont use from_legacy_cache')


class Qwen2Model(Q.Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Q.Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Q.Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Q.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @Q.add_start_docstrings_to_model_forward(Q.QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Union[Cache, Optional[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Q.BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                Q.logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            # use_legacy_cache = not isinstance(past_key_values, Cache)
            # if use_legacy_cache:
            #     print('USING LEGACY CACHE')
            #     breakpoint()
            #     past_key_values = MyCache.from_legacy_cache(past_key_values)
            if past_key_values is None:
                past_key_values = MyCache()

            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = Q._prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = Q._prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # next_cache = None
        # if use_cache:
        #     next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        next_cache = next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return Q.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Qwen2ForCausalLM(Q.Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @Q.add_start_docstrings_to_model_forward(Q.QWEN2_INPUTS_DOCSTRING)
    @Q.replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=Q._CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past



####################################################################################################
####################################################################################################
####################################################################################################
# Train LSTMs, each individually

if TRAIN_LSTMS:
    ##################################################
    # Load Model
    try:
        already_got_traces
    except:

        cpu_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",
            _attn_implementation='eager',
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # trace_outs will contain traced QK values
        model, trace_outs = Log.replace_attention(cpu_model)
        model = model.to(DEVICE)

        # trace_outs[LAYER_IX][{'attention_mask', 'hidden_states', 'k', 'q'}][BATCH_IX] => tensor shape [BATCH_SIZE,
        # trace_outs[LAYER_IX]['k'][BATCH_IX] => tensor shape [BATCH_SIZE, 2, S, D]
        # trace_outs[LAYER_IX]['q'][BATCH_IX] => tensor shape [BATCH_SIZE, 12, S, D]
        # trace_outs[LAYER_IX]['attention_mask'][BATCH_IX] => tensor shape [BATCH_SIZE, S]
        # trace_outs[LAYER_IX]['hidden_states'][BATCH_IX] => tensor shape [BATCH_SIZE, S, 1536]

        # >>> type(trace_outs)
        # <class 'dict'>
        # >>> trace_outs.keys()
        # dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
        # >>> type(trace_outs[0])
        # <class 'dict'>
        # >>> trace_outs[0].keys()
        # dict_keys(['attention_mask', 'hidden_states', 'k', 'q'])
        # >>> type(trace_outs[0]['attention_mask'])
        # <class 'list'>
        # >>> len(trace_outs[0]['attention_mask'])
        # 1
        # >>> type(trace_outs[0]['k'])
        # <class 'list'>
        # >>> len(trace_outs[0]['k'])
        # 1
        # >>> trace_outs[0]['k'][0].shape
        # torch.Size([1, 2, 7, 128])
        # >>> trace_outs[0]['q'][0].shape
        # torch.Size([1, 12, 7, 128])
        # >>> trace_outs[0]['hidden_states'][0].shape
        # torch.Size([1, 7, 1536])

        # Hand test logging stuff
        if False:
            prompt = "Once upon a time there was a"
            model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')

            # out = model.generate(**model_inputs, max_new_tokens=30)
            # response = tokenizer.batch_decode(out, skip_special_tokens=False)[0]

            out = model(**model_inputs, return_dict=True)  # don't use generate, we don't want to continually append the full state for each tok
            response = tokenizer.batch_decode(out.logits.argmax(dim=2), skip_special_tokens=False)[0]
            print()
            print(response)
            print(f'{model.model.layers[0].self_attn.trace_out["hidden_states"][0].shape=}')
            print(f'{model.model.layers[0].self_attn.trace_out["q"][0].shape=}')
            print(f'{model.model.layers[0].self_attn.trace_out["k"][0].shape=}')


        ##########
        # Dataset

        print('LSTM Training: Loading Datasets')

        # chargoddard/winogrande-train-10k splits = {train, train_4k, train_2k, fewshot}
        dataset = load_dataset("chargoddard/winogrande-train-10k", split="train")  #
        processed_dataset = Data.prepare_dataset(dataset)

        # Split the dataset into train and validation sets
        # train_size = int(0.8 * len(processed_dataset))
        # val_size = len(processed_dataset) - train_size

        train_size = 2000
        val_size = 200
        remainder_size = len(processed_dataset) - train_size - val_size
        train_dataset, val_dataset, remainder_dataset = random_split(processed_dataset, [train_size, val_size, remainder_size])
        train_dataloader = Data.create_dataloader(train_dataset, tokenizer, batch_size=BATCH_SIZE)


        ##########
        # Collect QK Traces

        print('LSTM Training: Collect QK Traces')
        # populate each `self_attn.trace_out`
        Log.generate_qk_dataset(model, train_dataloader)

        if False:
            a = model.model.layers[0].self_attn.trace_out["attention_mask"]
            hidden_states = model.model.layers[0].self_attn.trace_out["hidden_states"]
            q = model.model.layers[0].self_attn.trace_out["q"]
            k = model.model.layers[0].self_attn.trace_out["k"]

            print(f'{a[0].shape=}')
            print(f'{hidden_states[0].shape=}')
            print(f'{q[0].shape=}')
            print(f'{k[0].shape=}')

        # free up memory for training LSTMs

        del model
        torch.cuda.empty_cache()

        print('LSTM Training: Done collecting traces')
        already_got_traces = True

    # START_BLOCK_2


def train_lstm(trace_outs, num_epochs, orig_attn):
    percent_train = 0.8
    # lr = 5e-4
    # lr = 2e-3
    # lr = 1e-2
    lr = 2e-2
    wd = 0.0  # 5e-3
    dropout_p = 0.0

    input_dim = trace_outs['hidden_states'][0].shape[2]
    q_dim = trace_outs['q'][0].shape[2]
    k_dim = trace_outs['k'][0].shape[2]
    dtype = trace_outs['hidden_states'][0].dtype

    lstm_config = {
        'input_dim': input_dim,
        'q_dim': q_dim,
        'k_dim': k_dim,
        'dropout_p': dropout_p
    }
    lstm = FullLSTMQK(**lstm_config)

    # debugging with non-lstm linear layers
    if False:
        print('Copying Original Attn')
        with torch.no_grad():
            lstm.out_q.weight[:] = orig_attn.q_proj.weight
            lstm.out_q.bias[:] = orig_attn.q_proj.bias
            lstm.out_k.weight[:] = orig_attn.k_proj.weight
            lstm.out_k.bias[:] = orig_attn.k_proj.bias

    if False:
        print('Custom Initialization')
        with torch.no_grad():
            lstm.out_q.weight[:] = torch.randn_like(lstm.out_q.weight) * 0.25
            lstm.out_q.bias[:] = torch.randn_like(lstm.out_q.bias) * 0.25
            lstm.out_k.weight[:] = torch.randn_like(lstm.out_k.weight) * 0.25
            lstm.out_k.bias[:] = torch.randn_like(lstm.out_k.bias) * 0.25

    lstm = lstm.to(DEVICE, dtype=dtype)

    empty_backpack = {
        'hq': None,
        'cq': None,
        'hk': None,
        'ck': None
    }

    file_name = f'{LSTM_DIR}/lstm_{layer_idx}'
    config_path = f'{file_name}.json'
    weight_path = f'{file_name}.pth'

    optimizer = AdamW(lstm.parameters(), lr=lr, weight_decay=wd)

    assert (
        len(trace_outs['attention_mask']) ==
        len(trace_outs['hidden_states']) ==
        len(trace_outs['q']) ==
        len(trace_outs['k'])
    )

    # train on a portion of batches
    num_batches = len(trace_outs['q'])
    train_ix = int(num_batches * percent_train)

    # Training loop
    val_losses = []
    train_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(num_epochs):
        lstm.train()

        train_data = zip(trace_outs['attention_mask'][:train_ix],
                         trace_outs['hidden_states'][:train_ix],
                         trace_outs['q'][:train_ix],
                         trace_outs['k'][:train_ix])
        val_data   = zip(trace_outs['attention_mask'][train_ix:],
                         trace_outs['hidden_states'][train_ix:],
                         trace_outs['q'][train_ix:],
                         trace_outs['k'][train_ix:])

        # TRAINING
        batch_losses = []
        for a, h, q, k in train_data:
            # (Pdb) print(a.shape) <bool>
            # torch.Size([32, 73])
            # (Pdb) print(h.shape)
            # torch.Size([32, 73, 1536])
            # (Pdb) print(q.shape)
            # torch.Size([32, 12, 73, 128])
            # (Pdb) print(k.shape)
            # torch.Size([32, 2, 73, 128])

            B, S, _ = h.shape
            h = h.cuda()
            q = q.cuda()  # [32, 12, 73, 128]
            k = k.cuda()  # [32, 2, 73, 128]

            out_q, out_k, _ = lstm(h, empty_backpack)
            # out_q.shape [32, 12, 73, 128]
            # out_k.shape [32, 2, 73, 128]


            # ##########
            # # KL Divergence of attention
            # Warn: Theres an issue with how this is implemented
            # num_heads = 12
            # head_dim = input_dim // num_heads
            # num_key_value_heads = 2
            # num_key_value_groups = num_heads // num_key_value_heads

            # q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
            # k = k.view(B, S, num_key_value_heads, head_dim).transpose(1, 2)

            # out_q = out_q.view(B, S, num_heads, head_dim).transpose(1, 2)
            # out_k = out_k.view(B, S, num_key_value_heads, head_dim).transpose(1, 2)

            # # repeat k/v heads if n_kv_heads < n_heads (Grouped Query Attn)
            # k = Q.repeat_kv(k, num_key_value_groups)
            # out_k = Q.repeat_kv(out_k, num_key_value_groups)

            # # target_scores = torch.einsum('', q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            # target_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            # target_attn = target_scores.softmax(dim=-1)

            # output_scores = torch.matmul(out_q, out_k.transpose(2, 3)) / math.sqrt(head_dim)
            # output_attn = output_scores.log_softmax(dim=-1)

            # loss = F.kl_div(output_attn, target_attn, log_target=False, reduction='batchmean')

            ##########
            # no attn mask
            loss = (
                F.mse_loss(q, out_q) +
                F.mse_loss(k, out_k)
            ) / B / S

            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(train_loss)

        # VALIDATION
        lstm.eval()
        batch_val_losses = []
        with torch.no_grad():
            for a, h, q, k in val_data:
                B, S, _ = h.shape
                # aq = a.cuda().unsqueeze(1).unsqueeze(3).expand(-1, q.size(1), -1, q.size(3))  # applied on each head
                # ak = a.cuda().unsqueeze(1).unsqueeze(3).expand(-1, k.size(1), -1, k.size(3))  # applied on each head
                h = h.cuda()
                q = q.cuda()
                k = k.cuda()
                out_q, out_k, _ = lstm(h, empty_backpack)

                # no attn mask
                val_loss = (
                    F.mse_loss(q, out_q) +
                    F.mse_loss(k, out_k)
                ) / B / S

                batch_val_losses.append(val_loss.item())

                current_val_loss = sum(batch_val_losses) / len(batch_val_losses)
                val_losses.append(current_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} completed. train: {train_losses[-1]:.5f}, val: {val_losses[-1]:.5f}")

        # track best model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu() for k, v in lstm.state_dict().items()}

    # Save the best model configuration
    with open(config_path, 'w') as f:
        json.dump(lstm_config, f)

    # Save the best model weights
    torch.save(best_model_state, weight_path)
    print(f"Training completed. Best model was from epoch {best_epoch + 1} with validation loss: {best_val_loss:.3f}, saved to {weight_path}")

    if False:
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.show()

##################################################
# Train LSTMs

if TRAIN_LSTMS:

    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu",
        _attn_implementation='eager',
    )

    all_layers = trace_outs.keys()
    for layer_idx in tqdm(all_layers):
        if layer_idx <= 1:
            num_epochs = 50  # early layers appear to need more
        else:
            num_epochs = 20
        orig_attn = model.model.layers[layer_idx].self_attn
        train_lstm(trace_outs[layer_idx], num_epochs, orig_attn)

    BRK_TRAIN


# lin = nn.Linear(1536, 1536)
# plt.hist(lin.weight.flatten().detach().numpy())
# plt.show()

# idx = 3
# plt.hist(model.model.layers[idx].self_attn.q_proj.weight.flatten().float().detach().numpy(), bins=100)
# plt.show()



# END_BLOCK_2


# Visualize trace_outs

# layer_idx = 1
# batch = 0
# x = trace_outs[layer_idx]['hidden_states'][batch]
# plt.imshow(x[:,0].float().numpy())
# # plt.hist(x.flatten().float().numpy(), bins=300)
# plt.show()

# fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# fig.suptitle("Histograms of Hidden States for 16 Layers", fontsize=16)
# for layer_idx in range(16):
#     row = layer_idx // 4
#     col = layer_idx % 4
#     batch = 0
#     batch_item = 1
#     x = trace_outs[layer_idx]['hidden_states'][batch][batch_item]
#     axes[row, col].hist(x[17:].flatten().float().numpy(), bins=100, edgecolor='black')
#     # axes[row, col].imshow(x[:, 200:400].float().numpy())
#     axes[row, col].set_title(f"Layer {layer_idx + 1}")
#     axes[row, col].set_ylabel("Frequency")
# plt.tight_layout()
# plt.show()


# a = torch.tensor([[0.1, 1, 10.0]]).softmax(dim=-1).log()
# b = torch.tensor([[0.1, 1, 10.0]]).softmax(dim=-1).log()
# print(F.kl_div(a, b, log_target=True))

####################################################################################################
####################################################################################################
####################################################################################################
#

# START_BLOCK_3
cpu_model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu",
    _attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

already_loaded = True

model = cpu_model.to('cuda')

for layer_idx, layer in enumerate(model.model.layers):
    custom_attn = MyAttention.from_qwen2_attention(layer.self_attn, randomize_lstm=False)
    layer.self_attn = custom_attn

# Hand test model
if False:
    # prompt = """Once upon a"""

#     prompt = f"""Below is a fill-in-the-blank problem. Choose the correct option to complete the sentence.

# Problem: I ordered a coffee and muffin, but couldn't eat the _.

# Options:
# 1. coffee
# 2. muffin

# The correct answer is:"""

    prompt = '''Problem: Angela ran away from the fire, while Mary rushed towards it to put it out, since _ was a coward.

Options:
1. Angela
2. Mary

The correct answer is: '''

    model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print()
    print(prompt + response)

    BRK

# END_BLOCK_3

def calculate_accuracy(model, dataloader, tokenizer, device):
    model.eval()
    total_correct = 0
    total_answers = 0

    with torch.no_grad():
        for batch in dataloader:
            prompt_ids = batch['prompt_ids'].to(device)
            prompt_attention_mask = batch['prompt_attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            batch_size = prompt_ids.shape[0]
            total_answers += batch_size

            outputs = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=answer_ids.shape[1],  # Generate up to the max answer length
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Remove the prompt from the generated output
            predicted_answer_ids = outputs[:, prompt_ids.shape[1]:]

            # Compare predicted answer with actual answer
            for pred, true in zip(predicted_answer_ids, answer_ids):
                # Remove padding and end-of-sequence tokens
                pred = pred[pred != tokenizer.pad_token_id]
                pred = pred[:torch.where(pred == tokenizer.eos_token_id)[0][0]] if tokenizer.eos_token_id in pred else pred

                true = true[true != tokenizer.pad_token_id]
                true = true[:torch.where(true == tokenizer.eos_token_id)[0][0]] if tokenizer.eos_token_id in true else true

                if torch.equal(pred, true):
                    total_correct += 1

    accuracy = total_correct / total_answers
    return accuracy


##########
# Train

# chargoddard/winogrande-train-10k splits = {train, train_4k, train_2k, fewshot}
dataset = load_dataset("chargoddard/winogrande-train-10k", split="train")  #
processed_dataset = Data.prepare_dataset(dataset)

# Split the dataset into train and validation sets
# train_size = int(0.8 * len(processed_dataset))
# val_size = len(processed_dataset) - train_size

num_epochs = 5
train_size = 1000
val_size = 100
LR = 1e-4
WD = 0.0

remainder_size = len(processed_dataset) - train_size - val_size
train_dataset, val_dataset, remainder_dataset = random_split(processed_dataset, [train_size, val_size, remainder_size])

# Create dataloaders for train and validation sets
train_dataloader = Data.create_dataloader(train_dataset, tokenizer, batch_size=BATCH_SIZE)
val_dataloader = Data.create_dataloader(val_dataset, tokenizer, batch_size=BATCH_SIZE)
remainder_dataloader = Data.create_dataloader(remainder_dataset, tokenizer, batch_size=BATCH_SIZE)

# debug_dataloader(train_dataloader, tokenizer, 3)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


###########
# Params

print('training ALL params')
params = model.parameters()

# print('training LSTM params only')
# params = [x[1] for x in model.named_parameters() if 'lstm' in x[0]]


##########
# Go

# Set up optimizer and learning rate scheduler
optimizer = AdamW(params, lr=LR, weight_decay=WD)

num_training_steps = num_epochs * len(train_dataloader)

# Training loop
val_losses = []
train_losses = []

for epoch in range(num_epochs):

    # TRAINING
    model.train()
    batch_losses = []
    for batch in tqdm(train_dataloader):
        prompt_ids = batch['prompt_ids'].to(device)
        prompt_attention_mask = batch['prompt_attention_mask'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        # answer_attention_mask = batch['answer_attention_mask'].to(device)
        answer_attention_mask = torch.ones_like(answer_ids)  # NOTE: avoid possible is_padding_right warning
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, answer_attention_mask], dim=1)

        labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids], dim=1)  # tokens outside vocab are ignored in loss, ignore prompt ids
        # labels = input_ids

        TODO: not actually using labels, dbl check
        outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits

        loss = outputs.loss

        # ##########
        # # Loss
        # #   custom calculation that takes into account masking

        # answer_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(answer_ids)], dim=1)
        # loss_mask = (answer_mask * attention_mask).bool()

        # # Shift so that tokens < n predict n
        # shift_logits = logits[loss_mask][..., :-1, :].contiguous()
        # shift_labels = input_ids[loss_mask][..., 1:].contiguous()
        # # flatten
        # shift_logits = shift_logits.view(-1, model.config.vocab_size)
        # shift_labels = shift_labels.view(-1)
        # shift_labels = shift_labels.to(shift_logits.device)
        # loss = F.cross_entropy(shift_logits, shift_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.item())
        print(f'step loss: {loss.item():.3f}')


    train_losses.append(sum(batch_losses) / len(batch_losses))

    # VALIDATION
    model.eval()
    with torch.no_grad():
        batch_val_losses = []
        for batch in val_dataloader:
            prompt_ids = batch['prompt_ids'].to(device)
            prompt_attention_mask = batch['prompt_attention_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)
            # answer_attention_mask = batch['answer_attention_mask'].to(device)
            answer_attention_mask = torch.ones_like(answer_ids)  # NOTE: avoid possible is_padding_right warning
            input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            attention_mask = torch.cat([prompt_attention_mask, answer_attention_mask], dim=1)

            # labels = input_ids
            labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids], dim=1)  # tokens outside vocab are ignored in loss, ignore prompt ids
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits
            loss = outputs.loss

            # outputs = model(input_ids, labels=input_ids, return_dict=True)
            # logits = outputs.logits
            # loss = outputs.loss

            # ##########
            # # Loss
            # #   custom calculation that takes into account masking
            # answer_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(answer_ids)], dim=1)
            # loss_mask = (answer_mask * attention_mask).bool()

            # # Shift so that tokens < n predict n
            # shift_logits = logits[loss_mask][..., :-1, :].contiguous()
            # shift_labels = input_ids[loss_mask][..., 1:].contiguous()
            # # flatten
            # shift_logits = shift_logits.view(-1, model.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = F.cross_entropy(shift_logits, shift_labels)

            batch_val_losses.append(loss)
        val_losses.append(sum(batch_val_losses) / len(batch_val_losses))

    print(f"Epoch {epoch+1}/{num_epochs} completed. Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")

# # Save the fine-tuned model
# model.save_pretrained("./t11_llm_binding_attention")
# tokenizer.save_pretrained("./t11_llm_binding_attention")


##################################################

# START_BLOCK_1

eval_dataloader = train_dataloader
# eval_dataloader = val_dataloader

print("Calculating accuracy on validation set...")
val_accuracy = calculate_accuracy(model, eval_dataloader, tokenizer, device)
print(f"Validation Accuracy: {val_accuracy:.4f}")

num_samples = 3
sample_count = 0
with torch.no_grad():
    for batch in eval_dataloader:
        if sample_count >= num_samples:
            break
        print('=' * 50)
        prompt_ids = batch['prompt_ids'].to(device)
        prompt_attention_mask = batch['prompt_attention_mask'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        answer_attention_mask = batch['answer_attention_mask'].to(device)

        batch_size = prompt_ids.shape[0]

        generated_ids = model.generate(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=answer_ids.shape[1],  # Generate up to the max answer length
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            print(f"Sample {sample_count + 1}:")

            # prompt
            prompt = tokenizer.decode(prompt_ids[i][prompt_ids[i] != tokenizer.pad_token_id], skip_special_tokens=True)
            print(f"Prompt: {prompt}")

            # expected answer
            expected_answer = tokenizer.decode(answer_ids[i], skip_special_tokens=False)  # [answer_ids[i] != tokenizer.pad_token_id]
            print(f"Expected: {expected_answer}")

            # generated answer
            generated_answer = tokenizer.decode(generated_ids[i][prompt_ids.shape[1]:], skip_special_tokens=False)  # slice off orig prompt ids
            print(f"Generated: {generated_answer}")

            print('-' * 50)
            sample_count += 1

# END_BLOCK_1

# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')

# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Epochs')

# plt.grid(True)
# plt.legend()
# plt.show()


# for batch in remainder_dataloader:
#     print('----------')
#     prompt_ids = batch['prompt_ids'].to(device)
#     prompt_attention_mask = batch['prompt_attention_mask'].to(device)
#     answer_ids = batch['answer_ids'].to(device)
#     answer_attention_mask = batch['answer_attention_mask'].to(device)

#     generated_ids = model.generate(
#         prompt_ids,
#         attention_mask=prompt_attention_mask,
#         max_new_tokens=5,
#     )
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
#     print(response)


# debug_dataloader(dataloader, tokenizer, 3)


# # $$$$$$$$$$
# # Debug Data + Attention Mask
# p  = batch['prompt_ids']
# a  = batch['answer_ids']
# pm = batch['prompt_attention_mask']
# am = batch['answer_attention_mask']
# for i in range(3):
#     print('----------')
#     tok = torch.cat([p[i], a[i]], dim=0)
#     print(tokenizer.decode(tok, skip_special_tokens=False))
#     attention_mask = torch.cat([pm[i], am[i]], dim=0)
#     print(attention_mask)
