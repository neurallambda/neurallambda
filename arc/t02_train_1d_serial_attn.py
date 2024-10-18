'''

Serial Transformer Stuff

PROVENANCE:
- experiment/t11_llm_binding_attention

'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
import transformers.models.qwen2.modeling_qwen2 as Q
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
import transformers.modeling_attn_mask_utils as MaskUtil

from typing import Optional, Tuple, Union, List, Dict, Any
import warnings
import math

import os
import json
import random
import warnings


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


class SerialAttention(nn.Module):
    """
    Copied from: Qwen2Attention

    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Q.Qwen2Config, is_causal: bool, lstm_config, layer_idx: Optional[int] = None):
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
        self.is_causal = is_causal
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
        init_backpack:Dict[str, Any] = None,  # holds initialization for LSTM
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
            # LSTM Mode

            if len(past_key_value) <= self.layer_idx:  # is past_key_value[layer_idx] in cache yet?
                if init_backpack is not None:
                    backpack = init_backpack[self.layer_idx]
                else:
                    backpack = {
                        'hq': None,
                        'cq': None,
                        'hk': None,
                        'ck': None
                    }
            else:
                _, _, backpack = past_key_value[self.layer_idx]
            query_states, key_states, backpack = self.qk_proj(hidden_states, backpack)  # run LSTM
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            # query_states = query_states.view(B, self.num_q_heads, S, self.input_dim // self.num_q_heads)

            # breakpoint()
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # key_states = key_states.view(B, self.num_k_heads, S, self.input_dim // self.num_q_heads)


            # TODO: use RoPE Cheat?

            # warnings.warn('RoPE cheat ON')
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            warnings.warn('RoPE cheat OFF')
        else:
            # Non-LSTM Mode
            query_states = self.q_proj(hidden_states)  # [B,S,D]
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
    def from_qwen2_attention(cls, qwen2_attention, randomize_lstm, use_trained_lstm=True):
        with torch.no_grad():
            config = qwen2_attention.config
            layer_idx = qwen2_attention.layer_idx

            lstm_to_use = set(range(0, 28))
            # lstm_to_use = set(range(5, 28))
            # lstm_to_use = set(range(22, 28))
            # lstm_to_use = {}
            if use_trained_lstm and layer_idx in lstm_to_use:

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


##################################################
# Decoder Layer: Redefine to use SerialAttention, and remove causal masking

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Q.Qwen2Config, is_causal: bool, lstm_config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            Q.logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = SerialAttention(config, is_causal, lstm_config, layer_idx)

        self.mlp = Q.Qwen2MLP(config)
        self.input_layernorm = Q.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Q.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        init_backpack=None,  # list of per-layer params sent to LSTMs
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            init_backpack=init_backpack,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



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
            breakpoint()
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

        # Update the cache and backpack
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

    def __init__(self, config: Q.Qwen2Config, causal_decoders: bool, lstm_config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.causal_decoders = causal_decoders

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, causal_decoders, lstm_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
        init_backpack=None,  # list of per-layer params sent to LSTMs
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

        # if self._attn_implementation == "flash_attention_2":
        #     # 2d mask is passed through the layers
        #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # elif self._attn_implementation == "sdpa" and not output_attentions:
        #     # output_attentions=True can not be supported when using SDPA, and we fall back on
        #     # the manual implementation that requires a 4D causal mask in all cases.
        #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_key_values_length,
        #         sliding_window=self.config.sliding_window,
        #     )
        # else:
        if True:
            # 4d mask is passed through the layers
            if not self.causal_decoders:
                warnings.warn('Decoder, but using is_causal=False. This may be ok in transducer-type settings, but is probably wrong in auto-regressive settings.')

            if self.causal_decoders:

                attention_mask = Q._prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.config.sliding_window,
                    # is_causal=self.causal_decoders,
                )

            else:
                attention_mask = MaskUtil._prepare_4d_attention_mask(
                    attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_ids.shape[-1]
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
                raise NotImplementedError('not sure if init_backpack works with gradient checkpointing')
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    init_backpack,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    init_backpack = init_backpack, # list of per-layer params sent to LSTMs
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

    def __init__(self, config, causal_decoders:bool, lstm_config):
        super().__init__(config)
        self.model = Qwen2Model(config, causal_decoders, lstm_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.causal_decoders = causal_decoders

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
        init_backpack = None, # list of per-layer params sent to LSTMs
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
            init_backpack = init_backpack, # list of per-layer params sent to LSTMs
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
        raise NotImplementedError('model.generate does not work yet, see `prepare_inputs_for_generation` for details.')
        # The cache won't work yet with the new LSTM
        # addition. SerialAttention.forward accepts `init_backpack` which I
        # don't think will play nicely with kv-cached inference yet. The
        # `init_backpack`, if set, should only be set once at inference start,
        # and this needs to be ensured during model.generate too.


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
