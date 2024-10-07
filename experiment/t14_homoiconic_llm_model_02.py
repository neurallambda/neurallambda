'''

Build on previous version to allow LoR'd layers to have more control: instead of adding LoR outputs to regular outputs, allow LoR layer to choose.

Rewrite the model class to allow the `forward` function to accept online-generated Lora params. These get applied throughout all linear layers (QKVO + GUD)

Adapted from transformers.models.qwen2.modeling_qwen2


API:

# low rank attention params, list of per layer params (or None)
lor_qs
lor_ks
lor_vs
lor_os

# low rank mlp params, list of per layer params (or None)
lor_us
lor_gs
lor_ds


##########
# Example.
#   lor_qs is a list, one row per transformer layer. An item can be None, or a tuple of low rank projection matrices.

lor_qs = [
  None,
  None,
  ([B, D, 1], [B, 1, D]),
  ([B, D, 1], [B, 1, D]),
  None,
  ...
]


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

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import time
from neurallambda.lab.common import print_model_info

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state, lor_u=None, lor_g=None, lor_d=None):
        '''
        The original does this:

          out = self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

        The low rank version applies lor_g/lor_u/lor_d to that calculation.

        Args:
          lor_u: ([batch, hidden_size, rank], [batch, rank, hidden_size^])
            low rank matrices for up_proj

          lor_g: ([batch, hidden_size, rank], [batch, rank, hidden_size^])
            low rank matrices for gate_proj

          lor_d: ([batch, hidden_size^, rank], [batch, rank, hidden_size])
            low rank matrices for down_proj

        NOTE: the `hidden_size^` dimensions will be interpolated to `intermediate_size`

        '''

        # low rank gate_proj
        g = self.gate_proj(hidden_state)
        if lor_g is not None:
            g = lor_g(g, hidden_state)

        # low rank up_proj
        u = self.up_proj(hidden_state)
        if lor_u is not None:
            u = lor_u(u, hidden_state)

        # low rank down_proj
        d_in = self.act_fn(g) * u
        out = self.down_proj(d_in)
        if lor_d is not None:
            out = lor_d(out, d_in)

        return out


# @@@@@@@@@@@@@@@@@@@@

def test_qwen2mlp_low_rank_equivalence(hidden_size=1536, intermediate_size=8960, batch_size=2, seq_len=10, rank=8):
    class DummyConfig:
        def __init__(self):
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.hidden_act = 'silu'

    config = DummyConfig()
    model = Qwen2MLP(config)
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)

    # vanilla version
    output_none = model(hidden_state)

    # gate and up proj
    zero_lor = (
        torch.randn(batch_size, hidden_size, rank),
        torch.zeros(batch_size, rank, hidden_size)
    )

    # down proj
    zero_lor_d = (
        torch.randn(batch_size, hidden_size, rank),
        torch.zeros(batch_size, rank, hidden_size)
    )

    # low rank version
    output_lor = model(hidden_state, lor_g=zero_lor, lor_u=zero_lor, lor_d=zero_lor_d)
    assert torch.allclose(output_none, output_lor, atol=1e-6), 'MLP with low rank computation is not equivalent to naieve version.'

if False:
    test_qwen2mlp_low_rank_equivalence()

# @@@@@@@@@@@@@@@@@@@@


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Q.Qwen2Config, layer_idx: Optional[int] = None):
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

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Q.Qwen2RotaryEmbedding(
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
        cache_position: Optional[torch.LongTensor] = None,

        # low rank params
        lor_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_k: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_v: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_o: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        if lor_q is not None:
            query_states = lor_q(query_states, hidden_states)

        key_states = self.k_proj(hidden_states)
        if lor_k is not None:
            key_states = lor_k(key_states, hidden_states)

        value_states = self.v_proj(hidden_states)
        if lor_v is not None:
            value_states = lor_v(value_states, hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = Q.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = Q.repeat_kv(key_states, self.num_key_value_groups)
        value_states = Q.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

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
        if lor_o is not None:
            attn_output = lor_o(attn_output, hidden_states)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Q.Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            Q.logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Q.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Q.Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,

        # low rank attention params
        lor_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_k: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_v: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_o: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,

        # low rank mlp params
        lor_u: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_g: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            lor_q=lor_q,
            lor_k=lor_k,
            lor_v=lor_v,
            lor_o=lor_o,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, lor_u, lor_g, lor_d)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


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
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,

        # low rank attention params, list of per layer params (or None)
        lor_qs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_ks: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_vs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_os: List[Tuple[torch.Tensor, torch.Tensor]] = None,

        # low rank mlp params, list of per layer params (or None)
        lor_us: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_gs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_ds: List[Tuple[torch.Tensor, torch.Tensor]] = None,

    ) -> Union[Tuple, BaseModelOutputWithPast]:
        lors = [lor_qs, lor_ks, lor_vs, lor_os, lor_us, lor_gs, lor_ds]
        for l in lors:
            assert len(l) == len(self.layers), 'All `lor` params must be set'

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                Q.logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        # if use_cache and not isinstance(past_key_values, Cache) and not self.training:  # original, bans cache if training
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            Q.logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer, lor_q, lor_k, lor_v, lor_o, lor_u, lor_g, lor_d in zip(self.layers, *lors):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    lor_q=lor_q,
                    lor_k=lor_k,
                    lor_v=lor_v,
                    lor_o=lor_o,
                    lor_u=lor_u,
                    lor_g=lor_g,
                    lor_d=lor_d,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    lor_q=lor_q,
                    lor_k=lor_k,
                    lor_v=lor_v,
                    lor_o=lor_o,
                    lor_u=lor_u,
                    lor_g=lor_g,
                    lor_d=lor_d,
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

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = Q._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


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
        cache_position: Optional[torch.LongTensor] = None,

        # low rank attention params, list of per layer params (or None)
        lor_qs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_ks: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_vs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_os: List[Tuple[torch.Tensor, torch.Tensor]] = None,

        # low rank mlp params, list of per layer params (or None)
        lor_us: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_gs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        lor_ds: List[Tuple[torch.Tensor, torch.Tensor]] = None,

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
            cache_position=cache_position,

            # low rank attention params, list of per layer params (or None)
            lor_qs=lor_qs,
            lor_ks=lor_ks,
            lor_vs=lor_vs,
            lor_os=lor_os,

            # low rank mlp params, list of per layer params (or None)
            lor_us=lor_us,
            lor_gs=lor_gs,
            lor_ds=lor_ds,
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
            loss_fct = CrossEntropyLoss()
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


##################################################
# Generation functions

def generate_with_cache(model, model_inputs, max_new_tokens):
    '''Generate tokens autoregressively using the past_key_value cache. '''
    generated_tokens = []
    past_key_values = None
    next_token = None

    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']

    for i in range(max_new_tokens):
        # For the first iteration, use the full prompt. For subsequent
        # iterations, use only the last generated token. `attention_mask` will
        # continue to grow as the entire sequence length seen so far
        if i > 0:
            input_ids = next_token.unsqueeze(1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)
        out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, return_dict=True)
        next_token = out.logits[:, -1].argmax(dim=-1)
        generated_tokens.append(next_token)
        past_key_values = out.past_key_values
    return torch.stack(generated_tokens, dim=-1)


def generate_naive(model, model_inputs, max_new_tokens):
    '''Repeatedly concat the new token to the entire running input_ids tensor, and
run through again. Very inefficient, but useful for ensuring equivalence of the
cache version.'''
    next_token = None
    generated_tokens = []
    for i in range(max_new_tokens):
        if i > 0:
            model_inputs = {
                'input_ids': torch.cat([model_inputs['input_ids'], next_token.unsqueeze(1)], dim=-1),
                # 'attention_mask': torch.cat([model_inputs['attention_mask'], torch.ones_like(next_token.unsqueeze(1))], dim=-1),
            }
        outputs = model(**model_inputs, return_dict=True)
        next_token = outputs.logits[:, -1].argmax(dim=-1)
        generated_tokens.append(next_token)
    return torch.stack(generated_tokens, dim=-1)


# @@@@@@@@@@@@@@@@@@@@
# Double check cache works

if False:
    warnings.warn('NOT using custom model, just testing caching stuff')

    SEED = 152
    torch.manual_seed(152)
    random.seed(SEED)

    DEVICE = 'cuda:1'
    BATCH_SIZE = 32

    # model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
    model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
    # model_name = os.path.expanduser("~/_/models/Qwen2-7B")

    try:
        # fail
        already_loaded
    except:
        print('Loading model')
        model = Q.Qwen2ForCausalLM.from_pretrained(
            model_name,
            # torch_dtype="auto",
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            _attn_implementation='eager',
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        already_loaded = True

    max_new_tokens = 20

    prompt = "Once upon a time there was [A]. The noise a [A] makes is"
    prompts = [prompt] * 16
    input_ids = tokenizer(prompts, return_tensors="pt").to(DEVICE)

    print(f"Generating {max_new_tokens} new tokens\n")

    # Generate with cache
    start_time = time.time()
    output_ids_with_cache = generate_with_cache(model, input_ids, max_new_tokens)
    cache_time = time.time() - start_time

    output_with_cache = tokenizer.batch_decode(output_ids_with_cache, skip_special_tokens=False)
    print(f"Output (with cache):\n{output_with_cache[0]}")
    print(f"Time taken (with cache): {cache_time:.4f} seconds\n")

    # Generate naively
    start_time = time.time()
    output_ids_naive = generate_naive(model, input_ids, max_new_tokens)
    naive_time = time.time() - start_time

    output_naive = tokenizer.batch_decode(output_ids_naive, skip_special_tokens=False)
    print(f"Output (naive):\n{output_naive[0]}")
    print(f"Time taken (naive): {naive_time:.4f} seconds\n")

    # Compare outputs
    outputs_match = output_with_cache == output_naive
    print(f"Outputs match: {outputs_match}")

    # Compare speed
    speedup = naive_time / cache_time
    print(f"Speedup factor: {speedup:.2f}x")

# @@@@@@@@@@@@@@@@@@@@



##################################################
# Sandbox: Pre-chonking, adding some LoRWs

# def generate_with_cache(model, model_inputs, max_new_tokens):
#     ''' Use past_key_values for a theoretical speedup. '''
#     generated_tokens = []
#     past_key_values = None
#     next_token = None

#     input_ids = model_inputs['input_ids']
#     attention_mask = model_inputs['attention_mask']

#     num_layers = model.config.num_hidden_layers

#     # low rank attention params
#     lor_qs = [None] * num_layers
#     lor_ks = [None] * num_layers
#     lor_vs = [None] * num_layers
#     lor_os = [None] * num_layers

#     # low rank mlp params
#     lor_us = [None] * num_layers
#     lor_gs = [None] * num_layers
#     lor_ds = [None] * num_layers


#     # TODO: rm. This hardcodes some lorW
#     D = model.config.hidden_size
#     B, S = model_inputs['input_ids'].shape
#     device = model_inputs['input_ids'].device
#     dtype = model.model.embed_tokens.weight.dtype
#     lori = torch.randn(B, D, 1, dtype=dtype, device=device) * 1e-3
#     loro = torch.randn(B, 1, D, dtype=dtype, device=device) * 1
#     lor_us[0] = (lori, loro)


#     for i in range(max_new_tokens):
#         # For the first iteration, use the full prompt. For subsequent
#         # iterations, use only the last generated token. `attention_mask` will
#         # continue to grow as the entire sequence length seen so far
#         if i > 0:
#             input_ids = next_token.unsqueeze(1)
#             attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)
#         out = model(input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     past_key_values=past_key_values,
#                     return_dict=True,

#                     # low rank attention params, list of per layer params (or None)
#                     lor_qs=lor_qs,
#                     lor_ks=lor_ks,
#                     lor_vs=lor_vs,
#                     lor_os=lor_os,

#                     # low rank mlp params, list of per layer params (or None)
#                     lor_us=lor_us,
#                     lor_gs=lor_gs,
#                     lor_ds=lor_ds,
#                     )
#         next_token = out.logits[:, -1].argmax(dim=-1)
#         generated_tokens.append(next_token)
#         past_key_values = out.past_key_values
#     return torch.stack(generated_tokens, dim=-1)


# if True:
#     DEVICE = 'cuda:1'
#     torch.manual_seed(152)
#     model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
#     model = Qwen2ForCausalLM.from_pretrained(
#         model_name,
#         # torch_dtype="auto",
#         torch_dtype=torch.bfloat16,
#         device_map=DEVICE,
#         _attn_implementation='eager',
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     prompt = "Once upon a time in a galaxy far away, "
#     prompts = [prompt] * 1
#     input_ids = tokenizer(prompts, return_tensors="pt").to(DEVICE)
#     output_ids = generate_with_cache(model, input_ids, max_new_tokens=200)
#     output = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
#     print(output[0])
