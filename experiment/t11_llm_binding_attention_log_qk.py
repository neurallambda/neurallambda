'''

Trace QK outputs from LLM

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
import transformers.models.qwen2.modeling_qwen2 as Q

from typing import Optional, Tuple, List
import warnings
import math

from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid
import os
import re

import t11_llm_binding_attention_data as Data

torch.manual_seed(152)

DEVICE = 'cuda'
BATCH_SIZE = 128

# model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# # model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")


##################################################
# Tracing Stuff
#
# 1. copy class
# 2. augment with tracing capabilities
# 3. run it, and log outputs

def ninf_threshold(dtype):
    ''' for attention masks in (-inf, 0) '''
    if dtype == torch.float16:
        return -1e+4
    elif dtype == torch.bfloat16:
        return -1e+4
    else:
        return -1e+37  # Default for float32 and float64


def infer_position_ids(x, length):
    '''In order to not cache embeddings with RoPE already added in, we need to add
    RoPE after grabbing from the cache. To do that, we need to infer
    position_ids, which this fn does.

    '''
    # Ensure x is 2D
    x = x.view(-1, 1)

    # Replace any value <= 0 with 1
    x = torch.max(x, torch.ones_like(x))

    # Create a range sequence for each element
    batch_size = x.size(0)
    range_tensor = torch.arange(start=length, end=0, step=-1, device=x.device).expand(batch_size, -1)
    result = x - range_tensor + 1

    # I think pos-ids are expected to be >=1
    result = torch.max(result, torch.ones_like(result))

    return result

# x = torch.tensor([[9],
#                   [20]])
# print()
# print(infer_position_ids(x, 3))

def shift_halves(x):
    """Permute x in half for use in RoPE calculation"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    roped = (x * cos) + (shift_halves(x) * sin)
    return roped

class Qwen2AttentionWithLogging(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Q.Qwen2Config, layer_idx, trace_out):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.trace_out = trace_out

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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if q_len == 1:
            raise ValueError('Logging is only intended when you do a single forward pass, IE it works when you call `model(x)`, not `model.generate(x)`')

        # hidden_states: [B, S, 1536]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

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

        # # $$$$$$$$$$
        # # RoPE (normal location, pre cache)
        # # query_states, key_states = Q.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        # key_states  = apply_rotary_pos_emb(key_states, cos, sin, position_ids)
        # # $$$$$$$$$$

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Log intermediate states
        #
        #   undo the transposition, so that the trained replacement only has to
        #   mimic the QK weights, not the transposition too.
        log_q = query_states.transpose(1, 2).view(bsz, q_len, self.num_heads * self.head_dim)
        log_k = key_states.transpose(1, 2).view(bsz, q_len, self.num_key_value_heads * self.head_dim)
        self.log_q_k(attention_mask, hidden_states, log_q, log_k)

        # $$$$$$$$$$
        # RoPE (post cache)
        #
        # in order to rotate post cache, we need longer position ids. we're not
        # caching pos_ids, so we'll do a hack to create what we think are the
        # preceding ids for the keys (current position, ranging backward)
        if position_ids.shape[1] == 1:
            k_len = key_states.shape[2]
            key_position_ids = infer_position_ids(position_ids, k_len)
        else:
            key_position_ids = position_ids
        query_states = apply_rotary_pos_emb(query_states, cos, sin, position_ids)
        key_states  = apply_rotary_pos_emb(key_states, cos, sin, key_position_ids)
        # $$$$$$$$$$


        # repeat k/v heads if n_kv_heads < n_heads
        key_states   = Q.repeat_kv(key_states, self.num_key_value_groups)
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

    def log_q_k(self,
                attention_mask: torch.Tensor, # [B, 1, S, S] # diagonal, offset by padding tokens. On successive passes = [B, 1, 1, S+n]
                hidden_states: torch.Tensor,  # [B, S, D] first, then [_, 1, _]
                query_states: torch.Tensor,  # [B, NUM_HEADS, S, D // NUM_HEADS] first, then [_, _, 1, _]
                key_states: torch.Tensor,  # [B, NUM_KV_HEADS, S, D // NUM_HEADS] first, then [?]
                                           # (Group Query Attention)
                ):
        # print('FROM LOG')
        # print(f'{attention_mask.shape=}')
        # print(f'{hidden_states.shape=}')
        # print(f'{query_states.shape=}')
        # print(f'{key_states.shape=}')

        B, _, S, _ = attention_mask.shape
        device = attention_mask.device
        thresh = ninf_threshold(attention_mask.dtype)
        a = torch.ones((B, S), device=device, dtype=torch.bool)  # peel off last of the diagonal
        a[attention_mask[:, 0, -1] < thresh] = False

        self.trace_out['attention_mask'].append(a.detach().cpu())
        self.trace_out['hidden_states'].append(hidden_states.detach().cpu())
        self.trace_out['q'].append(query_states.detach().cpu())
        self.trace_out['k'].append(key_states.detach().cpu())

    # def log_q_k_to_dir(self,
    #             hidden_states: torch.Tensor,  # [B, S, D]
    #             query_states: torch.Tensor,  # [B, NUM_HEADS, S, D // NUM_HEADS]
    #             key_states: torch.Tensor,  # [B, NUM_KV_HEADS, S, D // NUM_HEADS] (Group Query Attention)
    #             past_key_value: Optional[DynamicCache] = None
    #             ):
    #     # a unique identifier for this forward pass
    #     pass_id = uuid.uuid4().hex

    #     # Log hidden_states (inputs)
    #     h_file = os.path.join(LOG_DIR, f"layer_{self.layer_idx}_hidden_states_{pass_id}.pt")
    #     torch.save(hidden_states.detach().cpu(), h_file)

    #     # Log Q
    #     q_file = os.path.join(LOG_DIR, f"layer_{self.layer_idx}_Q_{pass_id}.pt")
    #     torch.save(query_states.detach().cpu(), q_file)

    #     # Log K, including cached data if available
    #     if past_key_value is not None and self.layer_idx < len(past_key_value.key_cache):
    #         # Concatenate cached key states with current key states
    #         full_key_states = torch.cat([past_key_value.key_cache[self.layer_idx], key_states], dim=-2)
    #     else:
    #         full_key_states = key_states

    #     k_file = os.path.join(LOG_DIR, f"layer_{self.layer_idx}_K_{pass_id}.pt")
    #     torch.save(full_key_states.detach().cpu(), k_file)

    def state_dict(self, *args, **kwargs):
        # Customize state_dict if needed
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        # Customize load_state_dict if needed
        return super().load_state_dict(state_dict, strict)

    @classmethod
    def from_qwen2_attention(cls, qwen2_attention, trace_out):
        with torch.no_grad():
            config = qwen2_attention.config
            layer_idx = qwen2_attention.layer_idx
            custom_attn = cls(config, layer_idx, trace_out)

            # Copy weights from Qwen2Attention
            custom_attn.q_proj.weight.data = qwen2_attention.q_proj.weight.data.clone()
            custom_attn.q_proj.bias.data = qwen2_attention.q_proj.bias.data.clone()
            custom_attn.k_proj.weight.data = qwen2_attention.k_proj.weight.data.clone()
            custom_attn.k_proj.bias.data = qwen2_attention.k_proj.bias.data.clone()
            custom_attn.v_proj.weight.data = qwen2_attention.v_proj.weight.data.clone()
            custom_attn.v_proj.bias.data = qwen2_attention.v_proj.bias.data.clone()
            custom_attn.o_proj.weight.data = qwen2_attention.o_proj.weight.data.clone()

            # Ensure same device and dtype
            device = qwen2_attention.q_proj.weight.device
            dtype = qwen2_attention.q_proj.weight.dtype
            custom_attn = custom_attn.to(device=device, dtype=dtype)

            return custom_attn


def replace_attention(model):
    ''' qk_list is where the values will be saved, in memory '''
    trace_outs = {}
    for i, layer in enumerate(model.model.layers):
        trace_out = {
            'attention_mask': [],
            'hidden_states': [],
            'k': [],
            'q': []
        }  # holds values the model outputs
        trace_outs[i] = trace_out
        new_attention = Qwen2AttentionWithLogging.from_qwen2_attention(layer.self_attn, trace_out)
        layer.self_attn = new_attention
    return model, trace_outs


##################################################
# Trace QK

def generate_qk_dataset(model, dataloader):
    for batch in tqdm(dataloader):
        prompt_ids = batch['prompt_ids'].to(DEVICE)
        prompt_attention_mask = batch['prompt_attention_mask'].to(DEVICE)
        answer_ids = batch['answer_ids'].to(DEVICE)
        answer_attention_mask = batch['answer_attention_mask'].to(DEVICE)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, answer_attention_mask], dim=1)

        # populate each `self_attn.trace_out`
        model(input_ids,
              attention_mask=attention_mask,
              labels=input_ids)


##################################################
# Hand Check stuff

if False:
    print()
    print('Hand Checking Logging')

    model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu",
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # trace_outs will contain traced QK values
    model, trace_outs = replace_attention(cpu_model)
    model = model.to(DEVICE)

    prompt = "Once upon a time there was a"
    model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')


    # out = model.generate(**model_inputs, max_new_tokens=30)
    # response = tokenizer.batch_decode(out, skip_special_tokens=False)[0]
    # print(response)



# ##################################################
# # Logging/Reading from disk

# def collect_qk_data(log_dir: str, layer_idx: int, batch_size: int) -> DataLoader:
#     h_pattern = re.compile(f"layer_{layer_idx}_hidden_states_.*\\.pt$")
#     q_pattern = re.compile(f"layer_{layer_idx}_Q_.*\\.pt$")
#     k_pattern = re.compile(f"layer_{layer_idx}_K_.*\\.pt$")

#     h_files = sorted([f for f in os.listdir(log_dir) if h_pattern.match(f)])
#     q_files = sorted([f for f in os.listdir(log_dir) if q_pattern.match(f)])
#     k_files = sorted([f for f in os.listdir(log_dir) if k_pattern.match(f)])

#     assert len(q_files) == len(k_files) == len(h_files), f"Mismatch in number of Q, K, and hidden_states files for layer {layer_idx}"

#     h_tensors = []
#     q_tensors = []
#     k_tensors = []

#     for h_file, q_file, k_file in zip(h_files, q_files, k_files):
#         h_tensor = torch.load(os.path.join(log_dir, h_file))
#         q_tensor = torch.load(os.path.join(log_dir, q_file))
#         k_tensor = torch.load(os.path.join(log_dir, k_file))

#         h_tensors.append(h_tensor)
#         q_tensors.append(q_tensor)
#         k_tensors.append(k_tensor)

#     h_tensors = torch.cat(h_tensors, dim=0)
#     q_tensors = torch.cat(q_tensors, dim=0)
#     k_tensors = torch.cat(k_tensors, dim=0)

#     dataset = TensorDataset(h_tensors, q_tensors, k_tensors)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     return dataloader

# # Example usage
# layer_idx = 0  # Process data for the first layer
# qk_dataloader = collect_qk_data(LOG_DIR, layer_idx, batch_size=2)

# for batch in qk_dataloader:
#     h, q, k = batch
#     print(f"Hidden states shape: {h.shape}")
#     print(f"Q shape: {q.shape}")
#     print(f"K shape: {k.shape}")
#     break
