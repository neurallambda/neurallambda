'''

Apply Metalearning-Hypernet stuff to Transformer arch

NOTE: this first version builds up the machinery and ensures equivalence with naive approaches


TO CONSIDER:
  ** position_ids
  ** input_embeds
  ** cache_position

PROVENANCE:
- t13_metalearning_hypernet_03.py

----------
QWEN 0.5B

| Layer                                    | Parameter   | Size          |
|------------------------------------------+-------------+---------------+
| model.embed_tokens.weight                | 136,134,656 | [151936, 896] |
| **Transformer Layer (x24, layers 0-23)** |             |               |
| layer.self_attn.q_proj.weight            | 802,816     | [896, 896]    |
| layer.self_attn.q_proj.bias              | 896         | [896]         |
| layer.self_attn.k_proj.weight            | 114,688     | [128, 896]    | 128 * 7 = 896
| layer.self_attn.k_proj.bias              | 128         | [128]         |
| layer.self_attn.v_proj.weight            | 114,688     | [128, 896]    |
| layer.self_attn.v_proj.bias              | 128         | [128]         |
| layer.self_attn.o_proj.weight            | 802,816     | [896, 896]    |
| layer.mlp.gate_proj.weight               | 4,358,144   | [4864, 896]   | x5.43
| layer.mlp.up_proj.weight                 | 4,358,144   | [4864, 896]   |
| layer.mlp.down_proj.weight               | 4,358,144   | [896, 4864]   |
| layer.input_layernorm.weight             | 896         | [896]         |
| layer.post_attention_layernorm.weight    | 896         | [896]         |
|                                          |             |               |
| **Final Layers**                         |             |               |
| model.norm.weight                        | 896         | [896]         |
| lm_head                                  | (tied to embeddings)        |
| **Total Parameters**                     | 494,032,768 |               |


----------
QWEN 1.5B

| Layer                                    | Parameter   | Size          |
|------------------------------------------|-------------|---------------|
| model.embed_tokens.weight                | 233,373,696 | [151936, 1536]|
| **Transformer Layer (x28, layers 0-27)** |             |               |
| layer.self_attn.q_proj.weight            | 2,359,296   | [1536, 1536]  |
| layer.self_attn.q_proj.bias              | 1,536       | [1536]        |
| layer.self_attn.k_proj.weight            | 393,216     | [256, 1536]   | 256 * 6 = 1536
| layer.self_attn.k_proj.bias              | 256         | [256]         |
| layer.self_attn.v_proj.weight            | 393,216     | [256, 1536]   |
| layer.self_attn.v_proj.bias              | 256         | [256]         |
| layer.self_attn.o_proj.weight            | 2,359,296   | [1536, 1536]  |
| layer.mlp.gate_proj.weight               | 13,762,560  | [8960, 1536]  | x5.83
| layer.mlp.up_proj.weight                 | 13,762,560  | [8960, 1536]  |
| layer.mlp.down_proj.weight               | 13,762,560  | [1536, 8960]  |
| layer.input_layernorm.weight             | 1,536       | [1536]        |
| layer.post_attention_layernorm.weight    | 1,536       | [1536]        |
|                                          |             |               |
| **Final Layers**                         |             |               |
| model.norm.weight                        | 1,536       | [1536]        |
| lm_head                                  | (tied to embeddings)        |
| **Total Parameters**                     | 1,543,714,304 |             |


----------
QWEN 7B

| Layer                                    | Parameter   | Size          | Notes |
|------------------------------------------|-------------|---------------|-------|
| model.embed_tokens.weight                | 544,997,376 | [152064, 3584]|       |
| **Transformer Layer (x28, layers 0-27)** |             |               |       |
| layer.self_attn.q_proj.weight            | 12,845,056  | [3584, 3584]  |       |
| layer.self_attn.q_proj.bias              | 3,584       | [3584]        |       |
| layer.self_attn.k_proj.weight            | 1,835,008   | [512, 3584]   |       | 512 * 7 = 3584
| layer.self_attn.k_proj.bias              | 512         | [512]         |       |
| layer.self_attn.v_proj.weight            | 1,835,008   | [512, 3584]   |       |
| layer.self_attn.v_proj.bias              | 512         | [512]         |       |
| layer.self_attn.o_proj.weight            | 12,845,056  | [3584, 3584]  |       |
| layer.mlp.gate_proj.weight               | 67,895,296  | [18944, 3584] |       |
| layer.mlp.up_proj.weight                 | 67,895,296  | [18944, 3584] |       |
| layer.mlp.down_proj.weight               | 67,895,296  | [3584, 18944] |       | x5.3
| layer.input_layernorm.weight             | 3,584       | [3584]        |       |
| layer.post_attention_layernorm.weight    | 3,584       | [3584]        |       |
|                                          |             |               |       |
| **Final Layers**                         |             |               |       |
| model.norm.weight                        | 3,584       | [3584]        |       |
| lm_head.weight                           | 544,997,376 | [152064, 3584]|       |
| **Total Parameters**                     | 7,615,616,512 |             |       |


----------
Llama 3.1

| Component             | 8B                 | 70B                | 405B               |
|-----------------------+--------------------+--------------------+--------------------|
| Layers                | 32                 | 80                 | 126                |
| Model Dimension       | 4,096              | 8,192              | 16,384             |
| FFN Dimension         | 14,336             | 28,672             | 53,248             |
| Attention Heads       | 32                 | 64                 | 128                |
| Key/Value Heads       | 8                  | 8                  | 8                  |
| Peak Learning Rate    | 3 × 10^-4          | 1.5 × 10^-4        | 8 × 10^-5          |
| Activation Function   | SwiGLU             | SwiGLU             | SwiGLU             |
| Vocabulary Size       | 128,000            | 128,000            | 128,000            |


----------
Llama 3.1 8B

| Component                         | Parameters    | Shape          |
|-----------------------------------+---------------+----------------|
| model.embed_tokens.weight         | 525,336,576   | [128256, 4096] |
| model.layers (x32)                |               |                |
| - self_attn.q_proj.weight         | 16,777,216    | [4096, 4096]   |
| - self_attn.k_proj.weight         | 4,194,304     | [1024, 4096]   |
| - self_attn.v_proj.weight         | 4,194,304     | [1024, 4096]   |
| - self_attn.o_proj.weight         | 16,777,216    | [4096, 4096]   |
| - mlp.gate_proj.weight            | 58,720,256    | [14336, 4096]  | x3.5
| - mlp.up_proj.weight              | 58,720,256    | [14336, 4096]  |
| - mlp.down_proj.weight            | 58,720,256    | [4096, 14336]  |
| - input_layernorm.weight          | 4,096         | [4096]         |
| - post_attention_layernorm.weight | 4,096         | [4096]         |
| model.norm.weight                 | 4,096         | [4096]         |
| lm_head.weight                    | 525,336,576   | [128256, 4096] |
| Total Parameters                  | 8,030,261,248 |                |



forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, cache_position: Optional[torch.LongTensor] = None) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]
 |      The [`Qwen2ForCausalLM`] forward method, overrides the `__call__` special method.
 |
 |      <Tip>
 |
 |      Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
 |      instance afterwards instead of this since the former takes care of running the pre and post processing steps while
 |      the latter silently ignores them.
 |
 |      </Tip>
 |
 |      Args:
 |          input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
 |              Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
 |              it.
 |
 |              Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
 |              [`PreTrainedTokenizer.__call__`] for details.
 |
 |              [What are input IDs?](../glossary#input-ids)
 |          attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
 |              Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
 |
 |              - 1 for tokens that are **not masked**,
 |              - 0 for tokens that are **masked**.
 |
 |              [What are attention masks?](../glossary#attention-mask)
 |
 |              Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
 |              [`PreTrainedTokenizer.__call__`] for details.
 |
 |              If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
 |              `past_key_values`).
 |
 |              If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
 |              and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
 |              information on the default strategy.
 |
 |              - 1 indicates the head is **not masked**,
 |              - 0 indicates the head is **masked**.
 |          position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
 |              Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
 |              config.n_positions - 1]`.
 |
 |              [What are position IDs?](../glossary#position-ids)
 |          past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
 |              Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
 |              blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
 |              returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
 |
 |              Two formats are allowed:
 |              - a [`~cache_utils.Cache`] instance;
 |              - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
 |              shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
 |              cache format.
 |
 |              The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
 |              legacy cache format will be returned.
 |
 |              If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
 |              have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
 |              of shape `(batch_size, sequence_length)`.
 |          inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
 |              Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
 |              is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
 |              model's internal embedding lookup matrix.
 |          use_cache (`bool`, *optional*):
 |              If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
 |              `past_key_values`).
 |          output_attentions (`bool`, *optional*):
 |              Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
 |              tensors for more detail.
 |          output_hidden_states (`bool`, *optional*):
 |              Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
 |              more detail.
 |          return_dict (`bool`, *optional*):
 |              Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
 |          cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
 |              Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
 |              this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
 |              the complete sequence length.
 |
 |          Args:
 |              labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
 |                  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
 |                  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
 |                  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
 |
 |
 |          Returns:
 |              [`transformers.modeling_outputs.CausalLMOutputWithPast`] or `tuple(torch.FloatTensor)`: A [`transformers.modeling_outputs.CausalLMOutputWithPast`] or a tuple of
 |              `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
 |              elements depending on the configuration ([`Qwen2Config`]) and inputs.
 |
 |              - **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
 |              - **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
 |              - **past_key_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
 |                `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
 |
 |                Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
 |                `past_key_values` input) to speed up sequential decoding.
 |              - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
 |                one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
 |
 |                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
 |              - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
 |                sequence_length)`.
 |
 |                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
 |                heads.


'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from typing import Optional, Tuple, Union, List, Dict, Any
import warnings
import math

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import time
from neurallambda.lab.common import print_model_info

import t14_homoiconic_llm_model as Q
import t14_homoiconic_llm_add_tokens as AT


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
# Sandbox: Chonking, to collect lorws occassionally

def empty_lors(num_layers):
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

def forward_columns(model, col_inputs, lors):
    '''Recurrently process column blocks of ids, concatenating attention_mask and
past_key_values across generations. '''
    generated_tokens = []
    past_key_values = None
    next_token = None

    # Iterate over columns of the batch
    attention_mask = None
    for i, batch_column in enumerate(col_inputs):
        input_ids = batch_column['input_ids']
        new_attention_mask = batch_column['attention_mask']

        # `attention_mask` will continue to grow as the entire sequence length
        # seen so far
        if i > 0:
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
        if i == 0:
            attention_mask = new_attention_mask

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    **lors,
                    )
        generated_tokens.append(out.logits.argmax(dim=-1))
        past_key_values = out.past_key_values

    # for developing in interpreter mode
    save = ['past_key_values', 'hidden_states']
    for name, value in locals().items():
        if name in save:
            globals()[name] = value

    return torch.cat(generated_tokens, dim=-1), attention_mask, past_key_values


def generate_with_cache(model, input_ids, attention_mask, lors, max_new_tokens, past_key_values=None):
    '''Generate tokens autoregressively using or initializing the past_key_value cache.'''
    generated_tokens = []
    next_token = None
    for i in range(max_new_tokens):
        # For the first iteration, use the full prompt. For subsequent
        # iterations, use only the last generated token. `attention_mask` will
        # continue to grow as the entire sequence length seen so far
        if i > 0:
            input_ids = next_token.unsqueeze(1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            return_dict=True,
            **lors,
        )
        next_token = out.logits[:, -1].argmax(dim=-1)
        generated_tokens.append(next_token)
        attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)
        past_key_values = out.past_key_values
    return (
        torch.stack(generated_tokens, dim=-1),
        attention_mask,
        past_key_values
    )


##################################################

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
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    # Add new tokens
    AT.assert_unique_tokens(tokenizer, [x[0] for x in new_token_pairs])
    AT.add_and_initialize_tokens(model, tokenizer, new_token_pairs)
    AT.test_token_behavior(model, tokenizer, new_token_pairs)

    if False:
        print('Testing that new tokens are parsable')
        toks = [x[0] for x in new_token_pairs]
        AT.assert_parsability_1(tokenizer, toks)
        AT.assert_parsability_2(tokenizer, toks)
        AT.assert_parsability_3(tokenizer, toks)

    if False:
        print_model_info(model)

    already_loaded = True


# initialize LoRWs
lors = empty_lors(model.config.num_hidden_layers)

# # TODO: rm. This hardcodes some lorW
# D = model.config.hidden_size
# B, S = batch_column['input_ids'].shape
# device = batch_column['input_ids'].device
# dtype = model.model.embed_tokens.weight.dtype
# lori = torch.randn(B, D, 1, dtype=dtype, device=device) * 1e-3
# loro = torch.randn(B, 1, D, dtype=dtype, device=device) * 1
# lors['lor_us'][0] = (lori, loro)

prompt_chonkss = [
    ["Once upon a time in a galaxy far away, ", "there was a^@Q ", "lion who ", "went"],
    # ["Once upon a time in a galaxy far away, ", "there was a&nbsp"]
]

# reorient prompt_chonkss into columns, since batches of chonks will be processed column wise
col_prompt_chonkss = zip(*prompt_chonkss)

col_inputs = []
for prompt_chonks in col_prompt_chonkss:
    inputs = tokenizer(prompt_chonks, return_tensors="pt").to(DEVICE)
    col_inputs.append(inputs)
col_output_ids, attention_mask, past_key_values = forward_columns(model, col_inputs, lors)
col_outputs = tokenizer.batch_decode(col_output_ids, skip_special_tokens=False)

print('---------- col results')
for o in col_outputs:
    print(o)


#####
# Inference, ie Autoregressive Continuation

input_ids = col_output_ids[:, -1].unsqueeze(1)
attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)
output_ids, col_attention_mask, col_past_key_values = generate_with_cache(model, input_ids, attention_mask, lors, max_new_tokens=20, past_key_values=past_key_values)
# for display add in col outputs
output_ids = torch.cat([col_output_ids[:, -1:], output_ids], dim=-1)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
print('---------- gen results')
for o in outputs:
    print(o)


#####
# Naive generation to ensure equivalence

p_ids  = torch.cat([x['input_ids'] for x in col_inputs], dim=-1)
p_attn = torch.cat([x['attention_mask'] for x in col_inputs], dim=-1)

output_ids, attention_mask, past_key_values = generate_with_cache(model, p_ids, p_attn, lors, max_new_tokens=20)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
print('---------- naieve generation')
for o in outputs:
    print(o)


#####
# Check past key values are the same

for l_ix in range(model.config.num_hidden_layers):
    for i in [0, 1]:
        at = col_past_key_values[l_ix][i]
        bt = past_key_values[l_ix][i]
        for t in range(10):  # bc the lengths of both aren't the same
            a = at[:, :, t]
            b = bt[:, :, t]
            assert torch.allclose(a, b, atol=1e-4), f'{l_ix=}, {i=}, {t=}'
print('past_key_values are all same')
