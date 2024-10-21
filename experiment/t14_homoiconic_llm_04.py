'''

Apply Metalearning-Hypernet stuff to Transformer arch

The previous version wired everything up and did a smoke test, this version will train in a toy dataset.


TO CONSIDER:
  ** position_ids
     https://discuss.huggingface.co/t/llama-position-ids/75870/2
     position is -1 where padded
     prepare_inputs_for_generation demonstrates usage

  ** cache_position

  ** input_embeds


PROVENANCE:
- t14_homoiconic_llm_03.py

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


import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

from neurallambda.lab.common import print_model_info

import t14_homoiconic_llm_model as Q
import t14_homoiconic_llm_add_tokens as AT

from datasets import Dataset as HFDataset

import importlib
try:
    importlib.reload(C)
except NameError:
    import t14_homoiconic_llm_columnize_03 as C


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

def create_position_ids(attention_mask):
    """
    Create position IDs based on the attention mask.
    Set position to -1 for padding tokens.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, -1)
    return position_ids


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
    col_out_logits = []
    past_key_values = None
    attention_mask = None

    # Iterate over each column in the batch
    for i, batch_column in enumerate(col_inputs):
        input_ids = batch_column['input_ids']
        new_attention_mask = batch_column['attention_mask']
        position_ids = batch_column['position_ids']

        # skip empty batches. this can happen because of padded blocks.
        if input_ids.numel() == 0:
            continue

        # `attention_mask` will continue to grow as the entire sequence length
        # seen so far (even though input_ids will only be new inputs, not past)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
        else:
            attention_mask = new_attention_mask

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,  # OPTIM: we don't need this for non-lor blocks
                    **lors,
                    )
        col_out_logits.append(out.logits)

        past_key_values = out.past_key_values
        assert past_key_values is not None, "past_key_values cannot be None. Typically `transformers` models won't return past_key_values during training, so, you need to modify the underlying model class."

        # EXPERIMENT: TODO fix: for testing, odd columns are the lor_block
        warnings.warn('hardcoded lor blocks')
        if i % 2 == 1:
            h_emb = out.hidden_states[lor_layer]
            for lor_key, (lor_i, lor_j) in lor_parse.items():
                lors[lor_key][lor_layer] = (
                    h_emb[:, lor_i].unsqueeze(2),
                    h_emb[:, lor_j].unsqueeze(1)
                )

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



def run_epoch(model, dataloader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch_cols in tqdm(dataloader, desc="Training" if train else "Evaluating"):
            B = batch_cols[0]['input_ids'].shape[0]
            batch_cols = [x.to(device) for x in batch_cols]
            lors = empty_lors(model.config.num_hidden_layers)
            out_logits, _, _  = forward_columns(model, batch_cols, lors)
            loss = loss_fn([x['input_ids'] for x in batch_cols],
                           out_logits,
                           [x['loss_mask'] for x in batch_cols])

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples
    return avg_loss


##################################################

SEED = 152
torch.manual_seed(152)
random.seed(SEED)

DEVICE = 'cuda:0'
BATCH_SIZE = 8

# model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-7B")

try:
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
    AT.assert_unique_tokens(tokenizer, [x[0] for x in new_token_pairs])
    AT.add_and_initialize_tokens(model, tokenizer, new_token_pairs)
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

    already_loaded = True


##########
# LOR stuff

# hard code the layer these apply to
lor_layer = 2  # qwen1.5B has 28 layers
# Q|| K|| V|| O|| G|| U|| D||
lor_block = "^@Q^@|^@|^@K^@|^@|^@V^@|^@|^@O^@|^@|^@G^@|^@|^@U^@|^@|^@D^@|^@|"

# metatoken mask, for masking loss. metatokens should be predicted, thus make
# it to loss. meta weights should not be affected directly by the cross entropy
# loss.
lor_mask = [1, 0, 0] * 7  # note: should be list, not tensor. `datasets` converts it back to a list (then tensor again) anyway
lor_mask = [0, 0, 0] * 7  # TODO: rm, this mask does NOT learn output of metatokens

# NOTE: lor_metatokens and lor_parse are offset by 1, because they parse out of
# the outputs. The metatoken ^@Q in the input gets converted into the first
# metaweight; the next input token, a dummy weight, gets converted into the
# second metaweight.

# where to parse to collect metaweights
lor_parse = {
    'lor_qs': [0, 1],
    'lor_ks': [3, 4],
    'lor_vs': [6, 7],
    'lor_os': [9, 10],
    'lor_gs': [12, 13],
    'lor_us': [15, 16],
    'lor_ds': [18, 19],
}


####################
# Data v2

train_small = load_dataset("neurallambda/arithmetic_dataset", split="train_small").select(range(100))  # todo: rm, this shrinks training
test_small = load_dataset("neurallambda/arithmetic_dataset", split="test_small").select(range(20))

def insert_lor_blocks(dataset: HFDataset, block_content) -> HFDataset:
    """
    Insert LOR blocks between text elements in the dataset.
    """
    def process_row(row):
        prepared_data = []
        for item in row['input']:
            prepared_data.append({"type": "text", "content": item, 'include_in_loss': False})
            prepared_data.append({"type": "lor", "content": block_content, 'include_in_loss': True, 'loss_mask': lor_mask})
        # # Remove the last LOR block
        # prepared_data = prepared_data[:-1]
        # Add the output
        prepared_data.append({"type": "text", "content": row['output'], 'include_in_loss': True})
        return {"prepared_data": prepared_data}

    # NOTE: the `map` process injects missing keys found among any rows, ie loss_mask
    out = dataset.map(process_row, remove_columns=["input", "output"])
    return out


# Apply the preparation to a specific split
train_small = insert_lor_blocks(train_small, lor_block)
# # Sample
# for example in train_small.select(range(5)):  # Show first 5 examples
#     print(example)
#     print()
train_dl = C.create_dataloader(
    dataset=train_small,
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
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    device='cpu',
    shuffle=False,
    num_workers=0
)


##########
#

if True:
    num_epochs = 20
    lr = 1e-3
    wd = 0.0

    parameters = model.parameters()
    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=wd)

    # START_BLOCK_2

    # model.train()
    # xx = tokenizer('hi there', return_tensors='pt')
    # xx = xx.to(DEVICE)
    # o = model(**xx, **empty_lors(28))
    # print(o.past_key_values[0])

    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(num_epochs):
        train_loss = run_epoch(model, train_dl, optimizer, DEVICE, train=True)
        train_losses.append(train_loss)
        test_loss = run_epoch(model, test_dl, optimizer, DEVICE, train=False)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # END_BLOCK_2


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

def generate(model, col_inputs, lors, max_new_tokens):
    '''Recurrently process column blocks of ids, concatenating attention_mask and
    past_key_values across generations. Can generate new tokens if max_new_tokens > 0.'''
    col_out_logits = []
    past_key_values = None
    attention_mask = None
    all_output_ids = []
    tokens_generated = 0

    # Iterate over each column in the batch
    for i, batch_column in enumerate(col_inputs):
        input_ids = batch_column['input_ids']
        new_attention_mask = batch_column['attention_mask']
        position_ids = batch_column['position_ids']

        # skip empty batches. this can happen because of padded blocks.
        if input_ids.numel() == 0:
            continue

        # `attention_mask` will continue to grow as the entire sequence length
        # seen so far (even though input_ids will only be new inputs, not past)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
        else:
            attention_mask = new_attention_mask

        # run column
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,  # OPTIM: we don't need this for non-lor blocks
                    **lors,
                    )
        col_out_logits.append(out.logits)
        # all_output_ids.append(input_ids)

        past_key_values = out.past_key_values
        assert past_key_values is not None, "past_key_values cannot be None. Typically `transformers` models won't return past_key_values during training, so, you need to modify the underlying model class."

        # EXPERIMENT: TODO fix: for testing, odd columns are the lor_block
        warnings.warn('hardcoded lor blocks')
        if i % 2 == 1:
            h_emb = out.hidden_states[lor_layer]
            for lor_key, (lor_i, lor_j) in lor_parse.items():
                lors[lor_key][lor_layer] = (
                    h_emb[:, lor_i].unsqueeze(2),
                    h_emb[:, lor_j].unsqueeze(1)
                )

    # If max_new_tokens > 0, continue generating new tokens
    while tokens_generated < max_new_tokens:

        # Get the last token's logits
        last_token_logits = out.logits[:, -1, :]

        # Sample the next token (you can replace this with your preferred sampling method)
        next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)

        # Update input_ids for the next iteration
        input_ids = next_token
        all_output_ids.append(input_ids)

        # Update position_ids
        position_ids = (position_ids[:, -1] + 1).unsqueeze(-1)

        # Update attention_mask
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        # Generate the next token
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    **lors,
                    )

        col_out_logits.append(out.logits)
        past_key_values = out.past_key_values

        tokens_generated += 1

    return col_out_logits, attention_mask, past_key_values, torch.cat(all_output_ids, dim=-1)

def t(x):
    return {'type': 'text', 'content': x, 'include_in_loss': False}  # loss not needed during inference

def w(x):
    return {'type': 'lor', 'content': x, 'include_in_loss': False}  # loss not needed during inference

model.eval()
with torch.no_grad():

    test_prompts = [
        [t("var_0=1"), w(lor_block), t("var_1=2"), w(lor_block), t("var_3=var_0 + var_1"), w(lor_block), t("solve(var_3)=")],  # 3
        [t("var_0=3"), w(lor_block), t("var_1=5"), w(lor_block), t("var_3=var_0 - var_1"), w(lor_block), t("solve(var_3)=")],  # -2
        [t("var_0=2"), w(lor_block), t("var_1=3"), w(lor_block), t("var_3=var_0 * var_1"), w(lor_block), t("solve(var_3)=")],  # 6
        [t("hi there"), w(lor_block)],
        [t("wow whats up")],
    ]

    inputs = C.create_column_batch_inputs(test_prompts, tokenizer, device='cuda')
    lors = empty_lors(model.config.num_hidden_layers)

    output_logits, col_attention_mask, col_past_key_values, output_ids = generate(model, inputs, lors, max_new_tokens=20)
    # for display add in col outputs
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    print('---------- gen results')
    for o in outputs:
        print(f'"{o}"')

# END_BLOCK_1



# x = torch.arange(60).reshape(3, 4, 5)
# m = torch.tensor([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
# ], dtype=torch.bool)
