'''

Apply Metalearning-Hypernet stuff to Transformer arch

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
| layer.self_attn.k_proj.weight            | 114,688     | [128, 896]    |
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

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
from neurallambda.lab.common import print_model_info

SEED = 152
torch.manual_seed(152)
random.seed(SEED)

DEVICE = 'cuda'
BATCH_SIZE = 32

model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")

def add_and_initialize_tokens(model: Q.Qwen2ForCausalLM, tokenizer: AutoTokenizer, new_token_pairs: list[tuple[str, str]]):
    new_tokens = [pair[0] for pair in new_token_pairs]

    # Add new tokens to the tokenizer
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_tokens} new tokens to the tokenizer.")

    # Get the current sizes
    current_embed_size = model.model.embed_tokens.weight.shape[0]
    current_vocab_size = len(tokenizer)

    print(f"Current embed size: {current_embed_size}")
    print(f"Current vocab size: {current_vocab_size}")

    if current_vocab_size > current_embed_size:
        raise ValueError("Tokenizer vocab size exceeds model embed size. This scenario is not handled.")

    # Initialize new token embeddings
    assert id(model.model.embed_tokens.weight) == id(model.lm_head.weight), "this function expects tied weights, but embed_tokens and lm_head are different"
    for new_token, init_word in new_token_pairs:
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        init_word_id = tokenizer.convert_tokens_to_ids(init_word)
        if init_word_id is None:
            raise ValueError("init token was actually multiple tokens/an invalid token")

        # copy the init token embeddings to the new token
        init_embedding = model.model.embed_tokens.weight[init_word_id]
        model.model.embed_tokens.weight.data[new_token_id] = init_embedding
        print(f"Initialized '{new_token}' (id: {new_token_id}) with embedding of '{init_word}' (id: {init_word_id})")



def test_token_behavior(model, tokenizer, new_token_pairs: list[tuple[str, str]]):
    ''' Build and run a prompt using new tokens, and the tokens they were initialized from. They should produce identical outputs.'''
    # Smoke test
    print("Performing smoke test for new token embeddings weights...")
    for new_token, init_word in new_token_pairs:
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        init_word_id = tokenizer.convert_tokens_to_ids(init_word)

        # Check embedding similarities
        new_embedding = model.model.embed_tokens.weight.data[new_token_id]
        embed_similarities = F.cosine_similarity(new_embedding.unsqueeze(0), model.model.embed_tokens.weight, dim=1)
        print(f"\nSmoke test for '{new_token}':")

        # Check if new token matches itself in embeddings.
        #   Note: F.cosine_similarity is sometimes >1 (!?), maybe a bfloat16 issue? There's a pytorch issue tracking it
        assert (embed_similarities[new_token_id] >= 0.99).item(), f'new_token_id ({new_token}) is not self similar (weird), similarity={embed_similarities[new_token_id]}'
        assert (embed_similarities[init_word_id] >= 0.99).item(), f'new_token_id ({new_token}) is not similar to init_word_id ({init_word})'

        top_similarities, top_indices = embed_similarities.topk(6)
        print(f"  Top 5 similar tokens to '{new_token}':")
        for i, sim in zip(top_indices, top_similarities):
            token = tokenizer.convert_ids_to_tokens(i.item())
            print(f"    {token}: {sim.item():.4f}")

    # Integration test
    device = model.model.embed_tokens.weight.device
    new_ids  = torch.tensor([tokenizer.convert_tokens_to_ids(x[0]) for x in new_token_pairs], device=device).unsqueeze(0)
    orig_ids = torch.tensor([tokenizer.convert_tokens_to_ids(x[1]) for x in new_token_pairs], device=device).unsqueeze(0)
    attention_mask = torch.ones_like(new_ids).unsqueeze(0)
    with torch.no_grad():
        original_logits = model(input_ids=orig_ids, attention_mask=attention_mask).logits[0, -1]
        new_logits = model(input_ids=new_ids, attention_mask=attention_mask).logits[0, -1]
    assert torch.allclose(original_logits, new_logits), 'New tokens dont produce identical outputs as original tokens'
    print('New tokens pass the identical-output test')


##################################################
try:
    fail
    already_loaded
except:
    model = Q.Qwen2ForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add new tokens
    new_token_pairs = [("[A]", "cat"), ("[B]", "dog"), ("[C]", "bird")]
    add_and_initialize_tokens(model, tokenizer, new_token_pairs)
    test_token_behavior(model, tokenizer, new_token_pairs)

    already_loaded = True
# print_model_info(model)

prompt = "Once upon a time there was [A]. The noise a [A] makes is"
model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
out = model(**model_inputs, return_dict=True)  # type: CausalLMOutputWithPast
response = tokenizer.batch_decode(out.logits.argmax(dim=2), skip_special_tokens=False)[0]
print(response)


# Ex:
#
#   [BOQ] ^1 ^L | | | ^R | | [EOW]
#
new_tokens = [
    # Attention
    "[BOQ]",  # Q proj
    "[BOK]",  # K proj
    "[BOV]",  # V proj
    "[BOO]",  # O proj

    # MLP
    "[BOG]",  # Gate proj
    "[BOU]",  # Up proj
    "[BOD]",  # Down proj

    "[/EOW]",

    # Layer identifier
    "^1",
    "^2",
    "^3",

    # LoRA: y = Wx + LRx
    "^L",  # left singular vectors (column space, output space)
    "^R",  # right singular vectors (row space, input space)

    "load_weights",
    "do_metalearning",
    "remove_all_weights",

]
