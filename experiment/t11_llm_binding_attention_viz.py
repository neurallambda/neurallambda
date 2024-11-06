'''

Visualizing Transformer Latent Embeddings via LLE and t-SNE

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

import numpy as np
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import matplotlib.pyplot as plt


SEED = 152
torch.manual_seed(152)
random.seed(SEED)

DEVICE = 'cuda'
BATCH_SIZE = 32

model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")


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

    train_size = 10000 - 2
    val_size = 2
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



##################################################
# Viz

layer_idx = 0
final_q = [x[:, -3] for x in trace_outs[layer_idx]['q']]
q_data = torch.cat(final_q, dim=0)

data = q_data.detach().float().cpu().numpy()

# Perform t-SNE
print('training TSNE')
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data)

# Perform LLE
print('training LLE')
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
lle_results = lle.fit_transform(data)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot t-SNE results
ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, color='black', s=10, linewidths=0)
ax1.set_title('t-SNE Visualization')
ax1.set_xlabel('t-SNE feature 1')
ax1.set_ylabel('t-SNE feature 2')

# Plot LLE results
ax2.scatter(lle_results[:, 0], lle_results[:, 1], alpha=0.5, color='black', s=10, linewidths=0)
ax2.set_title('LLE Visualization')
ax2.set_xlabel('LLE feature 1')
ax2.set_ylabel('LLE feature 2')

plt.tight_layout()
plt.show()



##################################################
#

# Choose algorithm: 'tsne' or 'lle'
alg = 'lle'  # Change this to 'lle' for LLE visualization

# Assuming trace_outs is a list or dictionary containing the layer outputs
num_layers = len(trace_outs)

fig, axes = plt.subplots(7, 4, figsize=(20, 35))
fig.suptitle(f'{alg.upper()} Visualizations Across Layers', fontsize=16)

for layer_idx in tqdm(list(range(num_layers))):
    row = layer_idx // 4
    col = layer_idx % 4

    final_q = [x[:, -3] for x in trace_outs[layer_idx]['q']]
    q_data = torch.cat(final_q, dim=0)

    # print('truncating data for faster dev')
    # q_data = q_data[:100]

    data = q_data.detach().float().cpu().numpy()

    if alg == 'tsne':
        # Perform t-SNE
        reducer = TSNE(n_components=2, random_state=42)
    elif alg == 'lle':
        # Perform LLE
        reducer = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42, eigen_solver='dense')
    else:
        raise ValueError("Invalid algorithm choice. Use 'tsne' or 'lle'.")

    results = reducer.fit_transform(data)

    # Plot results
    ax = axes[row, col]
    ax.scatter(results[:, 0], results[:, 1], alpha=0.5, color='black', s=10, linewidths=0)
    ax.set_title(f'Layer {layer_idx}: {alg.upper()}')
    ax.set_xlabel(f'{alg.upper()} feature 1')
    ax.set_ylabel(f'{alg.upper()} feature 2')

# Remove any unused subplots
for i in range(num_layers, 28):
    row = i // 4
    col = i % 4
    fig.delaxes(axes[row, col])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
