'''.

Stabilize homoiconic approach for N-way k-shot learning task on simple Mixer model.

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


NaN ISSUES:
- rmsprop and adam have issues during the backward pass of the outerloop, and throw NaNs (at least I think this is where the NaNs are coming from. NaNs show up in weights after outer loop, so immediately bork the next fwd pass)


SPEEDING UP ITERATION:
- truncating model
- smaller N classes helps speed up iteration
- turn off tensorboard logging


STABILIZATION:
- sgd_momentum, lr=1.0, N_LOOP=20
- rmsprop and adam NaN issue
- rm Dropout in LORProject, helped a ton (DID IT?)
- randn up-down random intermediate seems important. Without, i can reach "better than chance" but even that's unstable. Using a projection instead of randn seems ok; starts off way worse so init matters, and is less stable.
- LORNorm: small init (normal_init * 1e-2) for RMSNorm helps a lot
- reducing MLP intermediate dims seems to help stability


DESIGN CHOICES:
- MLPMixer (rm dropout. intermediate size?. residuals? act fn? RMSNorm?)
- randn ud_intermediate in MLPMixer
- randn kv_intermediate in MLPMixer
- hyperparams (lr, wd, frozen params)
- metatokens as randn?  (embeddings seem to help, maybe)
- feedback updated metaweights each inner loop?
- img_proj: architecture and init
- LORNorm: keep norm in it?
- uncausal masking (support, metaweights, queries)
- targeted LOR_LAYER

TODO:
- [ ] optimize metaweights instead of LORProjected metaweights
- [ ] learnable inner loop lrs (perhaps different lrs per step)
- [ ] increase metaweight rank
- [ ] pass revised metaweights forward each inner loop

- [X] fix run_epoch
  - [X] update for new dataset
  - [X] add image projector
  - [X] tokenize labels
  - [X] append metatokens
  - [X] metalearn them
  - [X] prove loss can decrease for supports before worrying about queries
  - [X] test on queries

- options
  1. [ ] keep model frozen, only train LORProject
  2. [ ] use metalearned tokens in outerloop, and train entire model


EXPERIMENT SEQUENCE:
0. [ ] Overfit Simple Arch on 1 & Multi task
0. [ ] Overfit Simple Arch + Conv Encoder on 1 & Multi task
0. [ ] No LOR, overfit 1 task
1. [ ] No LOR, can bare transformer overfit training data?
2. [ ] LOR, but loss = equivalent in context loss (no query used)
3. [ ] LOR, Sham query without reordering
4. [ ] LOR, Sham with fixed reordering
5. [ ] LOR, Sham with random reordering
6. [ ] LOR, Query version


'''


import os
import math
import sys
import traceback
import importlib
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
from functools import partial
import itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt

from t13_metalearning_hypernet_data import omniglot_n_way_k_shot, ALPHABET_DICT
from neurallambda.lab.common import print_model_info


current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


LOG = False  # slows down training to ~0.5x

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'


##################################################
# Params

img_dim = 28

# data
train_alphabets = ["Latin", "Greek"]
test_alphabets = ["Mongolian"]
img_size = 28
n_way = 2  # N-way classification
k_shot = 1  # k-shot learning
q_query = 1  # query examples per class
num_tasks = 100  # number of tasks per epoch


##################################################

class Model(nn.Module):
    def __init__(self, dim_in, n_classes):
        super().__init__()
        idim = 512
        self.model = nn.Sequential(
            nn.Linear(dim_in, idim),
            nn.GELU(),
            nn.Linear(idim, n_classes, bias=False),
        )

    def forward(self, x):
        x = x.flatten(1, -1)
        out = self.model(x)
        return out


##################################################
# START_BLOCK_2

def mixer_layer_init(num_tokens, hidden_dim, token_dim, channel_dim):
    params = nn.ModuleDict({
        'token_mix_norm': nn.LayerNorm(hidden_dim),
        'token_mix': nn.ModuleList([
            nn.Linear(num_tokens, token_dim),
            nn.Linear(token_dim, num_tokens)
        ]),
        'channel_mix_norm': nn.LayerNorm(hidden_dim),
        'channel_mix': nn.ModuleList([
            nn.Linear(hidden_dim, channel_dim),
            nn.Linear(channel_dim, hidden_dim)
        ])
    })
    return params

def mixer_layer(x, params):
    # Token-mixing
    identity = x
    x = params['token_mix_norm'](x)
    x = x.transpose(1, 2)
    x = F.gelu(params['token_mix'][0](x))
    x = params['token_mix'][1](x)
    x = x.transpose(1, 2)
    x = x + identity

    # Channel-mixing
    identity = x
    x = params['channel_mix_norm'](x)
    x = F.gelu(params['channel_mix'][0](x))
    x = params['channel_mix'][1](x)
    x = x + identity

    return x

def mlp_mixer_init(dim_in, n_classes, patch_size=16, image_size=224,
                   num_blocks=8, hidden_dim=512, token_dim=256, channel_dim=2048):
    num_patches = (image_size // patch_size) ** 2
    patch_dim = dim_in * patch_size * patch_size

    params = nn.ModuleDict({
        'patch_embed': nn.Linear(patch_dim, hidden_dim),
        'mixer_blocks': nn.ModuleList([
            mixer_layer_init(num_patches, hidden_dim, token_dim, channel_dim)
            for _ in range(num_blocks)
        ]),
        'layer_norm': nn.LayerNorm(hidden_dim),
        'head': nn.Linear(hidden_dim, n_classes)
    })

    return params

def mlp_mixer(x, params, patch_size=16):
    # Calculate num_patches dynamically from input shape
    B, C, H, W = x.shape
    num_patches = (H // patch_size) ** 2

    # Reshape into patches and embed
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, -1)
    x = params['patch_embed'](x)

    # Apply mixer blocks
    for block_params in params['mixer_blocks']:
        x = mixer_layer(x, block_params)

    # Global average pooling and classification
    x = params['layer_norm'](x)
    x = x.mean(dim=1)
    x = params['head'](x)

    return x


#####
# Usage:

if False:
    # Initialize parameters for each item in batch
    batch_size = 2
    all_params = [
        mlp_mixer_init(
            dim_in=3,
            n_classes=10,
            patch_size=16,
            image_size=224,
            num_blocks=8,
            hidden_dim=512,
            token_dim=256,
            channel_dim=2048
        )
        for _ in range(batch_size)
    ]

    # Create a batch of inputs
    batch = torch.randn(batch_size, 3, 224, 224)

    # Process each item with its own parameters
    outputs = []
    for i in range(batch_size):
        output = mlp_mixer(
            batch[i:i + 1],  # Add batch dimension
            all_params[i],
            patch_size=16
        )
        outputs.append(output)

    # Combine outputs
    batch_output = torch.cat(outputs, dim=0)
    print(batch_output)
    print(batch_output.shape)

# END_BLOCK_2



##################################################
# Load Model

model = Model(img_dim ** 2, n_way)
model = model.to(DEVICE)
print_model_info(model)


##################################################
# Training


def run_epoch(model, img_proj, dataloader, optimizer, device, train=True, debug=False):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch_ix, batch in tqdm(enumerate(dataloader), desc="Training" if train else "Evaluating"):
            # supports: N*k tuples of batched images and labels
            # queries: N tuples (or N*q if multiple queries) of batched images and labels
            supports, queries = batch

            # Move to device, flatten and project images into embedding dim

            # unsqueeze channel dim
            # support_imgs = torch.stack([img_proj(x[0].to(device).unsqueeze(1)) for x in supports], dim=1).flatten(start_dim=1, end_dim=-1)  # N*k tensors [B, channel, IMG, IMG] -> [B, N*k, D]
            # support_labels = torch.stack([x[1].to(device) for x in supports], dim=1).flatten(start_dim=1, end_dim=-1)  # N*k tensors, shape=[B] -> [B, N*k]
            # query_imgs = torch.stack([img_proj(x[0].to(device).unsqueeze(1)) for x in queries], dim=1)  # N*k tensors [B, IMG, IMG] -> [B, N*k, D]
            # query_labels = torch.stack([x[1].to(device) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]

            B, Ss = support_labels.shape  # batch size, sequence (N*k)

            out = model(support_imgs)

            # calculate TRANSDUCTIVE loss, ie not autoregressive, ie don't offset logits/targets (and don't causal mask)
            logits = out.contiguous().view(-1, n_way)
            target = support_labels.view(-1)
            loss = F.cross_entropy(logits, target)

            if torch.isnan(loss):
                print('NaN encountered:')
                breakpoint()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples

    return avg_loss




##################################################
# Go

global_epoch = 0


# training
num_epochs = 1000
batch_size = 32
lr = 1e-5
wd = 0.0


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

parameters = [

{
    'params': model.parameters(),
    'lr': lr,
    'wd': 0.0
},

{
    'params': img_proj.parameters(),
    'lr': lr,
    'wd': wd
},

]

optimizer = optim.AdamW(parameters)

####################

train_losses = []
test_losses = []
best_loss = float('inf')


for epoch in range(num_epochs):
    global_epoch += 1

    model.train()
    train_loss = run_epoch(model, img_proj, train_dl, optimizer, DEVICE, train=True)
    train_losses.append((global_epoch, train_loss))

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = run_epoch(model, img_proj, test_dl, optimizer, DEVICE, train=False)
        test_losses.append((global_epoch, test_loss))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")



if num_epochs > 0:
    train_epochs, train_l = zip(*train_losses)
    test_epochs, test_l = zip(*test_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_l, label='Training Loss', color='blue')
    plt.plot(test_epochs, test_l, label='Test Loss', color='red')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
