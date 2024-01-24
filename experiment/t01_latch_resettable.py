'''

A Latch that's Resettable


1.) 3 values in:

  - Latch State
  - Enable: Set & Hold Symbols
  - Data: Arbitrary Vec

2. ) 1 value out:

  - Latch State

RESULTS: Learns perfectly, including for `data` that's out of domain.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import pdb
import torch
import numpy as np
import random
import string
from datasets import Dataset
from torch.utils.data.dataset import random_split
from torch import cosine_similarity

torch.set_printoptions(precision=3)
print('\n'*10)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = 'cpu'


##########
# Params

NUM_EPOCHS = 100
BATCH_SIZE = 10

INIT_SCALE = 1e-3
LR = 1e-3
WD = 0

N_DATASET_POS = 1000
N_DATASET_NEG = 1000
PERCENT_VAL = 0.5

VEC_SIZE = 128
DTYPE = torch.float32


##########
# Data

enable_vec = torch.randn(VEC_SIZE, )

pos_dataset = []
for _ in range(N_DATASET_POS):
    state = torch.randn(VEC_SIZE,)
    enable = enable_vec
    data = torch.randn(VEC_SIZE,)
    expected_output = data
    pos_dataset.append(
        (state, enable, data, expected_output)
    )

neg_dataset = []
for _ in range(N_DATASET_NEG):
    state = torch.randn(VEC_SIZE,)
    enable = torch.randn(VEC_SIZE,) # ie not `enable_vec`
    data = torch.randn(VEC_SIZE,)
    expected_output = state
    neg_dataset.append(
        (state, enable, data, expected_output)
    )

dataset = pos_dataset + neg_dataset
random.shuffle(dataset)

total_size = len(dataset)
split_size = int(total_size * PERCENT_VAL)
train_size = total_size - split_size
validation_size = split_size

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False)


##########
# Model

class Latch(nn.Module):
    def __init__(self, ):
        super(Latch, self).__init__()
        self.enable = nn.Parameter(torch.randn(VEC_SIZE, ) * INIT_SCALE)
        self.dropout = torch.nn.Dropout(0.01)

    def forward(self, state, enable, data):
        # state  : [batch_size, vec_size]
        # enable : [batch_size, vec_size]
        # data   : [batch_size, vec_size]

        matched = cosine_similarity(self.enable.unsqueeze(0), enable, dim=1)
        matched = torch.nn.functional.elu(matched)
        # matched = torch.nn.functional.selu(matched)
        # matched = torch.nn.functional.gelu(matched)
        # matched = torch.nn.functional.leaky_relu(matched)

        if_matched      = einsum('bv, b -> bv', data, matched)
        if_not_matched  = einsum('bv, b -> bv', state, 1-matched)

        out = if_matched + if_not_matched
        return self.dropout(out)

def run_epoch(data_loader, model, optimizer, device=DEVICE):
    model.train()
    total_loss = 0

    for batch in data_loader:
        state, enable, data, expected_output = [item.to(device) for item in batch]
        model.zero_grad()
        output = model(state, enable, data)
        loss = (1 - cosine_similarity(expected_output, output, dim=1)).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def accuracy(model, validation_loader, thresh=0.5, device=DEVICE):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for batch in validation_loader:
            state, enable, data, expected_output = [item.to(device) for item in batch]
            predictions = model(state, enable, data)
            sim_scores = cosine_similarity(predictions, expected_output, dim=1)
            correct += (sim_scores > thresh).sum().item()
            total += state.size(0)
    return correct / total


##########
# Go

model = Latch()
model = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

for epoch in range(NUM_EPOCHS):
    loss = run_epoch(train_loader, model, optimizer)
    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        print(f'Epoch {epoch}, Training Loss: {loss:.9f}')

    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        validation_accuracy = accuracy(model, validation_loader, thresh=0.95, device=DEVICE)
        print(f'Validation Accuracy: {validation_accuracy:.2f}')

print('ENABLE:', cosine_similarity(model.enable, enable_vec, dim=0))
print('VAL_LOSS:', run_epoch(validation_loader, model, optimizer))
