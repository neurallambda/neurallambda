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
torch.set_printoptions(precision=3)

# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

DEVICE = 'cuda'

##########
# Params

NUM_EPOCHS = 2000
BATCH_SIZE = 10
GRAD_CLIP = 10.0

LR = 1e-2
WD = 0

N_DATASET_POS = 100
N_DATASET_NEG = 100
VEC_SIZE = 512
INIT_SCALE = 1e-4


##########
# Data

# We'll try and find these same vectors being learned within the network.
predicate_vec = torch.randn(VEC_SIZE)
true_vec = torch.randn(VEC_SIZE)
false_vec = torch.randn(VEC_SIZE)

dataset = (
    # positives
    [(predicate_vec, true_vec) for _ in range(N_DATASET_POS)] +
    # negatives
    [(torch.randn(VEC_SIZE), false_vec) for _ in range(N_DATASET_POS)]
)

dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

##########
# Model


class Sim(nn.Module):
    def __init__(self, ):
        super(Sim, self).__init__()
        self.predicate = nn.Parameter(torch.randn(VEC_SIZE) * INIT_SCALE)
        self.true = nn.Parameter(torch.randn(VEC_SIZE) * INIT_SCALE)
        self.false = nn.Parameter(torch.randn(VEC_SIZE) * INIT_SCALE)

    def forward(self, input):
        # input  : [batch_size, vec_size]
        # output : [batch_size, vec_size]
        batch_size = input.size(0)
        predicate = self.predicate.unsqueeze(0)
        matched = torch.cosine_similarity(predicate, input, dim=1)
        return (
            einsum('v, b -> bv', self.true, matched) +
            einsum('v, b -> bv', self.false, 1 - matched)
        )

def run_epoch(data_loader, model, optimizer):
    model.train()
    total_loss = 0
    all_predictions = []
    all_true_values = []

    for batch in data_loader:
        input_tensor, target_tensor = batch
        input_tensor = input_tensor.to(DEVICE)
        target_tensor = target_tensor.to(DEVICE)

        model.zero_grad()
        output = model(input_tensor)
        loss = (1 - torch.cosine_similarity(target_tensor, output.unsqueeze(1))).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() / target_tensor.size(1)

    return total_loss / len(data_loader)


##########
# Go

model = Sim()
model = model.to(DEVICE)

##########
# Cheat and set to expected value

with torch.no_grad():
    model.predicate[:] = predicate_vec
    # model.true[:] = true_vec
    # model.false[:] = false_vec

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

def compare_vectors(froms, tos, device):
    ''' Compares each 'from' vector with each 'to' vector and prints a grid of similarities. '''
    # Ensure device is a valid torch device
    device = torch.device(device)

    # Calculate similarities and store in a 2D list
    similarities = []
    for from_label, from_vec in froms:
        similarities_row = []
        for to_label, to_vec in tos:
            similarity = torch.cosine_similarity(from_vec.to(device), to_vec.to(device), dim=0)
            similarities_row.append(similarity.item())  # Convert to Python float for printing
        similarities.append(similarities_row)

    # Get the labels for rows and columns from the tuples
    from_labels = [label for label, _ in froms]
    to_labels = [label for label, _ in tos]

    # Print the grid
    # Print header
    print('    ' + ' '.join([f'{label:>5}' for label in to_labels]))
    # Print each row with the corresponding from_label
    for i, row in enumerate(similarities):
        row_str = ' '.join([f'{sim:5.2f}' for sim in row])  # Format each similarity to two decimal places
        print(f'{from_labels[i]:>3}: {row_str}')


for epoch in range(NUM_EPOCHS):
    loss = run_epoch(dataset_loader, model, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss:>.9f}')

    if epoch % 100 == 0:
        print('checking:')
        froms = [("p", model.predicate), ("t", model.true), ("f", model.false)]
        tos = [("p", predicate_vec), ("t", true_vec), ("f", false_vec)]
        compare_vectors(froms, tos, 'cpu')
