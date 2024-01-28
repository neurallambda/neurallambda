'''WHAT I LEARNED DEBUGGING THIS.

In the end, I had a shape misalignment fed into cosine_similarity, which a torch broadcast hid from erroring.

But in the process, I learned:

- cosine_sim can obvious go negative. This can cause the `predicate` to go
  negative, and then the `true` param also goes negative, but it seems like
  there's a local minima there that hampers learning, and ideally it would just
  learn the positive version of a symbol.

- Things that don't particularly help that: .abs(), .clip(0,1), sim**2

- Things that DO HELP: .clip(-0.1, 1), but better is torch.nn.functional.{elu, selu, leaky_relu, gelu}

- The Training Loss can be reduced beyond the Analytical Solution, especially by
  getting weird superpositions of true + false, and a negative predicate. This
  doesn't seem to hurt accuracy, but likely *would* for bigger systems.

- Light Dropout is better than any other regularization I tried. Note, it
  severely damages training loss, BUT in a good way. Since I know the analytical
  solution, and therefore know the "loss floor", I should question when the loss
  goes below that floor. Even light dropout prevents that (eg
  Dropout(0.001)). Note, `matched.leaky_relu` kinda sucks on its own, but with
  dropout is fine.

- Regularization that failed: inhibitory-surround type things, pushing params away from known vecs, etc.

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

# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

DEVICE = 'cpu'
# DEVICE = 'cuda'

##########
# Params

NUM_EPOCHS = 100
BATCH_SIZE = 10

INIT_SCALE = 1e-3
LR = 1e-2
WD = 0

N_DATASET_POS = 100
N_DATASET_NEG = 1000
PERCENT_VAL = 0.5

VEC_SIZE = 512
DTYPE = torch.float32


##########
# Data

# We'll try and find these same vectors being learned within the network.
predicate_vec = torch.randn(VEC_SIZE, dtype=DTYPE)
true_vec = torch.randn(VEC_SIZE, dtype=DTYPE)
false_vec = torch.randn(VEC_SIZE, dtype=DTYPE)

pos_dataset = [('true', predicate_vec, true_vec) for _ in range(N_DATASET_POS)]
neg_dataset = [('false', torch.randn(VEC_SIZE, dtype=DTYPE), false_vec) for _ in range(N_DATASET_NEG)]
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

class Sim(nn.Module):
    def __init__(self, ):
        super(Sim, self).__init__()
        self.predicate = nn.Parameter(torch.randn(VEC_SIZE, dtype=DTYPE) * INIT_SCALE)
        self.true      = nn.Parameter(torch.randn(VEC_SIZE, dtype=DTYPE) * INIT_SCALE)
        self.false     = nn.Parameter(torch.randn(VEC_SIZE, dtype=DTYPE) * INIT_SCALE)

    def forward(self, inp):
        # inp  : [batch_size, vec_size]
        # output : [batch_size, vec_size]
        predicate = self.predicate.unsqueeze(0)
        matched = cosine_similarity(predicate, inp, dim=1)

        # matched = matched.real
        matched = torch.nn.functional.elu(matched)
        # matched = torch.nn.functional.selu(matched)
        # matched = torch.nn.functional.leaky_relu(matched)  # max(0.1 * x, x)


        if_matched = einsum('v, b -> bv', self.true, matched)
        if_not_matched = einsum('v, b -> bv', self.false, 1 - matched)

        out = if_matched + if_not_matched

        return out
        # return self.dropout(out)

def regularize_symbols(symbols, device):
    loss = torch.tensor(0.0, device=device)
    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols):
            if i == j:
                continue # skip self compared to self
            loss += cosine_similarity(s1, s2, dim=0).abs()

    loss /= (len(symbols) ** 2 - len(symbols))
    return loss

def run_epoch(data_loader, model, optimizer):
    model.train()
    total_loss = 0
    all_predictions = []
    all_true_values = []

    for batch in data_loader:
        labels, input_tensor, target_tensor = batch
        input_tensor = input_tensor.to(DEVICE)
        target_tensor = target_tensor.to(DEVICE)

        model.zero_grad()
        output = model(input_tensor)

        # Cos sim Loss
        loss = (1 - cosine_similarity(target_tensor, output)).sum() # mean()

        # # MSE
        # loss = torch.nn.functional.mse_loss(target_tensor, output)

        # regularization
        loss += torch.nn.functional.leaky_relu(cosine_similarity(model.true, model.false, dim=0)) * 1e-2
        # loss += cosine_similarity(model.true, false_vec, dim=0).abs() * 1e-2
        # loss += cosine_similarity(model.false, true_vec, dim=0).abs() * 1e-2

        # # params should not be close to superposition
        # if epoch < 95:
        #     loss += (cosine_similarity(model.true,  true_vec + false_vec) / 10).clip(0, 1)
        #     loss += (cosine_similarity(model.false, true_vec + false_vec) / 10).clip(0, 1)

        # # regularization
        # loss += regularize_symbols([
        #     # model.predicate,
        #     model.true,
        #     model.false,
        # ], DEVICE) * 1e-3

        loss = loss.real
        loss.backward()

        # if epoch < 10:
        #     noise_grads(model)

        optimizer.step()

        total_loss += loss.item() / target_tensor.size(1)

    return total_loss / len(data_loader)


def accuracy(model, validation_loader, thresh=0.5, device='cpu'):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients
        for batch in validation_loader:
            labels, inputs, true_outputs = batch
            inputs, true_outputs = inputs.to(device), true_outputs.to(device)

            # Forward pass to get outputs
            predictions = model(inputs)

            # Compute cosine similarity and compare with a threshold
            sim_scores = cosine_similarity(predictions, true_outputs, dim=1)
            sim_scores = sim_scores.real
            correct += (sim_scores > thresh).sum().item()
            total += inputs.size(0)

    return correct / total

##########
# Go

model = Sim()
model = model.to(DEVICE)

##########
# Cheat and set to expected value

# with torch.no_grad():
#     model.predicate[:] = predicate_vec
#     model.true[:] = true_vec
#     model.false[:] = false_vec
#     print('SETTING CHEAT')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=1e-2)

def print_grid(similarities, from_labels, to_labels):
    '''Prints a grid of similarities with headers and row labels.'''
    # Calculate column widths based on the formatted similarities
    col_widths = [max(len(f"{sim:.2f}") for sim in col) for col in zip(*similarities)]

    # Print header with appropriate spacing
    header = '     ' + ' '.join(f'{label:>{w}}' for label, w in zip(to_labels, col_widths))
    print(header)
    # Print separator
    print('    ' + '-' * (len(header) - 4))

    # Print each row with the corresponding from_label
    for i, row in enumerate(similarities):
        row_str = ' '.join(f'{sim:>{w}.2f}' for sim, w in zip(row, col_widths))
        print(f'{from_labels[i]:>3}: {row_str}')

def debug_symbol_similarities(symbols, labels):
    sims = []
    for s1 in symbols:
        row_sims = []
        for s2 in symbols:
            sim = cosine_similarity(s1, s2, dim=0).item()
            row_sims.append(sim)
        sims.append(row_sims)

    print_grid(sims, labels, labels)

def compare_vectors(froms, tos, device):
    '''Compares each 'from' vector with each 'to' vector and prints a grid of similarities.'''
    device = torch.device(device)
    similarities = []
    for _, from_vec in froms:
        similarities_row = []
        for _, to_vec in tos:
            similarity = cosine_similarity(from_vec.to(device), to_vec.to(device), dim=0)
            similarities_row.append(similarity.item())  # Convert to Python float for printing
        similarities.append(similarities_row)

    from_labels = [label for label, _ in froms]
    to_labels = [label for label, _ in tos]
    print_grid(similarities, from_labels, to_labels)

def run_checks():
    print()

    # print('pos ix:')
    # ix = predicate_vec > 0
    # # ix = torch.logical_and(model.predicate > 0, predicate_vec > 0)
    # froms = [("p", model.predicate[ix]), ("t", model.true[ix]), ("f", model.false[ix]),]
    # tos = [("p", predicate_vec[ix]), ("t", true_vec[ix]), ("f", false_vec[ix]),]
    # compare_vectors(froms, tos, 'cpu')
    # print()

    # print('neg ix:')
    # ix = model.predicate < 0
    # # ix = torch.logical_and(model.predicate < 0, predicate_vec < 0)
    # froms = [("p", model.predicate[ix]), ("t", model.true[ix]), ("f", model.false[ix]),]
    # tos = [("p", predicate_vec[ix]), ("t", true_vec[ix]), ("f", false_vec[ix]),]
    # compare_vectors(froms, tos, 'cpu')
    # print()

    print('full ix:')
    froms = [("p", model.predicate), ("t", model.true), ("f", model.false), ("t+f", model.true + model.false),]
    tos = [("p", predicate_vec), ("t", true_vec), ("f", false_vec), ("t+f", true_vec + false_vec)]
    compare_vectors(froms, tos, 'cpu')
    print()

    validation_accuracy = accuracy(model, validation_loader, thresh=0.95, device=DEVICE)
    print(f'Validation Accuracy: {validation_accuracy:.2f}')


    print()
    print('Symbol Similarities:')
    debug_symbol_similarities([model.predicate, model.true, model.false], ['p', 't', 'f'])
    print()


for epoch in range(NUM_EPOCHS):
    if epoch % 100 == 0:
        run_checks()

    loss = run_epoch(train_loader, model, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss:>.9f}')

run_checks()


# with torch.no_grad():
#     for label, inp, expected in validation_loader:
#         matched = cosine_similarity(model.predicate, inp, dim=1)
#         true = model.true
#         false = model.false
#         if_matched = einsum('v, b -> bv', true, matched)
#         if_not_matched = einsum('v, b -> bv', false, 1 - matched)
#         out = if_matched + if_not_matched
#         print(1 - cosine_similarity(out, expected))
