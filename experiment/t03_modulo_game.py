'''

Palindrome & Sliding-window-add-and-modulo Game

----------
HOW THE GAME WORKS:

Window=2 Example Dataset
1 2 <- seed
1 2 3 5 8 3 [1 4] _ _ _ _

Window=3 Example Dataset
1 2 3 <- seed
1 2 3 6 1 0 7 8 [5 0 3] _ _ _ _


----------
LVL1: All data is generated with locked window size.

LVL2: Agent has to infer the correct window size.

'''

import torch
import random
from datasets import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import neurallambda.stack as S
import neurallambda.hypercomplex as H

# DATASET = 'modulo_game'
DATASET = 'palindrome'

BATCH_SIZE = 32

NUM_SAMPLES = 1000  # total number of samples in the dataset
MAX_WINDOW_SIZE = 4  # maximum window size
MAX_SEQUENCE_LENGTH = 10  # maximum length of the sequence

LR = 1e-2

NUM_EPOCHS = 50
GRAD_CLIP = 10
WARM_UP = 5 # ignore loss for this many steps


##########
# Project Ints to Vectors

DEVICE = 'cuda'
VEC_SIZE = 128
int_range_start = -200
int_range_end   =  200

# A matrix where each row ix represents the int, projected to VEC_SIZE
int_vecs = torch.stack([
    torch.randn((VEC_SIZE,))
    for _ in range(int_range_start, int_range_end + 1)
]).to(DEVICE)

def project_int(integer):
    """Projects an integer to a vector space."""
    index = integer - int_range_start
    return int_vecs[index]

def unproject_int(vector):
    """Unprojects a vector from the vector space back to an integer.

    Assumes matrix formatted `vector`.
    """
    cs = torch.cosine_similarity(vector.unsqueeze(0), int_vecs, dim=1)
    max_index = torch.argmax(cs).item()
    return max_index + int_range_start

# Test round tripping
for i in range(int_range_start, int_range_end):
    assert i == unproject_int(project_int(i))


##########
# Palindrome Data

# Function to generate palindrome synthetic data
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def generate_palindrome_synthetic_data(num_samples, max_length, reflect_token):
    REFLECT_SYMBOL = 42
    NULL_SYMBOL = 1
    data = []
    for _ in range(num_samples):
        length = random.randint(5, max_length)
        hlength = length // 2
        seed = [random.choice(nums) for _ in range(hlength)]
        inputs  = seed + [REFLECT_SYMBOL] + [0] * hlength
        outputs = [NULL_SYMBOL] * (hlength + 1) + seed[::-1]
        data.append({
            'seed': seed,
            'inputs': inputs,
            'outputs': outputs,
        })
    return data

if DATASET == 'palindrome':
    synthetic_data = generate_palindrome_synthetic_data(NUM_SAMPLES, MAX_SEQUENCE_LENGTH, reflect_token=42)
    train_size = int(0.8 * len(synthetic_data))
    train_data = synthetic_data[:train_size]
    val_data = synthetic_data[train_size:]


##########
# Modulo Data

def generate_sequence(seed, window_size, sequence_length):
    """
    Generates a sequence using a seed and a sliding window.

    The function starts with a seed, which is a list of initial numbers.
    It then extends this seed into a longer sequence by repeatedly summing the last 'window_size' numbers,
    and adding this sum (modulo 10) to the sequence.

    Args:
    seed (list): The initial numbers of the sequence.
    window_size (int): The number of elements to sum in each step.
    sequence_length (int): The total length of the sequence to generate.

    Returns:
    list: The generated sequence with the specified length.
    """
    sequence = seed.copy()
    while len(sequence) < sequence_length:
        window_sum = sum(sequence[-window_size:]) % 10
        sequence.append(window_sum)
    return sequence

# @@@@@@@@@@
assert generate_sequence([1, 2], 2, sequence_length=10) == [1, 2, 3, 5, 8, 3, 1, 4, 5, 9]
assert generate_sequence([1, 2, 3], 3, sequence_length=15) == [1, 2, 3, 6, 1, 0, 7, 8, 5, 0, 3, 8, 1, 2, 1]
# @@@@@@@@@@

def generate_synthetic_data(num_samples, max_window_size, max_sequence_length):
    data = []
    for _ in range(num_samples):
        window_size = random.randint(2, max_window_size)
        sequence_length = random.randint(window_size+1, max_sequence_length)
        seed = [random.randint(0, 9) for _ in range(window_size)]
        sequence = generate_sequence(seed, window_size, sequence_length)
        data.append({
            'seed': seed,
            'window_size': window_size,
            'sequence': sequence
        })
    return data

if DATASET == 'modulo_game':
    synthetic_data = generate_synthetic_data(NUM_SAMPLES, MAX_WINDOW_SIZE, MAX_SEQUENCE_LENGTH)
    train_size = int(0.8 * len(synthetic_data))
    train_data = synthetic_data[:train_size]
    val_data = synthetic_data[train_size:]


##############################
# Collate Datasets

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset   = Dataset.from_list(val_data)

# Define the padding vector as a zero vector with the same dimension as VEC_SIZE
zero_offset = 1e-3
padding_vec = torch.zeros(VEC_SIZE, device=DEVICE) + zero_offset

def collate_fn(batch):
    # Find the longest sequence in the batch
    max_seq_len = max(len(item['inputs']) for item in batch)

    # Initialize the list to store the padded vector inputss
    inputs_batch = []
    outputs_batch = []

    for item in batch:
        inputs = item['inputs']
        outputs = item['outputs']

        inputs = [project_int(i) for i in inputs]
        num_padding = max_seq_len - len(inputs)
        padded_inputs = [padding_vec] * num_padding + inputs

        outputs = [project_int(i) for i in outputs]
        num_padding = max_seq_len - len(outputs)
        padded_outputs = [padding_vec] * num_padding + outputs

        # Stack the vectors and add to the batch
        inputs_batch.append(torch.stack(padded_inputs))
        outputs_batch.append(torch.stack(padded_outputs))

    # Stack all the padded inputss into a single tensor
    inputs_tensor = torch.stack(inputs_batch)
    outputs_tensor = torch.stack(outputs_batch)

    # Return the tensor with the batch of padded inputss
    return inputs_tensor, outputs_tensor

# Create DataLoaders with the new collate_fn
train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
val_dl = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)


##################################################
# Models

##########
# RNN

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


##########
# LSTM

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lc1 = nn.LSTMCell(input_dim, hidden_dim)
        self.lc2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize the hidden and cell states to zeros
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        c1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []

        for i in range(x.size(1)):
            h1, c1 = self.lc1(x[:, i, :], (h1, c1))
            h1s.append(h1)

        # Run LSTM2
        h1s = torch.stack(h1s, dim=1) # input to lstm2

        h2 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        c2 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h2s = []

        for i in range(x.size(1)):
            h2, c2 = self.lc2(h1s[:, i, :], (h2, c2))
            h2s.append(h2)

        # Stack the outputs (hidden states) from each time step
        outputs = torch.stack(h2s, dim=1)

        # Apply the linear layer to each time step's output
        # We reshape the outputs tensor to (-1, self.hidden_dim) before applying the linear layer
        # Then reshape it back to (batch_size, seq_len, output_dim)
        out = self.fc(outputs.view(-1, self.hidden_dim))
        out = out.view(-1, x.size(1), self.output_dim)

        return out


##########
# Stack

def to_hyper(xs, number_system):
    ''' Reshape a Real vector to fit in a hypercomplex matrix form

    Args:
      xs: [BATCH, N]

    Returns:
      ndarray([BATCH, N/(dim**2), dim, dim])

    '''
    n = xs.shape[1]
    d = number_system.dim
    return number_system.to_mat(xs.reshape(-1, n // d, d))

def from_hyper(xs, number_system):
    ''' Convert a hypercomplex matrix form back to a vec '''
    return number_system.from_mat(xs).flatten(start_dim=-2, end_dim=-1)

class StackModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StackModel, self).__init__()

        ##########
        # Stack

        self.n_stack = 12
        self.initial_sharpen = 5
        self.number_system = H.Complex
        self.cdim = self.number_system.dim # complex dim
        self.stack_vec_size = input_dim // self.cdim

        self.should_push    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.should_pop     = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.should_null_op = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # self.should_push    = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.should_pop     = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.should_null_op = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        ##########
        # Net
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lc1 = nn.LSTMCell(
            input_dim + # input
            self.stack_vec_size * self.cdim,
            hidden_dim)
        self.lc2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Stack 1
        device = x.device
        batch_size = x.shape[0]
        vec_size = x.shape[2]
        stack = S.Stack(self.n_stack, self.stack_vec_size, self.number_system)
        stack.init(batch_size, self.initial_sharpen, zero_offset, device)
        # self.complex_proj = self.complex_proj.to(device)

        # Initialize the hidden and cell states to zeros
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        c1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []

        for i in range(x.size(1)):
            # s = stack.read().flatten(start_dim=-3, end_dim=-1).squeeze(-1)
            # s = self.number_system.from_mat(stack.read()).flatten(start_dim=-2, end_dim=-1)
            s = from_hyper(stack.read(), self.number_system)
            inp = x[:, i, :]
            h1, c1 = self.lc1(
                torch.concat([inp, s], dim=1),
                (h1, c1))
            h1s.append(h1)

            # Stack
            push = self.should_push(c1).squeeze(-1)
            pop = self.should_pop(c1).squeeze(-1)
            null_op = self.should_null_op(c1).squeeze(-1)

            # stack(push, pop, null_op, (self.complex_proj @ h1.T).reshape(batch_size, -1, self.cdim, self.cdim))
            stack(push, pop, null_op, to_hyper(inp, self.number_system))


        # Run LSTM2
        h1s = torch.stack(h1s, dim=1) # input to lstm2

        h2 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        c2 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h2s = []

        for i in range(x.size(1)):
            h2, c2 = self.lc2(h1s[:, i, :], (h2, c2))
            h2s.append(h2)

        # Stack the outputs (hidden states) from each time step
        outputs = torch.stack(h2s, dim=1)

        # Apply the linear layer to each time step's output
        # We reshape the outputs tensor to (-1, self.hidden_dim) before applying the linear layer
        # Then reshape it back to (batch_size, seq_len, output_dim)
        out = self.fc(outputs.view(-1, self.hidden_dim))
        out = out.view(-1, x.size(1), self.output_dim)

        return out


##################################################
# Training

def train_epoch(model, dl, optimizer, criterion, warm_up, device, clip):
    '''
    Args:
      warm_up: ignore loss for this many steps
    '''
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(dl):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src)

        loss = criterion(output[:, warm_up:], trg[:, warm_up:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dl)

def evaluate(model, dl, criterion, warm_up, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(dl):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src)

            loss = criterion(output[:, warm_up:], trg[:, warm_up:])
            epoch_loss += loss.item()

    return epoch_loss / len(dl)

def accuracy(model, val_dl, warm_up, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(val_dl):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src)
            predicted_vectors = output[:, warm_up:, :]

            # Unproject the predicted vectors to integers
            predicted_integers = [unproject_int(v) for v in torch.flatten(predicted_vectors, start_dim=0, end_dim=1)]
            target_integers = [unproject_int(v) for v in torch.flatten(trg[:, warm_up:], start_dim=0, end_dim=1)]

            # Compare the predicted integers with the actual integers
            total_predictions += len(predicted_integers)
            correct_predictions += (torch.tensor(predicted_integers) == torch.tensor(target_integers)).sum()
            # breakpoint()

    acc = correct_predictions / total_predictions
    return acc


##########
# Setup

# MyModel = RNNModel
# MyModel = LSTMModel
MyModel = StackModel

model = MyModel(
    input_dim=VEC_SIZE,
    hidden_dim=64,
    output_dim=VEC_SIZE,
)
model.to(DEVICE)

def criterion(x, target):
    return torch.mean(1 - torch.cosine_similarity(x, target, dim=2))

optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_dl, optimizer, criterion, WARM_UP, DEVICE, GRAD_CLIP)
    val_loss = evaluate(model, val_dl, criterion, WARM_UP, DEVICE)
    tacc = accuracy(model, train_dl, WARM_UP, DEVICE)
    vacc = accuracy(model, val_dl, WARM_UP, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(tacc)
    val_accuracies.append(vacc)

    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')

# Plot training and validation loss
plt.figure()
n = np.arange(len(train_losses))
plt.plot(n, train_losses, label='Train Loss')
plt.plot(n, val_losses, label='Val Loss')
plt.plot(n, train_accuracies, label='Train Acc')
plt.plot(n, val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
