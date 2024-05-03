'''

Palindrome

RESULTS:
* Validation scores frequently surpass training scores
* Seems a little sensitive to the RNG seed

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
import neurallambda.latch as L
from torch import cosine_similarity, einsum
from torch.nn.functional import elu, selu, gelu, leaky_relu

torch.manual_seed(11)

DEBUG = False

DATASET = 'palindrome'

BATCH_SIZE = 32

NUM_SAMPLES = 1000  # total number of samples in the dataset
MAX_SEQUENCE_LENGTH = 10  # maximum length of the sequence

LR = 1e-2

NUM_EPOCHS = 8
GRAD_CLIP = 10


##########
# Project Ints to Vectors

DEVICE = 'cuda'
VEC_SIZE = 2048
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
REFLECT_SYMBOL = 42
NULL_SYMBOL = 0
PADDING_SYMBOL = -1

nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def generate_palindrome_synthetic_data(num_samples, max_length, reflect_token):
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


synthetic_data = generate_palindrome_synthetic_data(NUM_SAMPLES, MAX_SEQUENCE_LENGTH, reflect_token=42)
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
# padding_vec = torch.zeros(VEC_SIZE, device=DEVICE) + zero_offset
padding_vec = project_int(PADDING_SYMBOL)

def collate_fn(batch):
    # Find the longest sequence in the batch
    max_seq_len = max(len(item['inputs']) for item in batch)

    # Initialize the list to store the padded vector inputss
    loss_mask_batch = []
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

        half = (len(outputs) - 1) // 2
        loss_mask = [0] * num_padding + [0] * (half + 1) + [1.0] * half

        # Stack the vectors and add to the batch
        inputs_batch.append(torch.stack(padded_inputs))
        outputs_batch.append(torch.stack(padded_outputs))
        loss_mask_batch.append(torch.tensor(loss_mask))

    # Stack all the padded inputss into a single tensor
    inputs_tensor = torch.stack(inputs_batch)
    outputs_tensor = torch.stack(outputs_batch)
    loss_mask_tensor = torch.stack(loss_mask_batch)

    # Return the tensor with the batch of padded inputss
    return inputs_tensor, outputs_tensor, loss_mask_tensor

# Create DataLoaders with the new collate_fn
train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
val_dl = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# for i, x in enumerate(train_dl):
#     if i>5:
#         break
#     print(x)


# for i, x in enumerate(val_dl):
#     if i>5:
#         break
#     print(x)


##########
# Debugging

def print_grid(data, labels=None):
    data = list(data)
    column_widths = []
    max_columns = 0

    # Calculate the max number of columns in any row
    for row in data:
        for column in row:
            max_columns = max(max_columns, len(str(column)))

    # Initialize column widths to 0
    column_widths = [0] * max_columns

    # Update column widths based on the data
    for i, row in enumerate(data):
        for j, column in enumerate(row):
            for k, item in enumerate(column):
                if isinstance(item, float):
                    w = 6  # assumes fmt :>.2f
                else:
                    w = len(str(item))
                column_widths[k] = max(column_widths[k], w)

    # Print the grid with aligned columns
    max_label = max(map(len, labels))
    for row in data:
        if labels is None:
            labels = [''] * len(row)
        for lab, column in zip(labels, row):
            print(f"{lab.rjust(max_label)}", end=" ")
            for i, item in enumerate(column):
                if isinstance(item, float):
                    x = f'{item:>.2f}'
                else:
                    x = str(item)
                print(f"{x.rjust(column_widths[i])}", end=" ")
            print()
        print("-" * (sum(column_widths) + max_label + 1))  # Separator between different sets

if False:
    for ins, outs, mask in train_dl:
        assert ins.shape[0] == BATCH_SIZE
        assert ins.shape[2] == VEC_SIZE
        assert outs.shape[0] == BATCH_SIZE
        assert outs.shape[2] == VEC_SIZE
        assert mask.shape[0] == BATCH_SIZE
        assert mask.shape[1] == ins.shape[1] == outs.shape[1]
        # print()
        # for i, o, m in zip(ins, outs, mask):
        #     print()
        #     print([unproject_int(x) for x in i])
        #     print([unproject_int(x) for x in o])
        #     print(m.tolist())

        print()
        ins = [[unproject_int(x.to('cuda')) for x in xs] for xs in ins]
        outs = [[unproject_int(x.to('cuda')) for x in xs] for xs in outs]
        mask = [x.to(int).tolist() for x in mask]
        xx = zip(ins, outs, mask)
        print_grid(xx)
        BRK


##################################################
# Models


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

        return out, {}


##########
# NeuralStack Only

class NeuralstackOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralstackOnly, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ##########
        # Stack
        self.n_stack = 12
        self.initial_sharpen = 5
        self.stack_vec_size = input_dim
        self.sharpen_pointer = nn.Parameter(torch.tensor([5.0]))

        ##########
        # Latch
        self.latch = L.DataLatch(input_dim, init_scale=1e-2, dropout=0.02)

        ##########
        # NNs

        HIDDEN_DIM = 128

        self.should_pop = nn.Parameter(torch.randn(input_dim))
        self.null_symbol = nn.Parameter(torch.randn(output_dim))
        self.dropout = torch.nn.Dropout(0.02)

        # with torch.no_grad():
        #     print()
        #     print('CHEATING')
        #     self.latch.enable[:] = project_int(REFLECT_SYMBOL)
        #     self.should_pop[:] = project_int(REFLECT_SYMBOL)
        #     # self.null_symbol[:] = project_int(NULL_SYMBOL)

    def forward(self, x):
        # Stack Init
        device = x.device
        batch_size = x.shape[0]
        vec_size = x.shape[2]
        dtype = x.dtype

        # self.stack.init(batch_size, zero_offset, device)
        ss = S.initialize(self.input_dim, self.n_stack, batch_size, zero_offset, device, dtype=dtype)

        # Latch Init
        # latch_state = torch.zeros((batch_size, self.input_dim), device=device) + zero_offset
        latch_state = torch.randn((batch_size, self.input_dim), device=device)

        outputs = []

        # Debugging
        latch_states = []
        pops = []
        stack_tops = []

        relay = [latch_state, ]

        # Loop over sequence
        for i in range(x.size(1)):
            inp = x[:, i, :]

            lsr = relay[0]

            # Decide what to do
            pop = cosine_similarity(self.should_pop.unsqueeze(0), lsr)
            pop = elu(pop)
            pop = self.dropout(pop)
            # pop = gelu(pop)
            # pop = leaky_relu(pop)
            pops.append(pop)

            push    = 1 - pop
            null_op = torch.zeros_like(pop)

            # Update Stack
            ss, pop_val = S.push_pop_nop(ss, self.sharpen_pointer, push, pop, null_op, inp)
            s = S.read(ss)

            out = (
                einsum('b, bv -> bv', pop, s)
                # einsum('b,  v -> bv', 1 - pop, self.null_symbol)
            )
            # out = self.dropout(out)
            outputs.append(out)

            if DEBUG and False:
                print()

                lat = unproject_int(latch_state[0])
                lat_sim = cosine_similarity(latch_state[0], project_int(lat), dim=0).item()
                print(f'lat: {lat}  {lat_sim:>.3f}')
                print('inp:', unproject_int(x[0, i]))
                print('out:', unproject_int(out[0]))



            # Update Latch
            latch_state = self.latch(latch_state, enable=inp, data=inp)
            latch_states.append(latch_state)
            relay.append(latch_state)
            relay = relay[1:]
            stack_tops.append(S.read(ss))

        outputs = torch.stack(outputs, dim=1)

        return outputs, {
            'latch': {'data':torch.stack(latch_states, dim=1), 'fn': unproject_int},
            'pops': {'data':torch.stack(pops, dim=1), 'fn': lambda x: x.tolist()},
            'stack': {'data': torch.stack(stack_tops, dim=1), 'fn': unproject_int},
        }


##################################################
# Training

def train_epoch(model, dl, optimizer, device, clip):
    '''
    Args:
      warm_up: ignore loss for this many steps
    '''
    model.train()
    epoch_loss = 0

    for i, (src, trg, mask) in enumerate(dl):
        src = src.to(device)   # shape=[batch, seq, vec_size]
        trg = trg.to(device)   # shape=[batch, seq, vec_size]
        mask = mask.to(device) # shape=[batch, seq]

        optimizer.zero_grad()
        output, latch_states = model(src) # shape=[batch, seq, vec_size]

        # loss = criterion(output, trg) * mask
        loss = ((1 - torch.cosine_similarity(output, trg, dim=2)) * mask).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dl)

def evaluate(model, dl, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg, mask) in enumerate(dl):
            src = src.to(device)
            trg = trg.to(device)
            mask = mask.to(device)

            output, latch_states = model(src)

            # loss = criterion(output, trg) * mask
            loss = ((1 - torch.cosine_similarity(output, trg, dim=2)) * mask).mean()
            epoch_loss += loss.item()

    return epoch_loss / len(dl)

def accuracy(model, val_dl, device, debug=False):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i, (src, trg, mask) in enumerate(val_dl):
            src = src.to(device)
            trg = trg.to(device)
            mask = mask.to(device)
            mask_ix = mask > 0.00000001

            output, states = model(src)
            predicted_vectors = output

            # Unproject the predicted vectors to integers
            predicted_integers = [unproject_int(v) for v in predicted_vectors[mask_ix]]
            target_integers = [unproject_int(v) for v in trg[mask_ix]]

            # Compare the predicted integers with the actual integers
            total_predictions += len(predicted_integers)
            correct_predictions += (torch.tensor(predicted_integers) == torch.tensor(target_integers)).sum()

            if debug:
                labels = ['inps', 'trgs', 'outs', 'mask']
                ins  = [[unproject_int(x.to('cuda')) for x in xs] for xs in src]
                trgs = [[unproject_int(x.to('cuda')) for x in xs] for xs in trg]
                outs = [[unproject_int(x.to('cuda')) for x in xs] for xs in predicted_vectors]
                mask = [x.to(int).tolist() for x in mask]

                ss = []
                for k,v in states.items():
                    labels.append(k)
                    data = v['data']
                    fn = v['fn']
                    if fn is not None:
                        ss.append([[fn(x.to('cuda')) for x in xs] for xs in data])
                    else:
                        ss.append(data)
                all_seqs = [ins, trgs, outs, mask] + ss
                print_grid(zip(*all_seqs), labels)

    acc = correct_predictions / total_predictions
    return acc


##########
# Setup

# MyModel = LSTMModel
MyModel = NeuralstackOnly

model = MyModel(
    input_dim=VEC_SIZE,
    hidden_dim=64,
    output_dim=VEC_SIZE,
)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_dl, optimizer, DEVICE, GRAD_CLIP)
    val_loss = evaluate(model, val_dl, DEVICE)
    tacc = accuracy(model, train_dl, DEVICE)
    vacc = accuracy(model, val_dl, DEVICE)

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


##########

tacc = accuracy(model, val_dl, DEVICE, debug=True)
