'''

Variable Time Computation!

Test Variable Time Computation by building an LM that can process strings of numbers added together

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
import neurallambda.queue as Q
from torch import cosine_similarity, einsum
from torch.nn.functional import elu, selu, gelu, leaky_relu
import neurallambda.symbol as Sym
import copy

torch.manual_seed(152 + 1)

DEBUG = False

PAUSE_IRRADIATION_P = 0.1
PAUSE_NUM_APPEND = 3
PAUSE_APPEND_INDEX = -3  # append tokens before [..., L, 42, R]

DEVICE = 'cuda'
VEC_SIZE = 512

TRAIN_SPLIT = 0.5
BATCH_SIZE = 250

NUM_SAMPLES = 2000  # total number of samples in the dataset
MAX_SEQUENCE_LENGTH = 10  # maximum length of the sequence

LR = 5e-2

NUM_EPOCHS = 200


##################################################
# Pause Tokens

def irradiate_tokens(data, percentage, symbol, null_symbol):
    """Injects pause tokens at random positions in the data sequences based on a percentage of the sequence length."""
    for item in data:
        num_tokens_to_inject = max(1, int(len(item['inputs']) * percentage))
        for _ in range(num_tokens_to_inject):
            inject_pos = random.randint(0, len(item['inputs']) - 1)
            item['inputs'].insert(inject_pos, symbol)
            item['outputs'].insert(inject_pos, null_symbol)  # Maintain alignment
            item['loss_mask'].insert(inject_pos, 0)  # No loss for injected tokens
    return data

def insert_at_ix(data, num_tokens, index, symbol, null_symbol):
    """Inserts a block of pause tokens at a specified index in the data sequences."""
    for item in data:
        item['inputs'] = item['inputs'][:index] + [symbol] * num_tokens + item['inputs'][index:]
        item['outputs'] = item['outputs'][:index] + [null_symbol] * num_tokens + item['outputs'][index:]
        item['loss_mask'] = item['loss_mask'][:index] + [0] * num_tokens + item['loss_mask'][index:]
    return data


# @@@@@@@@@@

#  Test irradiate_tokens
data_sample = [{'inputs': ['1', '2', '3'], 'outputs': ['N', 'N', '6'], 'loss_mask': [0, 0, 1]}]
percentage = 0.5  # 50% of the sequence length

modified_data = irradiate_tokens(data_sample.copy(), percentage, 'T', 'N')

expected_length = len(data_sample[0]['inputs']) + int(len(data_sample[0]['inputs']) * percentage / 100)
for item in modified_data:
    assert len(item['inputs']) == expected_length
    assert len(item['outputs']) == expected_length
    assert len(item['loss_mask']) == expected_length

# Test append_pause_tokens

data_sample = [{'inputs': ['1', '2', '3', 'N', 'N', 'N'], 'outputs': ['N', 'N', 'N', 'L', '6', 'R'], 'loss_mask': [0, 0, 0, 1, 1, 1]}]
num_tokens = 2
insert_index = -3  # Index before the answer

modified_data = insert_at_ix(copy.deepcopy(data_sample), num_tokens, insert_index, 'T', 'N')

expected_length = len(data_sample[0]['inputs']) + num_tokens
for item in modified_data:
    assert len(item['inputs']) == expected_length
    assert len(item['outputs']) == expected_length
    assert len(item['loss_mask']) == expected_length
    # Ensure the pause tokens are correctly positioned
    if insert_index < 0:
        s = insert_index - num_tokens
        e = insert_index
    else:
        s = insert_index
        e = insert_index + num_tokens
    assert item['inputs'][s:e] == ['T'] * num_tokens


# @@@@@@@@@@



##################################################
# Addition Data

NULL_SYMBOL     = 'N'
PADDING_SYMBOL  = 'P'
THINKING_SYMBOL = 'T'
LTAG = 'L' # answer start
RTAG = 'R' # answer end

tokens = [NULL_SYMBOL, PADDING_SYMBOL, THINKING_SYMBOL, LTAG, RTAG]

all_symbols = Sym.nums + Sym.chars + tokens

# project symbols to vectors, and back
project, unproject, symbols_i2v, symbols_v2i, symbols_vec = Sym.symbol_map(VEC_SIZE, all_symbols, device=DEVICE)

def generate_addition_synthetic_data(num_samples, max_length):
    """Generates synthetic data for the addition task.

    Example 1:

     ins 4 1 5 9 7  N
    outs N N N N N 26
    mask 0 0 0 0 0  1


    Example 2:

     ins P P 2 2 3  N
    outs P P N N N  7
    mask 0 0 0 0 0  1

    """
    data = []
    for _ in range(num_samples):
        length = random.randint(2, max_length)

        # a list of numbers
        numbers = [random.randint(0, 9) for _ in range(length)]
        inputs = numbers + [NULL_SYMBOL] * 3 # *3 because of LTAG and RTAG

        # sum them together
        outputs = [NULL_SYMBOL] * len(numbers) + [LTAG, sum(numbers), RTAG]

        # loss mask helps to ignore loss on the random seed data
        loss_mask = [0] * len(numbers) + [1, 1, 1]

        assert len(inputs) == len(outputs) == len(loss_mask)
        data.append({
            'inputs': inputs,
            'outputs': outputs,
            'loss_mask': loss_mask,
        })
    return data

synthetic_data = generate_addition_synthetic_data(NUM_SAMPLES, MAX_SEQUENCE_LENGTH)

# Inject pause tokens
# synthetic_data = insert_at_ix(synthetic_data, PAUSE_NUM_APPEND, PAUSE_APPEND_INDEX, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)
# synthetic_data = irradiate_tokens(synthetic_data, PAUSE_IRRADIATION_P, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)

train_size = int(TRAIN_SPLIT * len(synthetic_data))
train_data = synthetic_data[:train_size]
val_data = synthetic_data[train_size:]

padding_vec = project(PADDING_SYMBOL)

def collate_fn(batch):
    """Collate function to handle variable-length sequences, and project symbols to
    vectors.

    """
    # Find the longest sequence in the batch
    max_seq_len = max(len(item['inputs']) for item in batch)

    inputs_batch = []
    outputs_batch = []
    loss_masks_batch = []

    for item in batch:
        inputs = item['inputs']
        outputs = item['outputs']
        loss_mask = item['loss_mask']

        inputs_vec = [project(i) for i in inputs]
        num_padding = max_seq_len - len(inputs_vec)
        padded_inputs_vec = [padding_vec] * num_padding + inputs_vec

        padded_outputs_vec = [padding_vec] * num_padding + [project(x) for x in outputs]
        padded_loss_mask = [0] * num_padding + loss_mask

        assert len(padded_inputs_vec) == len(padded_outputs_vec) == len(padded_loss_mask)

        inputs_batch.append(torch.stack(padded_inputs_vec))
        outputs_batch.append(torch.stack(padded_outputs_vec))
        loss_masks_batch.append(torch.tensor(padded_loss_mask))


    # Stack all the padded inputs into a single tensor
    inputs_tensor = torch.stack(inputs_batch).to(DEVICE)
    outputs_tensor = torch.stack(outputs_batch).to(DEVICE)
    loss_masks_tensor = torch.stack(loss_masks_batch).to(DEVICE)

    # Return the tensors with the batch of padded inputs and outputs
    return inputs_tensor, outputs_tensor, loss_masks_tensor

# Create DataLoaders with the new collate_fn
train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
val_dl = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

def print_grid(data, labels=None):
    data = list(data)  # Convert the data to a list if it's not already. Data should be iterable of iterables.
    column_widths = []  # This will store the maximum width needed for each column.
    max_columns = 0  # Stores the maximum number of columns in any row.

    # Calculate the max number of columns in any row
    for row in data:
        for column in row:
            max_columns = max(max_columns, len(str(column)))  # Update max_columns based on length of each column.

    # Initialize column widths to 0
    column_widths = [0] * max_columns  # Initialize column widths array with zeros based on max_columns.

    # Update column widths based on the data
    for i, row in enumerate(data):
        for j, column in enumerate(row):
            for k, item in enumerate(column):
                if isinstance(item, float):
                    w = 6  # For floating point numbers, fix width to 6 (including decimal point and numbers after it).
                else:
                    w = len(str(item))  # For other types, set width to the length of item when converted to string.
                column_widths[k] = max(column_widths[k], w)  # Update column width if current item's width is larger.

    # Print the grid with aligned columns
    for row in data:
        if labels is None:
            labels = [''] * len(row)  # If no labels provided, create empty labels for alignment.
        max_label = max(map(len, labels))  # Find the maximum label width for alignment.
        for lab, column in zip(labels, row):
            print(f"{lab.rjust(max_label)}", end=" ")  # Print label right-justified based on max_label width.
            for i, item in enumerate(column):
                if isinstance(item, float):
                    x = f'{item:>.2f}'  # Format floating point numbers to 2 decimal places.
                else:
                    x = str(item)  # Convert other types to string.
                print(f"{x.rjust(column_widths[i])}", end=" ")  # Right-justify item based on column width.
            print()  # Newline after each column.
        print("-" * (sum(column_widths) + max_label + 1))  # Print separator line after each row.

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
        #     print([unproject(x) for x in i])
        #     print([unproject(x) for x in o])
        #     print(m.tolist())

        print()
        ins = [[unproject(x.to('cuda')) for x in xs] for xs in ins]
        outs = [[unproject(x.to('cuda')) for x in xs] for xs in outs]
        mask = [x.to(int).tolist() for x in mask]
        xx = zip(ins, outs, mask)
        print_grid(xx, labels=['ins', 'outs', 'mask'])
        BRK


##################################################

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



##################################################
#

class Neuralsymbol(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_queue=4, n_stack=2, init_sharpen=5.0):
        super(Neuralsymbol, self).__init__()

        DROPOUT = 0.03

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        id = input_dim
        hd = hidden_dim
        od = output_dim

        # Containers
        self.q1 = Q.Queue(n_queue, input_dim)
        # self.q2 = Q.Queue(n_queue, input_dim)
        self.s = S.Stack(n_stack, input_dim)
        self.zero_offset = 1e-3

        self.sharpen_head = nn.Parameter(torch.tensor([init_sharpen]))
        self.sharpen_tail = nn.Parameter(torch.tensor([init_sharpen]))
        self.sharpen_stack = nn.Parameter(torch.tensor([init_sharpen]))

        # Decisions: Off of Queue
        self.qa_put     = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.qa_get     = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.qa_null_op = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())

        # Decisions: Onto Queue
        self.qb_put     = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.qb_get     = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.qb_null_op = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())

        # Decisions: Off of stack
        self.sa_push    = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.sa_pop     = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.sa_null_op = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())

        # Decisions: Onto stack
        self.sb_push    = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.sb_pop     = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())
        self.sb_null_op = nn.Sequential(nn.Linear(id * 3, hd), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hd, 1), nn.Sigmoid())

        # Summing
        # self.lc = nn.LSTMCell(input_dim * 2, output_dim)
        HH = 64
        self.ff = nn.Sequential(nn.Linear(id * 3, HH, bias=False), nn.GELU(),
                                nn.Linear(HH, output_dim, bias=False), nn.Tanh())


    def forward(self, x):
        # Containers Init
        device = x.device
        batch_size = x.shape[0]
        vec_size = x.shape[2]

        self.q1.init(batch_size, self.zero_offset, device,
                     init_head_ix=1, init_tail_ix=0)

        # # q2 has different head_ix
        # self.q2.init(batch_size, self.zero_offset, device,
        #              init_head_ix=2, init_tail_ix=0)

        self.s.init(batch_size, self.zero_offset, device)

        # LSTM init
        # h = torch.zeros(batch_size, self.output_dim).to(x.device)
        # c = torch.zeros(batch_size, self.output_dim).to(x.device)
        hs = []
        outputs = []

        # Debugging
        latch_states = []
        pops = []
        stack_tops = []

        # Loop over sequence
        for i in range(x.size(1)):
            inp = x[:, i, :]

            ##########
            # Read

            # Heads
            q1h = self.q1.read()
            # q2h = self.q2.read()
            sh = self.s.read()

            a = torch.hstack([inp, q1h, sh])

            # Off of Queues
            q_put     = self.qa_put(a).squeeze(1)
            q_get     = self.qa_get(a).squeeze(1)
            q_null_op = self.qa_null_op(a).squeeze(1)
            q1x = self.q1(self.sharpen_head, self.sharpen_tail, q_put, q_get, q_null_op, inp)
            # q2x = self.q2(self.sharpen_head, self.sharpen_tail, q_put, q_get, q_null_op, inp)

            # Off of Stack
            sa_push    = self.sa_push(a).squeeze(1)
            sa_pop     = self.sa_pop(a).squeeze(1)
            sa_null_op = self.sa_null_op(a).squeeze(1)
            sx = self.s(
                self.sharpen_stack,
                sa_push,
                sa_pop,
                sa_null_op,
                inp)


            ##########
            # Compute

            # Calculation on "working memory"
            xx = torch.hstack([inp, q1x, sx])
            # h, c = self.lc(xx, (h, c))
            # hs.append(h)
            out = self.ff(xx)


            ##########
            # Write

            # Onto Queue
            b = torch.hstack([out, q1x, sx])
            q_put     = self.qb_put(b).squeeze(1)
            q_get     = self.qb_get(b).squeeze(1)
            q_null_op = self.qb_null_op(b).squeeze(1)
            _ = self.q1(self.sharpen_head, self.sharpen_tail, q_put, q_get, q_null_op, out)


            # Onto Stack
            sb_push    = self.sb_push(b).squeeze(1)
            sb_pop     = self.sb_pop(b).squeeze(1)
            sb_null_op = self.sb_null_op(b).squeeze(1)

            _ = self.s(
                self.sharpen_stack,
                sb_push,
                sb_pop,
                sb_null_op,
                out)

            hs.append(out)

        out = torch.stack(hs, dim=1)
        return out, {}



##################################################
# Training

def train_epoch(model, dl, optimizer, device):
    '''
    Args:
      warm_up: ignore loss for this many steps
    '''
    model.train()
    epoch_loss = 0

    for i, (src, trg, mask) in enumerate(dl):
        # src = src.to(device)   # shape=[batch, seq, vec_size]
        # trg = trg.to(device)   # shape=[batch, seq, vec_size]
        # mask = mask.to(device) # shape=[batch, seq]

        optimizer.zero_grad()
        output, latch_states = model(src) # shape=[batch, seq, vec_size]

        # loss = criterion(output, trg) * mask
        loss = ((1 - torch.cosine_similarity(output, trg, dim=2)) * mask).mean()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dl)

def evaluate(model, dl, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg, mask) in enumerate(dl):
            # src = src.to(device)
            # trg = trg.to(device)
            # mask = mask.to(device)

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
            # src = src.to(device)  # Tensor: [batch_size, seq_len, vec_size]
            # trg = trg.to(device)  # Tensor: [batch_size, seq_len, vec_size]
            # mask = mask.to(device)  # Tensor: [batch_size, seq_len]

            output, states = model(src)  # output: Tensor [batch_size, seq_len, vec_size]

            for batch_idx in range(src.size(0)):  # Iterate over each element in the batch
                # Extract the sequence for the current batch element
                predicted_seq = output[batch_idx]
                target_seq = trg[batch_idx]
                mask_seq = mask[batch_idx]

                # Unproject the predicted vectors to symbols and extract numbers between LTAG and RTAG
                predicted_integers = []
                in_answer = False
                for v, m in zip(predicted_seq, mask_seq):
                    if m < 0.00000001:
                        continue  # Skip masked elements
                    symbol = unproject(v)
                    if symbol == LTAG:
                        in_answer = True
                    elif symbol == RTAG:
                        in_answer = False
                    elif in_answer and isinstance(symbol, int):
                        predicted_integers.append(symbol)

                # Unproject the target vectors to symbols and extract numbers between LTAG and RTAG
                target_integers = []
                in_answer = False
                for v, m in zip(target_seq, mask_seq):
                    if m < 0.00000001:
                        continue  # Skip masked elements
                    symbol = unproject(v)
                    if symbol == LTAG:
                        in_answer = True
                    elif symbol == RTAG:
                        in_answer = False
                    elif in_answer and isinstance(symbol, int):
                        target_integers.append(symbol)

                # Only compare if both lists are of the same length
                if len(predicted_integers) == len(target_integers):
                    total_predictions += len(predicted_integers)
                    correct_predictions += (torch.tensor(predicted_integers) == torch.tensor(target_integers)).sum().item()
                else:
                    total_predictions += max(len(predicted_integers), len(target_integers))

                if debug and batch_idx < 1:  # MAX_SHOW = 1 for debugging purposes
                    # Unproject and display the sequences for debugging
                    ins_seq = [unproject(x.to('cuda')) for x in src[batch_idx]]  # List of symbols
                    trgs_seq = [unproject(x.to('cuda')) for x in trg[batch_idx]]  # List of symbols
                    outs_seq = [unproject(x.to('cuda')) for x in output[batch_idx]]  # List of symbols
                    mask_seq = mask[batch_idx].to(int).tolist()  # List of integers (0 or 1)

                    ss = []
                    labels = ['inps', 'trgs', 'outs', 'mask']
                    for k, v in states.items():
                        labels.append(k)
                        data = v['data'][batch_idx]  # Tensor: [seq_len, vec_size]
                        fn = v['fn']
                        if fn is not None:
                            ss_seq = [fn(x.to('cuda')) for x in data]  # List of symbols or other representations
                        else:
                            ss_seq = data.tolist()  # List of list of floats
                        ss.append(ss_seq)

                    all_seqs = [[ins_seq], [trgs_seq], [outs_seq], [mask_seq]] + [[s] for s in ss]
                    print_grid(zip(*all_seqs), labels)  # Print the debug information

    acc = correct_predictions / total_predictions if total_predictions > 0 else 0
    return acc


##########
# Setup

# MyModel = LSTMModel
MyModel = Neuralsymbol

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
    train_loss = train_epoch(model, train_dl, optimizer, DEVICE)

    if epoch % 10 == 0:
        val_loss = evaluate(model, val_dl, DEVICE)
        tacc = accuracy(model, train_dl, DEVICE)
        vacc = accuracy(model, val_dl, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(tacc)
        val_accuracies.append(vacc)
        print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')
    else:
        print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f}')



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
