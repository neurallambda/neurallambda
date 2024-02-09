'''.

Variable Time Computation!

Test Variable Time Computation by building an LM that can process strings of numbers added together

----------
RESULTS:

* [X] Preloading known symbols into the CosineSimilarity weights helps training
      immensely, but I'm not sure I want to keep using the CosSim layers.

* [X] How does stack sharpening behave? Barely moves from init, but then after
      crossing energy thresholds, it can move a little. Pretty stable though,
      all things considered.

* [X] Make "operation vocabulary" separate from other symbol
      vocabulary. RESULT: GREAT! This gave a speed up, and improved learning,
      and generalization.

* [X] separate work decision and semantics. RESULTS: This seemed to really help generalization.

* [X] Make new number system for projecting ints. bin -> xor projection. RESULTS: Works okish.

* [X] Make separate path for adding numbers. Tested FFNN separately. RESULTS:

  HUGE: for the Addition sub network, having (x,y) go through a network cat'd
  together sucked, but doing element wise * or + to have a single vec_size
  vector go through the FFNN totally rocked.

* [X] Simplify addition problem, make it addition with mod 10. RESULTS: Way
      easier to learn, especially when dataset is sparse.

* Dubious
    * [X] DEBUG track pushes/pops/nops
    * [X] Add (CLEAN UP) a token of "ok, now output the answer".
    * [X] Add a token of "sequence init"
    * [X] Change "sharpen" param throughout training?
    * [X] Init with a particular control op
    * [X] Upgrade handrolled problem with thinking tokens
    * [X] Separate control op from val. RESULTS: *seemed* to help generalization go better.

----------
TODO:

* [ ] `isinstance`. Add an "int identifier"/"tag"/"type". It looks like the
      stack ops are struggling to always accurately push/pop/nop given a given
      input.

* [ ] increasing LR of sharpen

* [ ] bias stacks towards pushing at init (ie net nullop since it pops each step?)

* [ ] Cheat and add work_stack.peek() to dataset + loss function

* [ ] Add another stack/queue to help it handle Think tokens / decide when to output

* [ ] REGEX MASKED LOSS: Parse out [LEFT_TAG, <Num>, RIGHT_TAG], and mask loss on that.

* [ ] Is CosineSimilarity worth it? Or should I just use FFNNs?

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
from neurallambda.tensor import CosineSimilarity, Weight, ReverseCosineSimilarity
import re

torch.manual_seed(152)

DEBUG = False

PAUSE_IRRADIATION_P = 0.1
PAUSE_NUM_APPEND = 3 # "Think" tokens
PAUSE_APPEND_INDEX = -3  # append tokens before [..., L, 42, R]

DEVICE = 'cuda'
VEC_SIZE = 256

NUM_TRAIN_SAMPLES = 1000  # total number of samples in the dataset
MAX_TRAIN_SEQUENCE_LENGTH = 5  # maximum length of the sequence

NUM_VAL_SAMPLES = 100
MAX_VAL_SEQUENCE_LENGTH = 10  # maximum length of the sequence

BATCH_SIZE = 100
LR = 5e-3
INIT_SHARPEN = 5.0

NUM_EPOCHS = 100


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
    # for item in data:
    #     item['inputs'] = item['inputs'][:index] + [symbol] * num_tokens + item['inputs'][index:]
    #     item['outputs'] = item['outputs'][:index] + [null_symbol] * num_tokens + item['outputs'][index:]
    #     item['loss_mask'] = item['loss_mask'][:index] + [0] * num_tokens + item['loss_mask'][index:]

    # # Add F token
    # for item in data:
    #     item['inputs'] = item['inputs'][:index] + [symbol] * num_tokens + ['F'] + item['inputs'][index:]
    #     item['outputs'] = item['outputs'][:index] + [null_symbol] * (num_tokens + 1) + item['outputs'][index:]
    #     item['loss_mask'] = item['loss_mask'][:index] + [0] * (num_tokens + 1) + item['loss_mask'][index:]

    # Add F token
    for item in data:
        item['inputs'] = item['inputs'][:index] + [symbol] * (num_tokens - 1) + ['F'] + item['inputs'][index:]
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

if False: # turning off bc I'm hacking in an "F" token for "final, now output an answer"
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

START_DATA_SYMBOL  = 'O'
PADDING_SYMBOL  = 'P'
START_SEQ_SYMBOL = 'S'
NULL_SYMBOL     = 'N'
THINKING_SYMBOL = 'T'
LTAG = 'L' # answer start
RTAG = 'R' # answer end

tokens = [NULL_SYMBOL, PADDING_SYMBOL, THINKING_SYMBOL, LTAG, RTAG, START_DATA_SYMBOL, START_SEQ_SYMBOL]

all_symbols = Sym.nums + Sym.chars + tokens

# project symbols to vectors, and back
# project, unproject, symbols_i2v, symbols_v2i, symbols_vec = Sym.symbol_map(VEC_SIZE, all_symbols, device=DEVICE)

int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = int_map.project
unproject = int_map.unproject
# symbols_vec = int_map.projection_matrix

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

        # numbers = [random.randint(-9, 9) for _ in range(length)] # include negatives
        numbers = [random.randint(0, 9) for _ in range(length)] # positives only

        # numbers = [random.randint(-2, 2) for _ in range(length)] # a few numbers only
        inputs = [START_SEQ_SYMBOL] + numbers + [NULL_SYMBOL] * 3 # *3 because of LTAG and RTAG

        # sum them together
        # outputs = [NULL_SYMBOL] * len(numbers) + [LTAG, sum(numbers), RTAG]

        # print('DOING %10 VERSION')
        outputs = [NULL_SYMBOL] * (len(numbers) + 1) + [LTAG, sum(numbers) % 10, RTAG]

        # loss mask helps to ignore loss on the random seed data
        loss_mask = [0] * (len(numbers) + 1) + [1, 1, 1]

        assert len(inputs) == len(outputs) == len(loss_mask)
        data.append({
            'inputs': inputs,
            'outputs': outputs,
            'loss_mask': loss_mask,
        })
    return data

# TRAIN
train_data = generate_addition_synthetic_data(NUM_TRAIN_SAMPLES, MAX_TRAIN_SEQUENCE_LENGTH)
# Inject pause tokens
train_data = insert_at_ix(train_data, PAUSE_NUM_APPEND, PAUSE_APPEND_INDEX, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)
# train_data = irradiate_tokens(train_data, PAUSE_IRRADIATION_P, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)


# VAL
val_data = generate_addition_synthetic_data(NUM_TRAIN_SAMPLES, MAX_VAL_SEQUENCE_LENGTH)
# Inject pause tokens
val_data = insert_at_ix(val_data, PAUSE_NUM_APPEND, PAUSE_APPEND_INDEX, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)
# val_data = irradiate_tokens(val_data, PAUSE_IRRADIATION_P, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)


padding_vec = project(PADDING_SYMBOL)
start_data_vec = project(START_DATA_SYMBOL)

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
        # padded_inputs_vec = [padding_vec] * num_padding + inputs_vec
        padded_inputs_vec = [start_data_vec] + [padding_vec] * num_padding + inputs_vec

        padded_outputs_vec = [padding_vec] * (num_padding + 1) + [project(x) for x in outputs]
        padded_loss_mask = [0] * (num_padding + 1) + loss_mask

        assert len(padded_inputs_vec) == len(padded_outputs_vec) == len(padded_loss_mask), f'''
        padded_inputs_vec={len(padded_inputs_vec)} == padded_outputs_vec={len(padded_outputs_vec)} == padded_loss_mask={len(padded_loss_mask)}
        '''.strip()

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
            max_columns = max(max_columns, len(rm_format(str(column))))  # Update max_columns based on length of each column.

    # Initialize column widths to 0
    column_widths = [0] * max_columns  # Initialize column widths array with zeros based on max_columns.

    # Update column widths based on the data
    for i, row in enumerate(data):
        for j, column in enumerate(row):
            for k, item in enumerate(column):
                if isinstance(item, float):
                    w = 6  # For floating point numbers, fix width to 6 (including decimal point and numbers after it).
                else:
                    w = len(rm_format(str(item)))  # For other types, set width to the length of item when converted to string.
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

class Stack(nn.Module):
    def __init__(self, dim=1):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.stack(inputs, dim=self.dim)

class Cat(nn.Module):
    def __init__(self, dim=-1):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        # inputs is a tuple of tensors
        # Each input tensor shape=[batch, features...]
        return torch.cat(inputs, dim=self.dim)

class Parallel(nn.Module):
    def __init__(self, module_tuple):
        super(Parallel, self).__init__()
        self.ms = nn.ModuleList(module_tuple)

    def forward(self, inputs):
        # inputs is a tuple of tensors, parallelizing operations across the tuple
        # Each input tensor shape=[batch, features...]
        outputs = tuple(module(input) for module, input in zip(self.ms, inputs))
        return outputs  # Output is a tuple of tensors, shapes depend on respective modules

class Split(nn.Module):
    def __init__(self, split_sizes, dim=-1):
        super(Split, self).__init__()
        self.split_sizes = split_sizes  # Tuple of sizes to split the tensor into
        self.dim = dim

    def forward(self, input):
        # input shape=[batch, combined features...]
        # torch.split returns a tuple of tensors split according to self.split_sizes
        return torch.split(input, self.split_sizes, dim=self.dim)  # Output shapes depend on self.split_sizes

class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()
    def forward(self, x):
        return x

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(self.dim)

class F(nn.Module):
    def __init__(self, f, nargs=2):
        super(F, self).__init__()
        self.f = f
        self.nargs = nargs
    def forward(self, input):
        if self.nargs == 1:
            return self.f(input)
        if self.nargs == 2:
            x, y = input
            return self.f(x, y)
        if self.nargs == 3:
            x, y, z = input
            return self.f(x, y, z)

class Diagnose(nn.Module):
    def __init__(self, should_raise=True):
        super(Diagnose, self).__init__()
        self.should_raise = should_raise
    def forward(self, input):
        print(f'Input      :', input)
        print(f'Input Shape: {input.shape}')
        if self.should_raise:
            raise RuntimeError('Done diangosing')


##################################################
#

def colored(x):
    # Define the ANSI escape codes for the desired colors
    BLACK = '\033[30m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    RESET = '\033[0m'  # Resets the color to default terminal color

    # Retrieve the value
    value = x.item()

    # Determine the color based on the value
    if 0.0 <= value < 0.4:
        color = BLACK
    elif 0.4 <= value < 0.6:
        color = YELLOW
    else:
        color = RED
    return f'{color}{value:>.1f}{RESET}'

def rm_format(text):
    '''replace all occurrences of the ANSI escape codes with an empty string'''
    ansi_escape_pattern = re.compile(r'\x1B\[[0-9;]*m')
    return re.sub(ansi_escape_pattern, '', text)


class Neuralsymbol(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, output_dim,
                 ):
        super(Neuralsymbol, self).__init__()

        n_symbols = 128
        init_sharpen = INIT_SHARPEN
        dropout_p = 0.0
        n_control_stack = 8
        n_work_stack = 8

        vec_size = input_dim
        self.vec_size = vec_size

        self.input_dim       = input_dim
        self.hidden_dim      = hidden_dim
        self.output_dim      = output_dim
        self.n_symbols       = n_symbols
        self.dropout_p       = dropout_p
        self.n_control_stack = n_control_stack
        self.n_work_stack    = n_work_stack

        self.zero_offset = 1e-4

        ##########
        # Operational Symbols (like push/pop/nopping the stack)

        n_op_symbols = 8
        self.n_op_symbols = n_op_symbols

        # INIT_W = 'orthogonal'
        INIT_W = 'kaiming'
        op_sym_w = Weight(self.input_dim, n_op_symbols, INIT_W)
        self.op_sym_w = op_sym_w

        # cos sim for multiple stacked vecs
        OpCosSim = CosineSimilarity(
            op_sym_w,
            dim=2,
            unsqueeze_inputs=[-1],    # (batch, N, vec_size, _)  # control + input
            unsqueeze_weights=[0, 0], # (_,     _, vec_size, n_op_symbols)
        ) # (batch, N, n_op_symbols)

        # cos sim for a single vec
        OpCosSim_1 = CosineSimilarity(
            op_sym_w,
            dim=1,
            unsqueeze_inputs=[-1],    # (batch, vec_size, _)
            unsqueeze_weights=[0],    # (_,     vec_size, n_op_symbols)
        ) # (batch, n_op_symbols)


        ##########
        # Identifying Symbols

        # INIT_W = 'orthogonal'
        INIT_W = 'kaiming'
        sym_w = Weight(self.input_dim, self.n_symbols, INIT_W)
        self.sym_w = sym_w

        # preload some known symbols
        with torch.no_grad():
            for ix, i in enumerate(tokens + list(range(-40, 40))):
                self.sym_w.weight[:, ix] = project(i)

        # cos sim for multiple stacked vecs
        CosSim = CosineSimilarity(
            sym_w,
            dim=2,
            unsqueeze_inputs=[-1],    # (batch, N, vec_size, _)  # control + input
            unsqueeze_weights=[0, 0], # (_,     _, vec_size, n_symbols)
        ) # (batch, N, n_symbols)

        # cos sim for a single vec
        CosSim_1 = CosineSimilarity(
            sym_w,
            dim=1,
            unsqueeze_inputs=[-1],    # (batch, vec_size, _)
            unsqueeze_weights=[0],    # (_,     vec_size, n_symbols)
        ) # (batch, n_symbols)


        ##########
        #
        isinstance_dim = 8 # intentional bottleneck
        self.isinstance = nn.Sequential(
            Id(),

            # nn.Linear(vec_size, isinstance_dim),
            # nn.ReLU(),
            # nn.Linear(isinstance_dim, vec_size),
            # nn.Tanh(),
        )

        ##########
        # Control Stack
        self.control_stack = S.Stack(n_control_stack, input_dim)
        self.control_sharp = nn.Parameter(torch.tensor([init_sharpen]))

        # Control Ops:
        #   Inp: control dim + input dim
        #   Out: control_op + control_val
        self.control_op = nn.Sequential(
            F(lambda x, y: x * y),
            nn.Linear(vec_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            # nn.Linear(hidden_dim, self.n_op_symbols + 3, bias=True),
            # Split([1, 1, 1, n_op_symbols]),
            # Parallel([nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(),
            #           nn.Sequential(nn.Sigmoid(), ReverseCosineSimilarity(OpCosSim_1))
            #           ]),

            nn.Linear(hidden_dim, 3, bias=True),
            Split([1, 1, 1]),
            Parallel([
                nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(),
            ]),

        )

        # Control Ops:
        #   Inp: control dim + input dim
        #   Out: control_op + control_val
        self.control = nn.Sequential(
            F(lambda x, y: x * y),
            nn.Linear(vec_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            # nn.Linear(hidden_dim, self.n_op_symbols + 3, bias=True),
            # Split([1, 1, 1, n_op_symbols]),
            # Parallel([nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(),
            #           nn.Sequential(nn.Sigmoid(), ReverseCosineSimilarity(OpCosSim_1))
            #           ]),

            nn.Linear(hidden_dim, vec_size, bias=True),
            nn.Tanh()

        )


        ##########
        # Working Stack

        self.work_stack = S.Stack(n_work_stack, input_dim)
        self.work_sharp = nn.Parameter(torch.tensor([init_sharpen]))

        # Control Ops:
        #   Inp: work dim + input dim
        #   Out: work_op + work_val

        self.work_op = nn.Sequential(
            F(lambda x, y: x * y),
            nn.Linear(vec_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_dim, 3, bias=True),
            Split([1, 1, 1]),
            Parallel([nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(),]),


            # Stack(dim=1), # input = (control, input)
            # CosSim,
            # nn.GELU(), # prevent negative sim
            # nn.Flatten(start_dim=1, end_dim=2), # (batch, 2 * n_symbols)
            # nn.Linear(n_symbols * 2, 3), #(batch, work_decision + work_value)
            # Split([1, 1, 1]),
            # Parallel([nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()]),
        )

        self.work = nn.Sequential(
            F(lambda x, y: x * y),
            nn.Linear(vec_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(hidden_dim, vec_size, bias=True),
            nn.Tanh(),

            # nn.Linear(hidden_dim, self.n_symbols, bias=True),
            # nn.Sigmoid(),
            # ReverseCosineSimilarity(CosSim_1),

        )

        self.ff = nn.Sequential(
            F(f=lambda x, y: x * y, nargs=2),
            nn.Linear(vec_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(hidden_dim, vec_size, bias=True),
            nn.Tanh(),

            # nn.Linear(hidden_dim, n_symbols, bias=True),
            # nn.Sigmoid(),
            # ReverseCosineSimilarity(CosSim_1),
        )

    def forward(self, x):
        # Containers Init
        device = x.device
        batch_size = x.shape[0]
        self.control_stack.init(batch_size, self.zero_offset, device)
        self.work_stack.init(batch_size, self.zero_offset, device)

        # Push first instruction
        self.control_stack.push(self.op_sym_w.weight[:, 0].unsqueeze(0))
        self.work_stack.push(self.op_sym_w.weight[:, 0].unsqueeze(0))

        # Debugging
        pops = []
        debug = {
            'control_pop': [],
            'control_push': [],
            'control_null_op': [],
            'work_pop': [],
            'work_push': [],
            'work_null_op': [],
        }

        # Loop over sequence
        outputs = []
        for i in range(x.size(1)):
            inp = x[:, i, :]

            ##########
            # Containers

            # pop stacks no matter what
            control = self.control_stack.pop()
            work = self.work_stack.pop()

            c_push, c_pop, c_null_op = self.control_op([control, self.isinstance(inp)])
            new_control = self.control([control, self.isinstance(inp)])
            w_push, w_pop, w_null_op = self.work_op([control, self.isinstance(inp)])
            new_work = self.work([work, inp])
            out = self.ff([control, work])
            outputs.append(out)

            if True:
                debug['control_pop'].append(c_pop)
                debug['control_push'].append(c_push)
                debug['control_null_op'].append(c_null_op)
                debug['work_pop'].append(w_pop)
                debug['work_push'].append(w_push)
                debug['work_null_op'].append(w_null_op)

            ##########
            # Apply Ops
            c_push, c_pop, c_null_op = c_push.squeeze(1), c_pop.squeeze(1), c_null_op.squeeze(1)
            w_push, w_pop, w_null_op = w_push.squeeze(1), w_pop.squeeze(1), w_null_op.squeeze(1)

            csharp = self.control_sharp
            wsharp = self.work_sharp
            self.control_stack(csharp, c_push, c_pop, c_null_op, new_control)
            self.work_stack(wsharp, w_push, w_pop, w_null_op, new_work)

        out = torch.stack(outputs, dim=1)

        for k, v in debug.items():
            debug[k] = {
                'data': torch.stack(v, dim=1),
                'fn': colored,
            }
        return out, debug


##################################################
# Training

def run_epoch(model, dl, optimizer, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()
    epoch_loss = 0
    steps = 0
    process = torch.enable_grad if train_mode else torch.no_grad
    with process():
        for i, (src, trg, mask) in enumerate(dl):
            src, trg, mask = src.to(device), trg.to(device), mask.to(device)
            if train_mode:
                optimizer.zero_grad()
            output, latch_states = model(src)
            loss = ((1 - torch.cosine_similarity(output, trg, dim=2)) * mask).mean()
            if train_mode:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
    return epoch_loss / max(steps, 1)


def extract_integers_from_seq(sequence, mask):
    """Extract integers between LTAG and RTAG, ignoring masked elements."""
    integers = []
    in_answer = False
    for v, m in zip(sequence, mask):
        if m < 0.00000001:
            continue  # Skip masked elements
        symbol = unproject(v)
        if symbol == LTAG:
            in_answer = True
        elif symbol == RTAG:
            in_answer = False
        elif in_answer and isinstance(symbol, int):
            integers.append(symbol)
    return integers


def debug_output(src, trg, output, mask, batch_idx, states):
    """Print debug information for the first element of the batch."""
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


def accuracy(model, val_dl, device, debug=False):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, (src, trg, mask) in enumerate(val_dl):
            output, states = model(src)  # output: Tensor [batch_size, seq_len, vec_size]
            for batch_idx in range(src.size(0)):  # Iterate over each element in the batch
                # Extract the sequence for the current batch element
                mask_seq = mask[batch_idx]
                predicted_integers = extract_integers_from_seq(output[batch_idx], mask[batch_idx])
                target_integers = extract_integers_from_seq(trg[batch_idx], mask[batch_idx])
                # Only compare if both lists are of the same length
                if len(predicted_integers) == len(target_integers):
                    total_predictions += len(predicted_integers)
                    correct_predictions += (torch.tensor(predicted_integers) == torch.tensor(target_integers)).sum().item()
                else:
                    total_predictions += max(len(predicted_integers), len(target_integers))
                if debug and batch_idx < 1:
                    debug_output(src, trg, output, mask, batch_idx, states)
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

# print('disabling gradients for sharpening')
# model.control_sharp.requires_grad = False
# model.work_sharp.requires_grad = False

opt_params = list(filter(lambda p: p.requires_grad, model.parameters()))

optimizer = optim.Adam(opt_params, lr=LR)


'''
optimizer = optim.Adam(opt_params, lr=1e-1)
optimizer = optim.Adam(opt_params, lr=1e-2)
optimizer = optim.Adam(opt_params, lr=1e-3)
optimizer = optim.Adam(opt_params, lr=1e-4)


SHARP = 10.0
with torch.no_grad():
    model.control_sharp[:] = SHARP
    model.work_sharp[:] = SHARP

'''

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    train_loss = run_epoch(model, train_dl, optimizer, DEVICE, train_mode=True)

    # Experiment changing sharpen param
    if True: # TODO: rm experiment
        with torch.no_grad():
            # model.control_sharp[:] = torch.randint(2, 10, (1,))
            # model.work_sharp[:]    = torch.randint(2, 10, (1,))
            model.control_sharp[:] = torch.randint(4, 12, (1,))
            model.work_sharp[:]    = torch.randint(4, 12, (1,))


    if epoch % 10 == 0:
        val_loss = run_epoch(model, val_dl, None, DEVICE, train_mode=False)
        tacc = accuracy(model, train_dl, DEVICE)
        vacc = accuracy(model, val_dl, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(tacc)
        val_accuracies.append(vacc)
        print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')
    else:
        if isinstance(model, Neuralsymbol):
            print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f} | CSharp: {model.control_sharp.item():.3f} | WSharp: {model.work_sharp.item():.3f}')
        else:
            print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.5f}')


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

# Debug performance
tacc = accuracy(model, val_dl, DEVICE, debug=True)
