'''.






NOTE: further work continued at t04_addition_02.py
NOTE: further work continued at t04_addition_02.py
NOTE: further work continued at t04_addition_02.py

NOTE: further work continued at t04_addition_02.py
NOTE: further work continued at t04_addition_02.py
NOTE: further work continued at t04_addition_02.py



Variable Time Computation!

Test Variable Time Computation by building an LM that can process strings of numbers added together

----------
GOOD RECIPES:
  * DO NOT PROJECT SYMBOLS, merely interpolate them in/out
  * All layers NormalizedLinear; repeatedly anneal sharp from fuzzy to sharp;

  * Dumb things
    * Don't do Sigmoid into Softmax, duh. ReLU into softmax seems ok
    * Don't do NormalizedLinear into Sigmoid, duh.
    * Watch where you put Dropout. IE right after a Softmax? You're gonna break something.


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
      found better arch, namely, elem-wise multiply

  HUGE: for the Addition sub network, having (x,y) go through a network cat'd
  together sucked, but doing element wise * or + to have a single vec_size
  vector go through the FFNN totally rocked.

* [X] Simplify addition problem, make it addition with mod 10. RESULTS: Way
      easier to learn, especially when dataset is sparse.

* [X] Perfect LENGTH=2 First!


* Linear encourages rote memorization, NormalizedLinear helps algorithmic generalization?

* Dubious
    * [X] DEBUG track pushes/pops/nops
    * [X] Add (CLEAN UP) a token of "ok, now output the answer".
    * [X] Add a token of "sequence init"
    * [X] Change "sharpen" param throughout training?
    * [X] Init with a particular control op
    * [X] Upgrade handrolled problem with thinking tokens
    * [X] Separate control op from val. RESULTS: *seemed* to help generalization go better.
    * [X] Scale sharpen with epoch number. RESULTS: Worked great in one instance
    * [X] SGD optimizer. RESULTS: much more sensitive to LR, converges slower, can't break through loss barriers.
    * [X] Add softmax to "type"
    * [X] Add norm linear to "type"
    * [X] Split up all stack ops. RESULTS: helped a little?
    * [X] Verify, was it WEIGHT DECAY messing things up?. RESULTS: no actually, Adam sets to 0.0, AdamW sets to 1e-2
    * [X] ff should not project out of work_stack, but merely guard out of work_stack. RESULTS: super important!
    * [X] Is CosineSimilarity worth it? Or should I just use FFNNs? RESULTS: it's too much complexity for now, but worth revisiting. In this vein, I want to experiment again with sharing layers between different pieces of the computation.

----------
TODO:

* [ ] Limit symbol selection

* [ ] Share symbol weights in different components

* [ ] Push, Pop, Nop still seem reluctant to go to 0/1

* [ ] Are we still projecting symbols somewhere? With an init sharpen of 10.0,
      it's happy to memorize with no generalization at all.

* [ ] is val loss right?

* [ ] Norm weight inits

* [ ] Different convolution operator than elem-multiply?

* [ ] How to combine type info with values? F.conv1d?

* [ ] `isinstance`. Add an "int identifier"/"tag"/"type". It looks like the
      stack ops are struggling to always accurately push/pop/nop given a given
      input.

* [ ] Make use of NormalizedLinear, test it!

* [ ] Identify paths of memorization!

* [ ] Noramlizing Linear (like CosSim, but, think it through) Maybe this helps gradients stabilize?

* [ ] increasing LR of sharpen params

* [ ] bias stacks towards pushing at init (ie net nullop since it pops each step?)

* [ ] Cheat and add work_stack.peek() to dataset + loss function

* [ ] Add another stack/queue to help it handle Think tokens / decide when to output

* [ ] REGEX MASKED LOSS: Parse out [LEFT_TAG, <Num>, RIGHT_TAG], and mask loss on that.

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
from neurallambda.torch import cosine_similarity
from torch import einsum
import torch.nn.functional as F
from torch.nn.functional import elu, selu, gelu, leaky_relu
import neurallambda.symbol as Sym
import copy
from neurallambda.tensor import CosineSimilarity, Weight, ReverseCosineSimilarity
from neurallambda.torch import NormalizedLinear, Fn, Parallel, Cat, Stack, Diagnose, Id
import re

torch.manual_seed(152 + 1)

DEBUG = False

PAUSE_IRRADIATION_P = 0.2
MAX_THINK_TOKENS = 3 # "Think" tokens
PAUSE_APPEND_INDEX = -3  # append tokens before [..., L, 42, R]

DEVICE = 'cuda'
VEC_SIZE = 256

NUM_TRAIN_SAMPLES = 1000  # total number of samples in the dataset
MIN_TRAIN_SEQUENCE_LENGTH = 1  # maximum length of the sequence
MAX_TRAIN_SEQUENCE_LENGTH = 3  # maximum length of the sequence

NUM_VAL_SAMPLES = 100
MIN_VAL_SEQUENCE_LENGTH = MAX_TRAIN_SEQUENCE_LENGTH + 1
MAX_VAL_SEQUENCE_LENGTH = MAX_TRAIN_SEQUENCE_LENGTH * 2  # maximum length of the sequence

BATCH_SIZE = 100
LR = 1e-2
WD = 0.0

INIT_SHARPEN = 2.0

NUM_EPOCHS = 60


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

if False: # turning off bc I'm hacking in an "F" token for "final, now output an answer"
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

START_DATA_SYMBOL  = 'O'
PADDING_SYMBOL  = 'P'
START_SEQ_SYMBOL = 'S'
NULL_SYMBOL     = 'N'
THINKING_SYMBOL = 'T'
FINISHED_THINKING_SYMBOL = 'F'
LTAG = 'L' # answer start
RTAG = 'R' # answer end

tokens = [NULL_SYMBOL, PADDING_SYMBOL, THINKING_SYMBOL, FINISHED_THINKING_SYMBOL, LTAG, RTAG, START_DATA_SYMBOL, START_SEQ_SYMBOL]

all_symbols = Sym.nums + Sym.chars + tokens

# project symbols to vectors, and back
# project, unproject, symbols_i2v, symbols_v2i, symbols_vec = Sym.symbol_map(VEC_SIZE, all_symbols, device=DEVICE)

int_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = int_map.project
unproject = int_map.unproject
# symbols_vec = int_map.projection_matrix

def generate_addition_synthetic_data(num_samples, min_length, max_length, max_think_tokens):
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
        length = random.randint(min_length, max_length)
        n_think = random.randint(1, max_think_tokens)

        # a list of numbers

        # numbers = [random.randint(-9, 9) for _ in range(length)] # include negatives
        # numbers = [random.randint(0, 9) for _ in range(length)] # positives only
        numbers = [random.randint(0, 3) for _ in range(length)] # positives only

        # numbers = [random.randint(-2, 2) for _ in range(length)] # a few numbers only
        inputs = (
            [START_SEQ_SYMBOL] +
            numbers +
            [THINKING_SYMBOL] * (n_think - 1) +
            [FINISHED_THINKING_SYMBOL] +
            [NULL_SYMBOL] * 3 # *3 because of LTAG and RTAG
        )
        outputs = [NULL_SYMBOL] * (len(numbers) + 1 + n_think) + [LTAG, sum(numbers) % 10, RTAG]

        # loss mask helps to ignore loss on the random seed data
        loss_mask = [0] * (len(numbers) + 1 + n_think) + [1, 1, 1]

        assert len(inputs) == len(outputs) == len(loss_mask)
        data.append({
            'inputs': inputs,
            'outputs': outputs,
            'loss_mask': loss_mask,
        })
    return data

# TRAIN
train_data = generate_addition_synthetic_data(NUM_TRAIN_SAMPLES, MIN_TRAIN_SEQUENCE_LENGTH, MAX_TRAIN_SEQUENCE_LENGTH, MAX_THINK_TOKENS)
# Inject pause tokens
# train_data = insert_at_ix(train_data, MAX_THINK_TOKENS, PAUSE_APPEND_INDEX, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)
# train_data = irradiate_tokens(train_data, PAUSE_IRRADIATION_P, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)


# VAL
val_data = generate_addition_synthetic_data(NUM_TRAIN_SAMPLES, MIN_VAL_SEQUENCE_LENGTH, MAX_VAL_SEQUENCE_LENGTH, MAX_THINK_TOKENS)
# Inject pause tokens
# val_data = insert_at_ix(val_data, MAX_THINK_TOKENS, PAUSE_APPEND_INDEX, symbol=THINKING_SYMBOL, null_symbol=NULL_SYMBOL)
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
        ins = [[unproject(x.to(DEVICE)) for x in xs] for xs in ins]
        outs = [[unproject(x.to(DEVICE)) for x in xs] for xs in outs]
        mask = [x.to(int).tolist() for x in mask]
        xx = zip(ins, outs, mask)
        print_grid(xx, labels=['ins', 'outs', 'mask'])
        BRK


##################################################
# Training Functions


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
    ins_seq = [unproject(x.to(DEVICE)) for x in src[batch_idx]]  # List of symbols
    trgs_seq = [unproject(x.to(DEVICE)) for x in trg[batch_idx]]  # List of symbols
    outs_seq = [unproject(x.to(DEVICE)) for x in output[batch_idx]]  # List of symbols
    mask_seq = mask[batch_idx].to(int).tolist()  # List of integers (0 or 1)
    ss = []
    labels = ['inps', 'trgs', 'outs', 'mask']
    for k, v in states.items():
        labels.append(k)
        data = v['data'][batch_idx]  # Tensor: [seq_len, vec_size]
        fn = v['fn']
        if fn is not None:
            ss_seq = [fn(x.to(DEVICE)) for x in data]  # List of symbols or other representations
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
# Coloring

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
        self.fc = NormalizedLinear(hidden_dim, output_dim)

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

import torch.fft

def circular_convolution(input1, input2):
    """
    Performs circular convolution on two batches of vectors.

    Args:
    - input1: A tensor of shape (batch_size, vector_length)
    - input2: A tensor of shape (batch_size, vector_length)

    Returns:
    - output: The result of circular convolution, of shape (batch_size, vector_length)
    """
    # Check if inputs are of the same shape
    assert input1.shape == input2.shape, "Input tensors must have the same shape"

    # Perform FFT on both inputs
    fft_input1 = torch.fft.fft(input1, dim=1)
    fft_input2 = torch.fft.fft(input2, dim=1)

    # Element-wise multiplication in the frequency domain
    fft_product = fft_input1 * fft_input2

    # Inverse FFT to transform back to the time domain
    output = torch.fft.ifft(fft_product, dim=1)

    # Since ifft returns complex numbers, we take the real part as the output
    return output.real

def op(n_inp, vec_size, hidden_dim, symbol_weights):
    ''' (vec, vec, vec) -> scalar
    Useful in stack/queue operations.
    '''
    if n_inp == 1:
        fn = Id()
    elif n_inp == 2:
        fn = Fn(lambda a, b: a * b, nargs=2)
    elif n_inp == 3:
        fn = Fn(lambda a, b, c: a * b * c, nargs=3)
    elif n_inp == 4:
        fn = Fn(lambda a, b, c, d: a * b * c * d, nargs=4)
    elif n_inp == 5:
        fn = Fn(lambda a, b, c, d, e: a * b * c * d * e, nargs=5)
    elif n_inp == 6:
        fn = Fn(lambda a, b, c, d, e, f: a * b * c * d * e * f, nargs=6)

    return nn.Sequential(

        ##########
        # Circular Convolution
        Parallel([Fn(lambda x: F.normalize(x, dim=1)), Fn(lambda x: F.normalize(x, dim=1))]),
        Fn(circular_convolution, nargs=2),


        NormalizedLinear(vec_size, hidden_dim, weight=symbol_weights, bias=False),

        nn.Linear(hidden_dim, hidden_dim),
        nn.Softmax(dim=1),

        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
        Fn(lambda x: x.squeeze(1), nargs=1),

        # Fn(lambda x: x.max(dim=1).values, nargs=1),


        # ##########
        # # Conv
        # Stack(dim=1),
        # nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
        # Fn(lambda x: x.squeeze(1), nargs=1), # [batch, 1, vec_size] -> [batch, vec_size]
        # NormalizedLinear(vec_size, hidden_dim, weight=symbol_weights, bias=False),
        # Fn(lambda x: x.max(dim=1).values, nargs=1),


        # ##########
        # # Symbol Matching

        # fn,
        # NormalizedLinear(vec_size, hidden_dim, weight=symbol_weights, bias=False),

        # # # CosineSimilarity(symbol_weights, dim=1, unsqueeze_inputs=[], unsqueeze_weights=[]),
        # # #nn.Linear(vec_size, hidden_dim, bias=False),
        # # # nn.ReLU(),
        # # # nn.Linear(hidden_dim, hidden_dim, bias=False),
        # # # nn.Softmax(dim=1),
        # # # nn.Linear(symbol_weights.size(0), 1, bias=False),
        # # # nn.Sigmoid(),

        # Fn(lambda x: x.max(dim=1).values, nargs=1),

    )


class Neuralsymbol(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, output_dim,
                 ):
        super(Neuralsymbol, self).__init__()

        init_sharpen = INIT_SHARPEN
        dropout_p = 0.0
        n_control_stack = 4
        n_work_stack = 4

        vec_size = input_dim
        self.vec_size = vec_size

        self.input_dim       = input_dim
        self.hidden_dim      = hidden_dim
        self.output_dim      = output_dim
        self.dropout_p       = dropout_p
        self.n_control_stack = n_control_stack
        self.n_work_stack    = n_work_stack

        self.zero_offset = 1e-6

        n_symbols = hidden_dim
        self.n_symbols = n_symbols


        ##########
        # Symbols

        self.sym_w = nn.Parameter(torch.randn((n_symbols, vec_size)))


        ##########
        # Type

        isinstance_dim = 4 # intentional bottleneck
        self.isinstance = nn.Sequential(
            NormalizedLinear(vec_size, isinstance_dim, bias=False),
            # nn.ReLU(),
            nn.Softmax(dim=1),
            # NormalizedLinear(isinstance_dim, vec_size, bias=False),
            nn.Linear(isinstance_dim, vec_size, bias=False),
            nn.Tanh(),
        )

        ##########
        # Control Stack

        self.stack_init_vec = torch.randn(vec_size)

        self.control_stack = S.Stack(n_control_stack, input_dim)
        self.control_sharp = nn.Parameter(torch.tensor([init_sharpen]))

        # Control Ops:
        #   Inp: control dim + input dim
        self.cop1 = op(2, vec_size, hidden_dim, self.sym_w)
        self.cop2 = op(2, vec_size, hidden_dim, self.sym_w)
        self.cop3 = op(2, vec_size, hidden_dim, self.sym_w)

        self.control = nn.Sequential(

            # ##########
            # # Conventional
            # Fn(lambda x, y: x * y, nargs=2),
            # NormalizedLinear(vec_size, hidden_dim, weight=self.sym_w, bias=False),
            # nn.ReLU(),
            # # nn.Softmax(),
            # nn.Dropout(self.dropout_p),
            # NormalizedLinear(hidden_dim, vec_size, weight=self.sym_w, bias=False, reverse=True),
            # nn.Tanh()


            ##########
            # Symbol Matching

            # Fn(lambda x, y: x * y, nargs=2),
            # NormalizedLinear(vec_size, hidden_dim, weight=self.sym_w, bias=False),
            # nn.ReLU(),
            # # nn.Softmax(),
            # nn.Dropout(self.dropout_p),
            # NormalizedLinear(hidden_dim, vec_size, weight=self.sym_w, bias=False, reverse=True),
            # nn.Tanh()


            ##########
            # Convolutional

            # Stack(dim=1),
            # nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            # Fn(lambda x: x.squeeze(1), nargs=1), # [batch, 1, vec_size] -> [batch, vec_size]

            ##########
            # Circular Convolution
            Parallel([Fn(lambda x: F.normalize(x, dim=1)), Fn(lambda x: F.normalize(x, dim=1))]),
            Fn(circular_convolution, nargs=2),
            NormalizedLinear(vec_size, hidden_dim, weight=self.sym_w, bias=False),
            # nn.Linear(vec_size, hidden_dim, bias=True),
            nn.ReLU(),
            # nn.Softmax(),
            nn.Dropout(self.dropout_p),
            # NormalizedLinear(hidden_dim, vec_size, weight=self.sym_w, bias=False, reverse=True),
            nn.Linear(hidden_dim, vec_size, bias=False),
            nn.Tanh()

        )


        ##########
        # Working Stack

        self.work_stack = S.Stack(n_work_stack, input_dim)
        self.work_sharp = nn.Parameter(torch.tensor([init_sharpen]))

        # Control Ops:
        #   Inp: work dim + input dim
        self.wop1 = op(2, vec_size, hidden_dim, self.sym_w)
        self.wop2 = op(2, vec_size, hidden_dim, self.sym_w)
        self.wop3 = op(2, vec_size, hidden_dim, self.sym_w)

        self.work = nn.Sequential(

            # ##########
            # # Conventional
            # Fn(lambda x, y, z: x * y * z, nargs=3),
            # NormalizedLinear(vec_size, hidden_dim, weight=self.sym_w, bias=False),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_p),
            # NormalizedLinear(hidden_dim, vec_size, weight=self.sym_w, reverse=True, bias=False),
            # nn.Tanh(),

            # ##########
            # # Convolutional
            # Stack(dim=1),
            # nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1),
            # Fn(lambda x: x.squeeze(1), nargs=1), # [batch, 1, vec_size] -> [batch, vec_size]

            ##########
            # Circular Convolution
            Parallel([Fn(lambda x: F.normalize(x, dim=1)),
                      Fn(lambda x: F.normalize(x, dim=1)),
                      Fn(lambda x: F.normalize(x, dim=1)),]),

            Fn(lambda a, b, c: circular_convolution(circular_convolution(a, b), c), nargs=3),
            # nn.Linear(vec_size, hidden_dim, bias=True),
            NormalizedLinear(vec_size, hidden_dim, weight=self.sym_w, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            # NormalizedLinear(hidden_dim, vec_size, weight=self.sym_w, reverse=True, bias=False),
            nn.Linear(hidden_dim, vec_size, bias=False),
            nn.Tanh(),

        )

        self.n_out_sym = 4
        self.out_sym = nn.Parameter(torch.randn(self.n_out_sym, vec_size))
        self.select_out = nn.Sequential(
            Parallel([Fn(lambda x: F.normalize(x, dim=1)),
                      Fn(lambda x: F.normalize(x, dim=1)),
                      Fn(lambda x: F.normalize(x, dim=1)),]),

            Parallel([nn.Dropout(self.dropout_p), nn.Dropout(self.dropout_p), nn.Dropout(self.dropout_p)]),
            # Fn(f=lambda x, y, z: x * y * z, nargs=3),
            Fn(lambda a, b, c: circular_convolution(circular_convolution(a, b), c), nargs=3),
            NormalizedLinear(vec_size, hidden_dim, bias=False),
            # nn.Linear(vec_size, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_out_sym + 1, bias=False), # n_out + work
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        # Containers Init
        device = x.device
        batch_size = x.shape[0]
        self.control_stack.init(batch_size, self.zero_offset, device)
        self.work_stack.init(batch_size, self.zero_offset, device)

        # Push first instruction
        self.stack_init_vec = self.stack_init_vec.to(device=device)
        self.control_stack.push(self.stack_init_vec.unsqueeze(0))
        self.work_stack.   push(self.stack_init_vec.unsqueeze(0))

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
            control = self.control_stack.read()
            work = self.work_stack.read()

            typ = self.isinstance(inp)

            # TODO: rm ablation experiment
            # control = control * 0
            # control = control + 1e-5

            # control ops
            op_inp = [control, typ]
            c_push_, c_pop_, c_null_op_ = self.cop1(op_inp), self.cop2(op_inp), self.cop3(op_inp)
            c_push, c_pop, c_null_op = c_push_, c_pop_, c_null_op_

            # c_push_, c_pop_, c_null_op_ = c_push.unsqueeze(-1), c_pop.unsqueeze(-1), c_null_op.unsqueeze(-1)
            # c_push, c_pop, c_null_op = torch.split(torch.softmax(torch.hstack([c_push_, c_pop_, c_null_op_]), dim=1), [1, 1, 1], dim=1)

            # work ops
            w_push_, w_pop_, w_null_op_ = self.wop1(op_inp), self.wop2(op_inp), self.wop3(op_inp)
            w_push, w_pop, w_null_op = w_push_, w_pop_, w_null_op_

            # w_push_, w_pop_, w_null_op_ = w_push.unsqueeze(-1), w_pop.unsqueeze(-1), w_null_op.unsqueeze(-1)
            # w_push, w_pop, w_null_op = torch.split(torch.softmax(torch.hstack([w_push_, w_pop_, w_null_op_]), dim=1), [1, 1, 1], dim=1)

            new_control = self.control([control, typ])
            new_work = self.work([work, inp, typ])

            select_out = self.select_out([control, work, typ])
            options = torch.cat([
                work.unsqueeze(1),        # [batch, 1, vec_size]
                self.out_sym.unsqueeze(0).repeat(batch_size, 1, 1) # [1, n_out_sym, vec_size]
            ], dim=1)
            out = einsum('bn, bnv -> bv', select_out, options)
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
            # c_push, c_pop, c_null_op = c_push.squeeze(1), c_pop.squeeze(1), c_null_op.squeeze(1)
            # w_push, w_pop, w_null_op = w_push.squeeze(1), w_pop.squeeze(1), w_null_op.squeeze(1)

            self.control_stack(self.control_sharp, c_push, c_pop, c_null_op, new_control)
            self.work_stack(self.work_sharp, w_push, w_pop, w_null_op, new_work)

        out = torch.stack(outputs, dim=1)

        for k, v in debug.items():
            debug[k] = {
                'data': torch.stack(v, dim=1),
                'fn': colored,
            }
        return out, debug


##################################################
# Training


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

optimizer = optim.AdamW(opt_params, lr=LR, weight_decay=WD)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''
optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-1)
optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-2)
optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-3)
optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-4)

SHARP = 5.0
with torch.no_grad():
    model.control_sharp[:] = SHARP
    model.work_sharp[:] = SHARP

'''

for epoch in range(NUM_EPOCHS):
    train_loss = run_epoch(model, train_dl, optimizer, DEVICE, train_mode=True)

    # Experiment changing sharpen param
    if True: # TODO: rm experiment
        with torch.no_grad():
            # model.control_sharp[:] = torch.randint(2, 10, (1,))
            # model.work_sharp[:]    = torch.randint(2, 10, (1,))
            model.control_sharp[:] = torch.randint(4, 12, (1,))
            model.work_sharp[:]    = torch.randint(4, 12, (1,))

    if False: # TODO: rm experiment
        LO = 5
        HI = 15
        with torch.no_grad():
            a = epoch / NUM_EPOCHS
            model.control_sharp[:] = (1-a) * LO + a * HI
            model.work_sharp[:]    = (1-a) * LO + a * HI

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


# with torch.no_grad():
#     model.wop1[1].weight[:] = torch.randn_like(model.wop1[1].weight)
#     model.wop2[1].weight[:] = torch.randn_like(model.wop2[1].weight)
#     model.wop3[1].weight[:] = torch.randn_like(model.wop3[1].weight)


# Debug performance
tacc = accuracy(model, val_dl, DEVICE, debug=True)
