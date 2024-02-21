'''.

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


* [X] Add Control Ops to dataset
* [X] Incorporate new dataset
* [X] Remove "Think" tokens for now
* [X] Add Control signals to debug_state
* [X] add to loss
* [ ] Freeze symbols, train selectors
* [ ] Train with/without control signals
* [ ] `work` is calculated with nonlinearity



* [ ] model.push_or_null & model.pop_or_null, do both per step

* [ ] Normalize weights of Linears following Choices at the start of forward
* [ ] Freeze all params except Control Ops
* [ ] Apply Think tokens/noise outside datasaet (so each epoch is different)


-----

* [ ] Inspiration via ResNet?

* [ ] Train sub components separately before full model run.

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

* [ ] bias stacks towards nullop

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
from torch import cosine_similarity, einsum
import torch.nn.functional as F
from torch.nn.functional import elu, selu, gelu, leaky_relu
import neurallambda.symbol as Sym
import copy
from neurallambda.tensor import CosineSimilarity, Weight, ReverseCosineSimilarity
from neurallambda.torch import NormalizedLinear, Fn, Parallel, Cat, Stack, Diagnose, Id, cosine_similarity
import re
import pandas as pd
import itertools


torch.manual_seed(152 + 1)

DEBUG = False

PAUSE_IRRADIATION_P = 0.2
MAX_THINK_TOKENS    = 3 # "Think" tokens
PAUSE_APPEND_INDEX  = -3  # append tokens before [..., L, 42, R]

DEVICE = 'cuda'
VEC_SIZE = 128

NUM_TRAIN_SAMPLES = 1000  # total number of samples in the dataset
MIN_TRAIN_SEQUENCE_LENGTH = 1  # maximum length of the sequence
MAX_TRAIN_SEQUENCE_LENGTH = 3  # maximum length of the sequence

NUM_VAL_SAMPLES = 100
MIN_VAL_SEQUENCE_LENGTH = MAX_TRAIN_SEQUENCE_LENGTH + 1
MAX_VAL_SEQUENCE_LENGTH = MAX_TRAIN_SEQUENCE_LENGTH * 2  # maximum length of the sequence

# CHOICE_METHOD = 'softmax'
CHOICE_METHOD = 'outer_projection'
REDUNDANCY = 4

BATCH_SIZE = 100
LR = 1e-2
WD = 0.0

INIT_SHARPEN = 2.0

NUM_EPOCHS = 60


##################################################
# Problem Data

START_DATA_SYMBOL  = 'O'
PADDING_SYMBOL  = '.'
START_SEQ_SYMBOL = 'S'
WILD_SYMBOL     = '_'
THINKING_SYMBOL = 'T'
FINISHED_THINKING_SYMBOL = 'F'
LTAG = '<' # answer start
RTAG = '>' # answer end

# Stack ops
NULL_OP_SYMBOL = 'NOP'
PUSH_SYMBOL    = 'PSH'
POP_SYMBOL     = 'POP'

# Control Vals
SEQUENCE_STARTED_SYMBOL = 'SS'
SEQUENCE_FINISHED_SYMBOL = 'FF'
RETURN_L_SYMBOL = "RL"
RETURN_SUM_SYMBOL = "RS"
RETURN_R_SYMBOL = "RR"

tokens = [
    WILD_SYMBOL, PADDING_SYMBOL, THINKING_SYMBOL, FINISHED_THINKING_SYMBOL,
    LTAG, RTAG, START_DATA_SYMBOL, START_SEQ_SYMBOL,
    NULL_OP_SYMBOL, PUSH_SYMBOL, POP_SYMBOL,
    SEQUENCE_STARTED_SYMBOL, SEQUENCE_FINISHED_SYMBOL, RETURN_L_SYMBOL, RETURN_SUM_SYMBOL, RETURN_R_SYMBOL
]

all_symbols = Sym.nums + Sym.chars + tokens
sym_map = Sym.SymbolMapper(VEC_SIZE, all_symbols, device=DEVICE)
project = sym_map.project
unproject = sym_map.unproject


##########
# Read Haskell-generated CSV.
#
#   Each column contains a sequence which must be parsed into a list.

pattern = re.compile(r'\(PSH +[^\^ ^)]+\)|[^ ]+')

def parse_cell(cell):
    xs = pattern.findall(cell)
    return xs

# @@@@@@@@@@
if False:
    cell_content = "(PSH value) command1 (PSH value) command2"
    parsed_elements = parse_cell(cell_content)
    print(parsed_elements)
# @@@@@@@@@@

def create_loss_mask(seq:[str])->[int]:
    ''' Given a list of strings, replace every element with 0 except for those
    in the sub-sequence `L ... R` which get a 1 '''
    # Initialize a mask with zeros of the same length as the sequence
    mask = [0] * len(seq)
    try:
        start_index = seq.index(LTAG)
        end_index = seq.index(RTAG)
        # Mark elements between 'L' and 'R' inclusive as 1
        for i in range(start_index, end_index + 1):
            mask[i] = 1
    except ValueError:
        pass
    return mask

def read_csv(data_path):
    df = pd.read_csv(data_path, sep="|")
    # df = df[['Input', 'Output', 'PreGlobalOp', 'PreWorkOp', 'PostGlobalOp', 'PostWorkOp']]
    for col in df.columns:
        df[col] = df[col].apply(parse_cell)
    # Add Loss Mask
    df['LossMask'] = df['Output'].apply(create_loss_mask)
    return df

data_path_3 = "experiment/t04_addition/mod_sum_length_3.csv"
data_path_5 = "experiment/t04_addition/mod_sum_length_5.csv"
data_path_10 = "experiment/t04_addition/mod_sum_length_10.csv"
data_path_20 = "experiment/t04_addition/mod_sum_length_20.csv"

df = read_csv(data_path_3)

# TRAIN
train_data = Dataset.from_pandas(read_csv(data_path_3))
# Inject pause tokens
# train_data = insert_at_ix(train_data, MAX_THINK_TOKENS, PAUSE_APPEND_INDEX, symbol=THINKING_SYMBOL, null_symbol=WILD_SYMBOL)
# train_data = irradiate_tokens(train_data, PAUSE_IRRADIATION_P, symbol=THINKING_SYMBOL, null_symbol=WILD_SYMBOL)


# VAL
val_data = Dataset.from_pandas(read_csv(data_path_5))
# Inject pause tokens
# val_data = insert_at_ix(val_data, MAX_THINK_TOKENS, PAUSE_APPEND_INDEX, symbol=THINKING_SYMBOL, null_symbol=WILD_SYMBOL)
# val_data = irradiate_tokens(val_data, PAUSE_IRRADIATION_P, symbol=THINKING_SYMBOL, null_symbol=WILD_SYMBOL)


padding_vec = project(PADDING_SYMBOL)
start_data_vec = project(START_DATA_SYMBOL)
nop_vec = project(NULL_OP_SYMBOL)
wild_vec = project(WILD_SYMBOL)

def project_(x):
    ''' Try to cast strings to ints '''
    try:
        x = int(x)
    except:
        pass
    return project(x)

def collate_fn(batch):
    """Collate function to handle variable-length sequences, and project symbols to
    vectors.

    """
    # Find the longest sequence in the batch
    max_seq_len = max(len(item['Input']) for item in batch)

    out = {k:[] for k in batch[0].keys()}

    # symbol versions (ie not vecs)
    out['PreGlobalOp_'] = []
    out['PreWorkOp_'] = []
    out['PostGlobalOp_'] = []
    out['PostGlobalVal_'] = []
    out['PostWorkOp_'] = []
    out['PostWorkVal_'] = []

    for item in batch:

        # Input
        inputs = item['Input']
        inputs_vec = [project_(i) for i in inputs]
        num_padding = max_seq_len - len(inputs_vec)
        inputs_vec = torch.stack([start_data_vec] + [padding_vec] * num_padding + inputs_vec)
        out['Input'].append(inputs_vec)

        # Output
        outputs = item['Output']
        outputs_vec = torch.stack([padding_vec] * (num_padding + 1) + [project_(x) for x in outputs])
        out['Output'].append(outputs_vec)

        # loss_mask
        loss_mask = item['LossMask']
        loss_mask_vec = torch.tensor([0] * (num_padding + 1) + loss_mask)
        out['LossMask'].append(loss_mask_vec)

        # stack projected vecs
        out['PreGlobalOp'].append   (torch.stack([nop_vec] * (num_padding + 1) + [project_(x) for x in item['PreGlobalOp']]))
        out['PreWorkOp'].append     (torch.stack([nop_vec] * (num_padding + 1) + [project_(x) for x in item['PreWorkOp']]))
        out['PostGlobalOp'].append  (torch.stack([nop_vec] * (num_padding + 1) + [project_(x) for x in item['PostGlobalOp']]))
        out['PostGlobalVal'].append (torch.stack([wild_vec] * (num_padding + 1) + [project_(x) for x in item['PostGlobalVal']]))
        out['PostWorkOp'].append    (torch.stack([nop_vec] * (num_padding + 1) + [project_(x) for x in item['PostWorkOp']]))
        out['PostWorkVal'].append   (torch.stack([wild_vec] * (num_padding + 1) + [project_(x) for x in item['PostWorkVal']]))

        # non projected vecs
        out['PreGlobalOp_'].append(item['PreGlobalOp'])
        out['PreWorkOp_'].append(item['PreWorkOp'])
        out['PostGlobalOp_'].append(item['PostGlobalOp'])
        out['PostGlobalVal_'].append(item['PostGlobalVal'])
        out['PostWorkOp_'].append(item['PostWorkOp'])
        out['PostWorkVal_'].append(item['PostWorkVal'])


    # Stack all the padded inputs into a single tensor
    out['Input'] = torch.stack(out['Input'])
    out['Output'] = torch.stack(out['Output'])
    out['LossMask'] = torch.stack(out['LossMask'])
    out['PreGlobalOp'] = torch.stack(out['PreGlobalOp'])
    out['PreWorkOp'] = torch.stack(out['PreWorkOp'])
    out['PostGlobalOp'] = torch.stack(out['PostGlobalOp'])
    out['PostGlobalVal'] = torch.stack(out['PostGlobalVal'])
    out['PostWorkOp'] = torch.stack(out['PostWorkOp'])
    out['PostWorkVal'] = torch.stack(out['PostWorkVal'])

    # # non projected vecs
    # out['PreGlobalOp_'] = out['PreGlobalOp_']
    # out['PreWorkOp_'] = out['PreWorkOp_']
    # out['PostGlobalOp_'] = out['PostGlobalOp_']
    # out['PostGlobalOpVal_'] = out['PostGlobalOpVal_']
    # out['PostWorkOp_'] = out['PostWorkOp_']
    # out['PostWorkOpVal_'] = out['PostWorkOpVal_']


    # assert shapes
    ignore_names = {
        'PreGlobalOp_',
        'PreWorkOp_',
        'PostGlobalOp_',
        'PostGlobalVal_',
        'PostWorkOp_',
        'PostWorkVal_',
    }
    for name, v in out.items():
        if name in ignore_names:
            continue
        elif name in {'LossMask'}:
            assert v.size(0) == out['Input'].size(0)
            assert v.size(1) == out['Input'].size(1)
        else:
            assert out['Input'].shape == v.shape, f'Shapes must be the same, input={out["Input"].shape}, {name}={v.shape}'

    # Return the tensors with the batch of padded inputs and outputs
    return out

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
# Training Functions

def mask_stack_op(xs: [str], keep_op):
    assert keep_op in {'PSH', 'POP', 'NOP'}
    out = [1 if x == keep_op else 0 for x in xs]
    return torch.tensor(out)


def run_epoch(model, dl, optimizer, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()
    epoch_loss = 0
    steps = 0
    process = torch.enable_grad if train_mode else torch.no_grad
    with process():
        for i, data in enumerate(dl):
            for k, v in data.items():
                if k.endswith('_'): # ignore the non-projected symbols
                    continue
                data[k] = v.to(device)
            src = data['Input']
            trg = data['Output']
            mask = data['LossMask']

            with torch.no_grad():
                trg_pre_c_op = data['PreGlobalOp']
                trg_pre_w_op = data['PreWorkOp']
                trg_post_c_op = data['PostGlobalOp']
                trg_post_c_val = data['PostGlobalVal']
                trg_post_w_op = data['PostWorkOp']
                trg_post_w_val = data['PostWorkVal']

            if train_mode:
                optimizer.zero_grad()
            output, debug = model(src)

            pre_c_pop = debug['pre_c_pop']['data']
            pre_c_nop = debug['pre_c_nop']['data']
            pre_w_pop = debug['pre_w_pop']['data']
            pre_w_nop = debug['pre_w_nop']['data']
            post_c_push = debug['post_c_push']['data']
            post_c_val = debug['post_c_val']['data']
            post_c_nop = debug['post_c_nop']['data']
            post_w_push = debug['post_w_push']['data']
            post_w_val = debug['post_w_val']['data']
            post_w_nop = debug['post_w_nop']['data']

            loss = ((1 - cosine_similarity(output, trg, dim=2)) * mask).mean()
            # loss = (F.mse_loss(br_output, br_trg) * mask).mean()

            # REGULARIZATION EXPERIMENT
            #   TODO: rm experiment

            op_loss = 0
            for bix in range(src.size(0)):
                for i in range(src.size(1)):

                    # post control
                    trg = trg_post_c_op[bix][i]
                    utrg = unproject(trg)
                    if utrg == 'NOP':
                        op_loss += 1 - post_c_nop[bix][i]
                    elif utrg == 'PSH':
                        op_loss += 1 - post_c_push[bix][i]

                    # post work
                    trg = trg_post_w_op[bix][i]
                    utrg = unproject(trg)
                    if utrg == 'NOP':
                        op_loss += 1 - post_w_nop[bix][i]
                    elif utrg == 'PSH':
                        op_loss += 1 - post_w_push[bix][i]
            op_loss = op_loss / (src.size(0) * src.size(1))

            val_loss = (
                (1 - cosine_similarity(trg_post_c_val, post_c_val, dim=2)) +
                (1 - cosine_similarity(trg_post_w_val, post_w_val, dim=2))
            ).mean()

            loss = loss + op_loss * 0.1  + val_loss * 0.1


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
        for i, data in enumerate(val_dl):
            # for k, v in data.items():
            #     if k.endswith('_'): # ignore non projected stack ops
            #         continue
            #     data[k] = v.to(device)
            src = data['Input'].to(device)
            trg = data['Output'].to(device)
            mask = data['LossMask'].to(device)
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

    # Retrieve the single torch value
    if isinstance(x, tuple):
        string, value = x
    else:
        value = x.item()
        string = f'{value:>.1f}'

    # Determine the color based on the value
    if 0.0 <= value < 0.4:
        color = BLACK
    elif 0.4 <= value < 0.6:
        color = YELLOW
    else:
        color = RED
    return f'{color}{string}{RESET}'

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

class Choice(nn.Module):
    ''' N-vectors -> [0, 1] '''
    def __init__(self, vec_size, n_vecs, n_choices, redundancy, has_guard=False, method='mean'):
        ''' `has_guard` uses separate weights to calculate a gate that
        multiplies against the values of `self.ff(inp)` '''
        super(Choice, self).__init__()
        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.has_guard = has_guard
        self.method = method
        assert method in {'mean', 'max', 'softmax', 'outer_projection'}

        # Stack inputs
        if n_vecs == 1:
            Vecs = Id()
        elif n_vecs == 2:
            Vecs = Fn(lambda a, b: torch.hstack([a, b]), nargs=2)
        elif n_vecs == 3:
            Vecs = Fn(lambda a, b, c: torch.hstack([a, b, c]), nargs=3)
        elif n_vecs == 4:
            Vecs = Fn(lambda a, b, c, d: torch.hstack([a, b, c, d]), nargs=4)
        elif n_vecs == 5:
            Vecs = Fn(lambda a, b, c, d, e: torch.hstack([a, b, c, d, e]), nargs=5)
        elif n_vecs == 6:
            Vecs = Fn(lambda a, b, c, d, e, f: torch.hstack([a, b, c, d, e, f]), nargs=6)

        self.ff = nn.Sequential(Vecs, nn.Linear(vec_size * n_vecs, redundancy * n_choices, bias=False), nn.Sigmoid())

        if has_guard:
            self.guard = nn.Sequential(Vecs, nn.Linear(vec_size * n_vecs, redundancy * n_choices, bias=False), nn.Sigmoid())

        if method == 'outer_projection':
            if n_choices == 2:
                self.outer = lambda xs: torch.einsum('za, zb -> zab', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices == 3:
                self.outer = lambda xs: torch.einsum('za, zb, zc -> zabc', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices == 4:
                self.outer = lambda xs: torch.einsum('za, zb, zc, zd -> zabcd', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices == 5:
                self.outer = lambda xs: torch.einsum('za, zb, zc, zd, ze -> zabcde', *xs).flatten(start_dim=1, end_dim=-1)
            elif n_choices > 5:
                raise ValueError(f'The outer_projection method scales as O(redundancy ** n_choices). You probably dont want n_choices>5, but you picked n_choices={n_choices}')

            self.proj = nn.Sequential(
                nn.Linear(redundancy ** n_choices, redundancy, bias=True),
                nn.ReLU(),
                nn.Linear(redundancy, n_choices, bias=True),
                nn.Sigmoid(),
                # nn.Softmax(dim=1)
            )

    def forward(self, inp):
        if self.has_guard:
            f = self.ff(inp)
            g = self.guard(inp)
            outs = f * g
        else:
            outs = self.ff(inp)

        if self.method == 'outer_projection':
            chunks = torch.chunk(outs, self.n_choices, dim=1)
            return self.proj(self.outer(chunks))
        elif self.method == 'mean':
            return torch.mean(outs.view(-1, self.n_choices, self.redundancy), dim=2)
        elif self.method == 'max':
            return torch.max(outs.view(-1, self.n_choices, self.redundancy), dim=2).values
        elif self.method == 'softmax':
            return torch.sum(outs.softmax(dim=1).view(-1, self.n_choices, self.redundancy), dim=2)



####################

# def generate_combinations(elements, max_length):
#     """
#     Generate all possible combinations of the elements up to a specified maximum length.

#     :param elements: A list of elements to combine.
#     :param max_length: The maximum length of the combinations.

#     Example sizes of combos:

#         len(generate_combinations(range(2), 2))  ==  3
#         len(generate_combinations(range(3), 2))  ==  6
#         len(generate_combinations(range(3), 3))  ==  7
#         len(generate_combinations(range(4), 2))  ==  10
#         len(generate_combinations(range(4), 3))  ==  14
#         len(generate_combinations(range(4), 4))  ==  15
#         len(generate_combinations(range(5), 2))  ==  15
#         len(generate_combinations(range(5), 3))  ==  25
#         len(generate_combinations(range(5), 4))  ==  30
#         len(generate_combinations(range(5), 5))  ==  31
#     """
#     # Store all combinations in a list
#     all_combinations = []

#     # Generate combinations for every length up to max_length
#     for length in range(1, max_length + 1):
#         # itertools.combinations generates combinations of the current length
#         combinations = itertools.combinations(elements, length)
#         # Add the current combinations to the total list
#         all_combinations.extend(combinations)

#     return all_combinations

# # Test the function
# elements = ['a', 'b', 'c', 'd', 'e']  # A list of length 5
# max_length = 3  # Generate combinations up to length 3

# # Generate and print all combinations
# combinations = generate_combinations(elements, max_length)
# for combo in combinations:
#     print(combo)


##############################


class Choice2(nn.Module):
    ''' N-vectors -> [0, 1] '''
    def __init__(self, vec_size, n_vecs, symbols, n_include_symbols=0, has_guard=False, method='mean'):
        ''' `has_guard` uses separate weights to calculate a gate that
        multiplies against the values of `self.ff(inp)` '''
        super(Choice2, self).__init__()
        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.symbols = symbols
        self.n_include_symbols = n_include_symbols
        self.has_guard = has_guard
        self.method = method
        assert method in {'mean', 'max', 'softmax', 'outer_projection'}

        n_sym = symbols.size(0) + n_include_symbols

        HIDDEN_DIM = 64
        self.choice = nn.Sequential(
            nn.Linear((n_vecs + n_sym) * vec_size, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, n_vecs + n_sym),
        )

    def forward(self, inp, include=None):
        assert len(inp) == self.n_vecs, f'shapes no the same: len(inp)={len(inp)} vs self.n_vecs={self.n_vecs}'
        if self.n_include_symbols > 0:
            assert len(include) == self.n_include_symbols

        vecs = torch.stack(inp, dim=1)
        if include is not None:
            vecs = torch.concat(vecs, include)
        # # inp     = [batch, n_vecs, _    , vec_size]
        # # symbols = [_    , _     , n_sym, vec_size]
        # # out     = [batch, n_vecs, n_sym]
        # inp_symbols = cosine_similarity(
        #     inps.unsqueeze(2),
        #     self.symbols.unsqueeze(0).unsqueeze(0), dim=4)
        breakpoint()
        HERE
        choose = self.choice(vecs)
        return einsum('nv, b -> bv', self.symbols, choose)


def op(n_inp, vec_size):
    ''' (vec, vec, vec) -> scalar
    Useful in stack/queue operations.
    '''
    return nn.Sequential(
        Choice(vec_size, n_inp, n_choices=2, redundancy=REDUNDANCY, has_guard=True, method=CHOICE_METHOD),
        Fn(lambda x: (x[:,0] - x[:,1] + 1) / 2, nargs=1), # convert softmax choice to scalar
        Fn(lambda x: x.unsqueeze(-1)),
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
        symbols = nn.Parameter(torch.vstack([
            project(0), project(1), project(2), project(3), project(4),
            project(5), project(6), project(7), project(8), project(9),

            project(START_DATA_SYMBOL),
            project(PADDING_SYMBOL),
            project(START_SEQ_SYMBOL),
            project(WILD_SYMBOL),
            project(THINKING_SYMBOL),
            project(FINISHED_THINKING_SYMBOL),
            project(LTAG),
            project(RTAG),
            project(NULL_OP_SYMBOL),
            project(PUSH_SYMBOL),
            project(POP_SYMBOL),
            project(SEQUENCE_STARTED_SYMBOL),
            project(SEQUENCE_FINISHED_SYMBOL),
            project(RETURN_L_SYMBOL),
            project(RETURN_SUM_SYMBOL),
            project(RETURN_R_SYMBOL),
        ]))
        self.symbols = symbols

        ##########
        # Type

        sym_type_hidden = 8
        self.sym_type = Choice2(vec_size, 1, symbols)
        # self.sym_type = nn.Sequential(
        #     # Choice(vec_size, n_vecs=1, n_choices=2, redundancy=REDUNDANCY, has_guard=True, method=CHOICE_METHOD),
        #     # nn.Linear(2, vec_size, bias=True),
        #     #
        #     nn.Linear(vec_size, hidden_dim, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, vec_size, bias=True),
        #     Fn(lambda x: F.normalize(x, dim=1)),
        # )

        ##########
        # Control Stack

        self.stack_init_vec = torch.randn(vec_size)
        self.control_stack = S.Stack(n_control_stack, input_dim)
        self.control_sharp = nn.Parameter(torch.tensor([init_sharpen]))
        self.pre_c_pop,   self.pre_c_nop  = op(3, vec_size), op(3, vec_size)
        self.post_c_push, self.post_c_nop = op(3, vec_size), op(3, vec_size)
        self.post_c_val = Choice2(vec_size, 3, symbols)
        # self.post_c_val = nn.Sequential(
        #     # Choice(vec_size, n_vecs=3, n_choices=n_choices, redundancy=REDUNDANCY, has_guard=True, method=CHOICE_METHOD),
        #     # nn.Linear(n_choices, vec_size, bias=False),
        #     Fn(lambda a, b, c: a * b * c, nargs=3),
        #     nn.Linear(vec_size, hidden_dim, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, vec_size, bias=True),
        #     Fn(lambda x: F.normalize(x, dim=1)),
        # )

        ##########
        # Working Stack

        self.work_stack = S.Stack(n_work_stack, input_dim)
        self.work_sharp = nn.Parameter(torch.tensor([init_sharpen]))
        self.pre_w_pop,   self.pre_w_nop   = op(3, vec_size), op(3, vec_size)
        self.post_w_push, self.post_w_nop  = op(3, vec_size), op(3, vec_size)

        self.work_guard = op(1, vec_size)
        self.post_w_val = Choice2(vec_size, 3, symbols)
        # self.post_w_val = nn.Sequential(
        #     # Choice(vec_size, n_vecs=3, n_choices=n_choices, redundancy=3, has_guard=True, method=CHOICE_METHOD),
        #     # nn.Linear(n_choices, vec_size, bias=False),
        #     Fn(lambda a, b, c: a * b * c, nargs=3),
        #     nn.Linear(vec_size, hidden_dim, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, vec_size, bias=True),
        #     Fn(lambda x: F.normalize(x, dim=1)),
        # )

        ##########
        # Select output

        self.n_out_sym = 4
        self.out_sym = nn.Parameter(torch.randn(self.n_out_sym, vec_size))
        self.select_out = Choice2(vec_size, 3, symbols)
        # self.select_out = nn.Sequential(
        #     Choice(vec_size, n_vecs=3, n_choices=self.n_out_sym + 1, redundancy=REDUNDANCY, has_guard=True, method=CHOICE_METHOD),
        # )

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
            'pre_c_pop': [],
            'pre_c_nop': [],
            'pre_w_pop': [],
            'pre_w_nop': [],
            'post_c_push': [],
            'post_c_val': [],
            'post_c_nop': [],
            'post_w_push': [],
            'post_w_val': [],
            'post_w_nop': [],
        }

        # Loop over sequence
        outputs = []
        for i in range(x.size(1)):
            inp = x[:, i, :]

            ##########
            # Pre

            # inp_typ = inp
            inp_typ = self.sym_type([inp])
            c_peek = self.control_stack.read()
            w_peek = self.work_stack.read()
            pre_inp = [c_peek, w_peek, inp_typ]

            # control stack maybe pop
            pre_c_pop, pre_c_nop = self.pre_c_pop(pre_inp).squeeze(1), self.pre_c_nop(pre_inp).squeeze(1)
            # pre_c_pop, pre_c_nop = (pre_c_pop - pre_c_nop + 1) / 2, (pre_c_nop - pre_c_pop + 1) / 2
            self.control_stack.pop_or_null_op(self.control_sharp, pre_c_pop, pre_c_nop)

            # work stack maybe pop
            pre_w_pop, pre_w_nop = self.pre_w_pop(pre_inp).squeeze(1), self.pre_w_nop(pre_inp).squeeze(1)
            # pre_w_pop, pre_w_nop = (pre_c_pop - pre_c_nop + 1) / 2, (pre_c_nop - pre_c_pop + 1) / 2
            self.work_stack.pop_or_null_op(self.work_sharp, pre_w_pop, pre_w_nop)

            ##########
            # Post
            post_inp = [c_peek, w_peek, inp_typ]

            # control stack maybe push
            post_c_push, post_c_nop = self.post_c_push(post_inp).squeeze(1), self.post_c_nop(post_inp).squeeze(1)
            # post_c_push, post_c_nop = (post_c_push - post_c_nop + 1) / 2, (post_c_nop - post_c_push + 1) / 2
            control_val = self.post_c_val(post_inp)
            self.control_stack.push_or_null_op(self.control_sharp, post_c_push, post_c_nop, control_val)

            # work stack maybe push
            post_w_push, post_w_nop = self.post_w_push(post_inp).squeeze(1), self.post_w_nop(post_inp).squeeze(1)
            # post_w_push, post_w_nop = (post_c_push - post_c_nop + 1) / 2, (post_c_nop - post_c_push + 1) / 2

            # work_guard = self.work_guard(inp_typ)

            # "sum mod 10" happens here
            work_val = self.post_w_val(post_inp)
            self.work_stack.push_or_null_op(self.work_sharp, post_w_push, post_w_nop, work_val)

            ##########
            # Select out

            # select_out = self.select_out([control_val, work_val, inp_typ])
            # options = torch.cat([
            #     work_val.unsqueeze(1),        # [batch, 1, vec_size]
            #     self.out_sym.unsqueeze(0).repeat(batch_size, 1, 1) # [1, n_out_sym, vec_size]
            # ], dim=1)
            # out = einsum('bn, bnv -> bv', select_out, options)

            out = self.select_out([control_val, work_val, inp_typ], include=[work_val])


            outputs.append(out)

            if True:
                debug['pre_c_pop'].append(pre_c_pop)
                debug['pre_c_nop'].append(pre_c_nop)
                debug['pre_w_pop'].append(pre_w_pop)
                debug['pre_w_nop'].append(pre_w_nop)
                debug['post_c_push'].append(post_c_push)
                debug['post_c_val'].append(control_val)
                debug['post_c_nop'].append(post_c_nop)
                debug['post_w_push'].append(post_w_push)
                debug['post_w_val'].append(work_val)
                debug['post_w_nop'].append(post_w_nop)

        out = torch.stack(outputs, dim=1)

        for k, v in debug.items():
            if k in {'pre_c_pop', 'pre_c_nop', 'pre_w_pop', 'pre_w_nop',
                     'post_c_push', 'post_c_nop', 'post_w_push', 'post_w_nop'}:
                debug[k] = {
                    'data': torch.stack(v, dim=1),
                    'fn': colored if not k.endswith('_') else None
                }
            elif k in {'post_c_val', 'post_w_val'}:
                def fn(x):
                    return colored(unproject(x))

                debug[k] = {
                    'data': torch.stack(v, dim=1),
                    'fn': lambda x: colored(unproject(x, return_sim=True)) if not k.endswith('_') else None
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

def format_number(num):
    """
    Formats a number with suffixes 'k', 'M', or 'B' for thousands, millions, and billions respectively.

    Parameters:
    - num (int): The number to format.

    Returns:
    - str: The formatted number as a string.
    """
    if abs(num) >= 1_000_000_000:  # Billion
        formatted_num = f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:  # Million
        formatted_num = f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:  # Thousand
        formatted_num = f"{num / 1_000:.1f}k"
    else:
        formatted_num = str(num)

    return formatted_num

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Params: {format_number(n_params)}')

# print('DISABLING GRADIENTS')
# model.control_sharp.requires_grad = False
# model.work_sharp.requires_grad = False
# model.control.requires_grad = False
# model.work.requires_grad = False
# model.cop1.requires_grad = False
# model.cop2.requires_grad = False
# model.cop3.requires_grad = False
# model.wop1.requires_grad = False
# model.wop2.requires_grad = False
# model.wop3.requires_grad = False

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
    if False: # TODO: rm experiment
        with torch.no_grad():
            model.control_sharp[:] = torch.randint(4, 12, (1,))
            model.work_sharp[:]    = torch.randint(4, 12, (1,))

    if True: # TODO: rm experiment
        LO = 10
        HI = 20
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
