'''.

Previous versions not working, so, testing my latest NAND and FwdNAND work.

----------
GOOD RECIPES:
  * DO NOT PROJECT SYMBOLS, merely interpolate them in/out
  * All layers NormalizedLinear; repeatedly anneal sharp from fuzzy to sharp;

  * Dumb things
    * Don't do Sigmoid into Softmax, duh. ReLU into softmax seems ok
    * Don't do NormalizedLinear into Sigmoid, duh.
    * Watch where you put Dropout. IE right after a Softmax? You're gonna break something.

  * Initializing symbols to choose from eliminates transient learning
    * I'm suspicious that symbols should not be learnable. Things started working better once they were not learnable.

----------
RESULTS:

* Memorization learning *finally* improved, a little, (in symbolic-only
  model) once I calculated output based on stack pop results, or stack peeks
  from END of for loop, instead of beginning.

  It also paid to increase # symbols being chosen after NAND. Too many yielded
  training memorization.

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
    * [X] Add Control Ops to dataset
    * [X] Incorporate new dataset
    * [X] Remove "Think" tokens for now
    * [X] Add Control signals to debug_state
    * [X] add to loss
    * [X] Freeze symbols, train selectors
    * [X] Train with/without control signals
    * [X] model.push_or_null & model.pop_or_null, do both per step

----------
TODO:

* [ ] Review each line for problems
* [ ] Hand code solution / Train sub components separately before full model run.

-----

* [ ] Normalize weights of Linears following Choices at the start of forward
* [ ] Freeze all params except Control Ops
* [ ] Apply Think tokens/noise outside datasaet (so each epoch is different)
* [ ] Inspiration via ResNet?
* [ ] Limit symbol selection
* [ ] Share symbol weights in different components
* [ ] Are we still projecting symbols somewhere? With an init sharpen of 10.0,
      it's happy to memorize with no generalization at all.
* [ ] is val loss right?
* [ ] Norm weight inits
* [ ] Different convolution operator than elem-multiply?
* [ ] How to combine type info with values? F.conv1d?
* [ ] `isinstance`. Add an "int identifier"/"tag"/"type". It looks like the
      stack ops are struggling to always accurately push/pop/nop given a given
      input.
* [ ] Identify paths of memorization!
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
from neurallambda.torch import NormalizedLinear, Fn, Parallel, Cat, Stack, Diagnose, Id, cosine_similarity, NuLinear, Choice
from neurallambda.util import print_grid, colored, format_number
import re
import pandas as pd
import itertools
import math
import warnings
from typing import Any, Iterable, List, Optional, Tuple
from neurallambda.nand import NAND, FwdNAND

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log/t04_addition_03')
LOG = True

torch.manual_seed(152)
torch.set_printoptions(precision=3, sci_mode=False)

DEBUG = False

PAUSE_IRRADIATION_P = 0.2
MAX_THINK_TOKENS    = 3 # "Think" tokens
PAUSE_APPEND_INDEX  = -3  # append tokens before [..., L, 42, R]

DEVICE = 'cuda'
VEC_SIZE = 128

BATCH_SIZE = 100
LR = 1e-2
WD = 0.0

INIT_SHARPEN = 10.0

NUM_EPOCHS = 60

# NAND_CLIP = 'leaky_relu'
NAND_CLIP = 'abs'
# NAND_CLIP = 'none'

NAND_BIAS = 3.0
# NAND_BIAS = None

REDUNDANCY = 2


##########
# More experimental knobs

USE_LOSS    = True
REG_OP      = False # regularization cheats
REG_VAL     = False # regularization cheats
NOISE_LVL   = 1e-2
CHEAT = False

SHARP_SCHED = 'random'
SHARP_SCHED = 'anneal'
SHARP_SCHED = None

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

all_symbols = tokens + [str(x) for x in range(10)]
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
        inputs_vec = [project(i) for i in inputs]
        num_padding = max_seq_len - len(inputs_vec)
        inputs_vec = torch.stack([start_data_vec] + [padding_vec] * num_padding + inputs_vec)
        out['Input'].append(inputs_vec)

        # Output
        outputs = item['Output']
        outputs_vec = torch.stack([padding_vec] * (num_padding + 1) + [project(x) for x in outputs])
        out['Output'].append(outputs_vec)

        # loss_mask
        loss_mask = item['LossMask']
        loss_mask_vec = torch.tensor([0] * (num_padding + 1) + loss_mask)
        out['LossMask'].append(loss_mask_vec)

        # stack projected vecs
        out['PreGlobalOp'].append   (torch.stack([nop_vec] * (num_padding + 1) + [project(x) for x in item['PreGlobalOp']]))
        out['PreWorkOp'].append     (torch.stack([nop_vec] * (num_padding + 1) + [project(x) for x in item['PreWorkOp']]))
        out['PostGlobalOp'].append  (torch.stack([nop_vec] * (num_padding + 1) + [project(x) for x in item['PostGlobalOp']]))
        out['PostGlobalVal'].append (torch.stack([wild_vec] * (num_padding + 1) + [project(x) for x in item['PostGlobalVal']]))
        out['PostWorkOp'].append    (torch.stack([nop_vec] * (num_padding + 1) + [project(x) for x in item['PostWorkOp']]))
        out['PostWorkVal'].append   (torch.stack([wild_vec] * (num_padding + 1) + [project(x) for x in item['PostWorkVal']]))

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

    # REMOVE START TOKENS
    if False:
        print('EXPERIMENT: REMOVING START TOKENS')
        out['Input'] = out['Input'][:, 1:]
        out['Output'] = out['Output'][:, 1:]
        out['LossMask'] = out['LossMask'][:, 1:]

    return out

# Create DataLoaders with the new collate_fn
train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
val_dl = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)


##################################################
# Training Functions

def run_epoch(epoch, model, dl, optimizer, device, train_mode=True):
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

            src  = data['Input']
            trg  = data['Output']
            mask = data['LossMask']

            # with torch.no_grad():
            #     trg_pre_c_op = data['PreGlobalOp']
            #     trg_pre_w_op = data['PreWorkOp']
            #     trg_post_c_op = data['PostGlobalOp']
            #     trg_post_c_val = data['PostGlobalVal']
            #     trg_post_w_op = data['PostWorkOp']
            #     trg_post_w_val = data['PostWorkVal']

            if train_mode:
                optimizer.zero_grad()

            ##########
            # Add Noise
            #   TODO: rm experiment.
            if NOISE_LVL is not None:
                with torch.no_grad():
                    src = src + torch.randn_like(src) * NOISE_LVL
                    # trg = trg + torch.randn_like(trg) * NOISE_LVL

            output, debug = model(src)

            # pre_c_pop = debug['pre_c_pop']['data']
            # pre_c_nop = debug['pre_c_nop']['data']
            # pre_w_pop = debug['pre_w_pop']['data']
            # pre_w_nop = debug['pre_w_nop']['data']
            # post_c_push = debug['post_c_push']['data']
            # post_c_val = debug['post_c_val']['data']
            # post_c_nop = debug['post_c_nop']['data']
            # post_w_push = debug['post_w_push']['data']
            # post_w_val = debug['post_w_val']['data']
            # post_w_nop = debug['post_w_nop']['data']

            loss = 0

            if USE_LOSS:
                loss = loss + ((1 - cosine_similarity(output, trg, dim=2)) * mask).mean()
                # loss = loss + ((1 - cosine_similarity(output[:,-3:], trg[:,-3:], dim=2))).mean()

            # REGULARIZATION EXPERIMENT
            #   TODO: rm experiment
            if REG_OP:

                ##########
                # Operation Loss
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

                loss = loss + op_loss * 0.05

            if REG_VAL:
                ##########
                # Val Loss
                val_loss = (
                    (1 - cosine_similarity(trg_post_c_val, post_c_val, dim=2))
                    + (1 - cosine_similarity(trg_post_w_val, post_w_val, dim=2))
                ).mean()

                loss = loss + val_loss * 0.05

            if train_mode:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()


    # Log weight histogram
    if LOG and train_mode:
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'weights/{name}', param.data.cpu().numpy(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f'grads/{name}', param.grad.data.cpu().numpy(), epoch)

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
        elif in_answer and symbol in {str(i) for i in range(10)}: # explicitly check str of ints
            integers.append(symbol)
    return integers

def up(x, return_sim=False):
    ''' Unproject, but if it's too dissimilar, return '*' instead '''
    y, sim = unproject(x.to('cuda'), return_sim=True)
    if sim > 0.3:
        out = y
    else:
        out = '*'

    if return_sim:
        return out, sim
    else:
        return out

def debug_output(src, trg, output, mask, batch_idx, states):
    """Print debug information for the first element of the batch."""
    # Unproject and display the sequences for debugging
    ins_seq = [up(x.to('cuda')) for x in src[batch_idx]]  # List of symbols
    trgs_seq = [up(x.to('cuda')) for x in trg[batch_idx]]  # List of symbols
    outs_seq = [up(x.to('cuda')) for x in output[batch_idx]]  # List of symbols
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
                    correct_predictions += sum([p==t for p,t in zip(predicted_integers, target_integers)])
                else:
                    total_predictions += max(len(predicted_integers), len(target_integers))
                if debug and batch_idx < 1:
                    debug_output(src, trg, output, mask, batch_idx, states)
    acc = correct_predictions / total_predictions if total_predictions > 0 else 0
    return acc


##################################################

##########
# LSTM

class LSTMModel(nn.Module):
    def __init__(self, vec_size):
        super(LSTMModel, self).__init__()
        self.vec_size = vec_size
        self.lc1 = nn.LSTMCell(vec_size, vec_size)
        self.lc2 = nn.LSTMCell(vec_size, vec_size)
        self.fc = NormalizedLinear(vec_size, vec_size)

    def forward(self, x):
        # Initialize the hidden and cell states to zeros
        h1 = torch.zeros(x.size(0), self.vec_size).to(x.device)
        c1 = torch.zeros(x.size(0), self.vec_size).to(x.device)
        h1s = []

        for i in range(x.size(1)):
            h1, c1 = self.lc1(x[:, i, :], (h1, c1))
            h1s.append(h1)

        # Run LSTM2
        h1s = torch.stack(h1s, dim=1) # input to lstm2

        h2 = torch.zeros(x.size(0), self.vec_size).to(x.device)
        c2 = torch.zeros(x.size(0), self.vec_size).to(x.device)
        h2s = []

        for i in range(x.size(1)):
            h2, c2 = self.lc2(h1s[:, i, :], (h2, c2))
            h2s.append(h2)

        # Stack the outputs (hidden states) from each time step
        outputs = torch.stack(h2s, dim=1)

        # Apply the linear layer to each time step's output
        # We reshape the outputs tensor to (-1, self.vec_size) before applying the linear layer
        # Then reshape it back to (batch_size, seq_len, vec_size)
        out = self.fc(outputs.view(-1, self.vec_size))
        out = out.view(-1, x.size(1), self.vec_size)

        return out, {}


##################################################
#
# NeuralSymbol
#


def op(n_inp, vec_size):
    ''' (vec, vec, vec) -> scalar
    Useful in stack/queue operations.
    '''

    H = 64

    return nn.Sequential(

        # Fn(lambda x: torch.cat(x, dim=-1)),
        nn.Linear(n_inp * vec_size, H),
        nn.ReLU(),
        nn.Linear(H, 2),
        nn.Sigmoid(),
        Fn(lambda x: (x[:,0], x[:,1]))

        # NAND(vec_size, n_inp, 2 * REDUNDANCY, clip=NAND_CLIP),
        # Fn(lambda x: (x[:,:REDUNDANCY].max(dim=-1).values,
        #               x[:,REDUNDANCY:].max(dim=-1).values
        #               ))

        # NAND(vec_size, n_inp, (2 + 1) * REDUNDANCY, clip=NAND_CLIP),
        # Fn(lambda x: (x[:,:REDUNDANCY].max(dim=-1).values,
        #               x[:,REDUNDANCY:2*REDUNDANCY].max(dim=-1).values
        #               ))

        # Fn(lambda x: (x[:,:REDUNDANCY].max(dim=-1).values,
        #               x[:,REDUNDANCY:].max(dim=-1).values
        #               ))
    )


class Neuralsymbol(nn.Module):
    def __init__(self, vec_size ):
        super(Neuralsymbol, self).__init__()

        init_sharpen = INIT_SHARPEN
        n_control_stack = 4
        n_work_stack = 4

        vec_size = vec_size
        self.vec_size = vec_size

        self.vec_size       = vec_size
        self.vec_size      = vec_size
        self.n_control_stack = n_control_stack
        self.n_work_stack    = n_work_stack

        self.zero_offset = 1e-6


        ##########
        # Type

        # n_choices = 3
        # self.sym_type_outs = nn.Parameter(torch.randn(vec_size, n_choices))

        # self.sym_type_choose = NAND(
        #     vec_size, 1, n_choices=n_choices,
        #     redundancy=REDUNDANCY,
        #     method='max'
        # )


        ##########
        # Control Stack

        # self.stack_init_vec = nn.Parameter(torch.randn(vec_size) * 1e-2)
        self.stack_init_vec = F.normalize(torch.randn(vec_size), dim=0)

        self.control_stack = S.Stack(n_control_stack, vec_size)
        self.control_sharp = nn.Parameter(torch.tensor([init_sharpen]))
        self.pre_c_op = op(3, vec_size)
        self.post_c_op = op(3, vec_size)

        n_vecs = 3

        # n_choices = 6
        # self.post_c_outputs = nn.Parameter(torch.randn(vec_size, n_choices))
        # with torch.no_grad(): self.post_c_outputs[:] = F.normalize(self.post_c_outputs, dim=0)
        # self.post_c_choice = NAND(vec_size, n_vecs, n_choices=n_choices, clip=NAND_CLIP)

        HHH = 64
        self.post_c_outputs = nn.Sequential(
            Fn(lambda x: torch.cat(x, dim=-1)),
            nn.Linear(vec_size * n_vecs, HHH),
            nn.ReLU(),
            nn.Linear(HHH, vec_size),
            nn.Tanh()
        )


        ##########
        # Working Stack

        self.work_stack = S.Stack(n_work_stack, vec_size)
        self.work_sharp = nn.Parameter(torch.tensor([init_sharpen]))
        self.pre_w_op = op(3, vec_size)
        self.post_w_op = op(3, vec_size)

        # # Choose calculation
        # n_choices = 11

        # self.post_w_val_outputs = nn.Parameter(torch.randn(vec_size, n_choices))
        # with torch.no_grad(): self.post_w_val_outputs[:] = F.normalize(self.post_w_val_outputs, dim=0)
        # self.post_w_val_choice = NAND(vec_size, 2, n_choices * 10, clip=NAND_CLIP)

        # # Choose between calculation and defaults
        # n_vecs = 3
        # n_choices = 4
        # self.post_w_outputs = nn.Parameter(torch.randn(vec_size, n_choices))
        # with torch.no_grad(): self.post_w_outputs[:] = F.normalize(self.post_w_outputs, dim=0)

        # self.post_w_choice = NAND(vec_size, n_vecs, (n_choices + 2) * REDUNDANCY, clip=NAND_CLIP)


        HHH = 64
        self.work_hack = nn.Sequential(
            Fn(lambda x: torch.cat(x, dim=-1)),
            nn.Linear(vec_size * 3, HHH),
            nn.ReLU(),
            nn.Linear(HHH, vec_size),
            nn.Tanh()
        )

        ##########
        # Select output

        n_vecs = 4

        # n_choices = 5
        # self.select_out_outputs = nn.Parameter(torch.randn(vec_size, n_choices))
        # with torch.no_grad(): self.select_out_outputs[:] = F.normalize(self.select_out_outputs, dim=0)

        # self.select_out_choice = NAND(vec_size, n_vecs, n_choices=n_choices + 2, clip=NAND_CLIP)
        self.out = nn.Sequential(
            Fn(lambda x: torch.cat(x, dim=-1)),
            nn.Linear(vec_size * n_vecs, HHH),
            nn.ReLU(),
            nn.Linear(HHH, vec_size),
            nn.Tanh(),
        )

    def forward(self, x):
        # Containers Init
        device = x.device
        batch_size = x.shape[0]
        self.control_stack.init(batch_size, self.zero_offset, device)
        self.work_stack.init(batch_size, self.zero_offset, device)

        # Push first instruction
        self.stack_init_vec = self.stack_init_vec.to(device=device)
        self.control_stack.push(self.stack_init_vec.expand(batch_size, -1))
        self.work_stack.   push(self.stack_init_vec.expand(batch_size, -1))

        # Debugging
        pops = []
        debug = {
            'pre_c_nop': [],
            'pre_c_pop': [],

            'pre_w_nop': [],
            'pre_w_pop': [],

            'post_c_nop': [],
            'post_c_push': [],
            'post_c_val': [],

            'post_w_nop': [],
            'post_w_push': [],
            'post_w_val': [],

        }

        # Loop over sequence
        outputs = []
        for i in range(x.size(1)):
            inp = x[:, i, :]

            ##########
            # Pre

            # # inp_typ = inp
            # inp_type_c = self.sym_type_choose([inp])
            # inp_typ = einsum('vc, bc -> bv', self.sym_type_outs, inp_type_c)

            c_peek = self.control_stack.read()
            w_peek = self.work_stack.read()
            pre_inp = [c_peek, w_peek, inp]

            # control stack maybe pop
            pre_c_pop, pre_c_nop = self.pre_c_op(pre_inp)
            self.control_stack.pop_or_null_op(self.control_sharp, pre_c_pop, pre_c_nop)

            # work stack maybe pop
            pre_w_pop, pre_w_nop = self.pre_w_op(pre_inp)
            self.work_stack.pop_or_null_op(self.work_sharp, pre_w_pop, pre_w_nop)


            ##########
            # Post

            # control stack
            post_c_inp = [c_peek, w_peek, inp]
            post_c_push, post_c_nop = self.post_c_op(post_c_inp)

            # control_choice = self.post_c_choice(post_c_inp) # [batch, n_choice]
            # control_val = einsum('bc, vc -> bv', control_choice, self.post_c_outputs)

            control_val = self.post_c_outputs(post_c_inp)

            self.control_stack.push_or_null_op(self.control_sharp, post_c_push, post_c_nop, control_val)

            # work stack
            post_w_inp = [c_peek, w_peek, inp]
            post_w_push, post_w_nop = self.post_w_op(post_w_inp)

            work_val = self.work_hack(post_w_inp)


            # # "sum mod 10" happens here
            # post_w_val_choice_inp = [w_peek, inp]
            # # post_w_val_choice_inp = [w_peek * inp]  # led to overfitting?

            # work_val_choice =self.post_w_val_choice(post_w_val_choice_inp) # [batch, n_choice]
            # work_val_option = einsum('bcr, vc -> bv',
            #                          work_val_choice.view(batch_size, self.post_w_val_choice.n_choices // 10, 10),
            #                          self.post_w_val_outputs)

            # post_w_choice_inp = [c_peek, w_peek, inp]
            # work_choice = self.post_w_choice(post_w_choice_inp) # [batch, n_choice]
            # opts = torch.cat([
            #     self.post_w_outputs.expand(batch_size, -1, -1),
            #     work_val_option.unsqueeze(2),
            #     work_val_option.unsqueeze(2), # for redundancy
            # ], dim=2)
            # work_val = einsum('bcr, bvc -> bv',
            #                   work_choice.view(batch_size, self.post_w_choice.n_choices // REDUNDANCY, REDUNDANCY),
            #                   opts)


            self.work_stack.push_or_null_op(self.work_sharp, post_w_push, post_w_nop, work_val)


            ##########
            # Select out

            out_inp = [c_peek, w_peek, control_val, work_val]

            # select_choice = self.select_out_choice(out_inp) # [batch, n_choice]
            # stck = torch.cat([
            #     # [vec_size, n_choice-1] -> [batch, vec_size, n_choice-1]
            #     self.select_out_outputs.expand(batch_size, -1, -1),
            #     # [batch, vec_size] -> [batch, vec_size, 1]
            #     work_val.unsqueeze(2),
            #     work_val.unsqueeze(2) # for redundancy
            # ],
            # dim=2)
            # out = einsum('bc, bvc -> bv', select_choice, stck)

            out = self.out(out_inp)

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

        outputs = torch.stack(outputs, dim=1)

        for k, v in debug.items():
            if k in {'pre_c_pop', 'pre_c_nop', 'pre_w_pop', 'pre_w_nop',
                     'post_c_push', 'post_c_nop', 'post_w_push', 'post_w_nop'}:
                debug[k] = {
                    'data': torch.stack(v, dim=1),
                    'fn': colored if not k.endswith('_') else None
                }
            elif k in {'post_c_val', 'post_w_val'}:
                def fn(x):
                    return colored(up(x))

                debug[k] = {
                    'data': torch.stack(v, dim=1),
                    'fn': lambda x: colored(up(x, return_sim=True)) if not k.endswith('_') else None
                }
        return outputs, debug


##################################################
#
# Cyborg
#

class Cyborg(nn.Module):
    def __init__(self, vec_size, ):
        super(Cyborg, self).__init__()
        self.vec_size = vec_size

        N_STACKS = 2
        self.N_STACKS = N_STACKS

        self.REDUNDANCY = 2
        H = 30  # number of symbols
        self.H = H

        # Shared Symbol Table
        self.syms = torch.randn((H, vec_size))  # frozen, must be set externally

        self.calc_out = NAND(vec_size,
                             1 + N_STACKS,
                             (H + N_STACKS) * self.REDUNDANCY,
                             clip=NAND_CLIP, nand_bias=NAND_BIAS)

        # STACKS
        stack_depth = 4
        init_sharpen = INIT_SHARPEN
        self.zero_offset = 1e-6
        self.stack_init_vec = F.normalize(torch.randn(vec_size), dim=0)

        self.stacks = nn.ParameterList([])
        for _ in range(N_STACKS):
            stack = S.Stack(stack_depth, vec_size)
            sharp = nn.Parameter(torch.tensor([init_sharpen]))
            pop_op  = NAND(vec_size, 1 + N_STACKS, 2 * self.REDUNDANCY, clip=NAND_CLIP, nand_bias=NAND_BIAS)
            push_op = NAND(vec_size, 1 + N_STACKS, 2 * self.REDUNDANCY, clip=NAND_CLIP, nand_bias=NAND_BIAS)
            push    = NAND(vec_size, 1 + N_STACKS, (H + N_STACKS) * self.REDUNDANCY, clip=NAND_CLIP, nand_bias=NAND_BIAS)
            params = nn.ParameterList([stack, sharp, pop_op, push_op, push])
            self.stacks.append(params)

        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        self.stack_init_vec = self.stack_init_vec.to(device=device)
        self.syms = self.syms.to(device=device)

        # Stacks prep
        for (stack, _, _, _, _) in self.stacks:
            stack.init(batch_size, self.zero_offset, device)
            stack.push(self.stack_init_vec.expand(batch_size, -1))

        # RNN prep
        h1 = torch.zeros(x.size(0), self.vec_size).to(x.device)

        outputs = []

        for i in range(x.size(1)):
            inp = x[:, i, :]

            # Peek Stacks
            peeks = []
            for (stack, _, _, _, _) in self.stacks:
                peeks.append(stack.read())

            inps = torch.cat([inp] + peeks, dim=-1)

            out_opts = torch.cat([self.syms.expand(batch_size, -1, -1)] +
                                 [p.unsqueeze(1) for p in peeks]
                                 , dim=1)

            # Handle Stacks
            all_pops = []
            for (stack, sharp, pop_op, push_op, push_fn) in self.stacks:
                pops = pop_op(inps)
                pushes = push_op(inps)
                push = einsum('bcv, bcr -> bv',
                              out_opts,
                              (push_fn(inps)).view(batch_size, (self.H + self.N_STACKS), self.REDUNDANCY)
                              )
                # push = self.dropout(push)
                p = stack.pop_or_null_op(sharp,
                                         pops[:,:self.REDUNDANCY].max(dim=1).values,
                                         pops[:,self.REDUNDANCY:].max(dim=1).values)
                all_pops.append(p)
                stack.push_or_null_op(sharp,
                                      pushes[:,:self.REDUNDANCY].max(dim=1).values,
                                      pushes[:,self.REDUNDANCY:].max(dim=1).values,
                                      push)

            # Peek Again
            peek_agains = []
            for (stack, _, _, _, _) in self.stacks:
                peek_agains.append(stack.read())

            # Output
            inps = torch.cat([inp] + peek_agains, dim=-1)

            out_opts_again = torch.cat([self.syms.expand(batch_size, -1, -1)] +
                                 [p.unsqueeze(1) for p in peek_agains]
                                 , dim=1)

            out    = einsum('bcv, bcr -> bv',
                            out_opts_again,
                            (self.calc_out(inps)).view(batch_size,
                                                     self.H + self.N_STACKS,
                                                     self.REDUNDANCY)
                            )

            # out = self.dropout(out)
            outputs.append(out)


        outputs = torch.stack(outputs, dim=1)
        return outputs, {}


##################################################
# Training

##########
# Setup

# MyModel = LSTMModel
MyModel = Cyborg
# MyModel = Neuralsymbol

model = MyModel(
    vec_size=VEC_SIZE,
)
model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Params: {format_number(n_params)}')


##########
# Hand code solution

def set_weights(nand: nn.Module,
                out_vecs: torch.Tensor,
                mappings: List[Tuple[Tuple[str, str], Tuple[int, int], str]],
                redundancy,
                confidence=3):
    '''set weights in nand and out_vecs according to mappings. confidence scales
    the initial nand_weight, which gets sigmoided. When confidence=3,
    sigmoid(3)=0.953, which is confident but with reasonable gradients.

    Redundancy is handled by view-wrapping tensors, ie, redundant elems are not
    in adjacent indices.

    '''

    n_choices = nand.weight.size(0) // redundancy

    with torch.no_grad():

        nand.nand_weight[:, :] = torch.tensor([confidence])

        temp_weight = torch.zeros(n_choices, redundancy, nand.weight.size(1), device=nand.weight.device)
        temp_nand_weight = torch.zeros(n_choices, redundancy, nand.nand_weight.size(1), device=nand.nand_weight.device)

        for choice_i, (vals, out) in enumerate(mappings):
            out_vecs[:, choice_i] = project(out) # no redundancy

            # add all the vecs into the matcher vec
            for r_i_, (ins, nand_weight) in enumerate(vals):
                # TODO: modulo redundancy is buggy in nand_weight if it wraps around
                r_i = r_i_ % redundancy
                temp_weight[choice_i, r_i, :] += torch.concat([project(x) for x in ins])
                temp_nand_weight[choice_i, r_i, :] = torch.tensor([confidence if x==1 else -confidence for x in nand_weight]).to(DEVICE)


        upto = len(mappings) * redundancy  # don't overwrite rows for which no mappings were set
        nand.weight[:upto] = temp_weight.flatten(start_dim=0, end_dim=1)[:upto]
        nand.nand_weight[:upto] = temp_nand_weight.flatten(start_dim=0, end_dim=1)[:upto]

def reset_cheat(model):
    # CONTROL
    # model.pre_c_op
    # model.post_c_op
    # model.post_c_choice

    # WORK
    # model.pre_w_op
    # model.post_w_op

    # model.post_w_val_choice
    set_weights(
        model.post_w_val_choice,
        model.post_w_val_outputs,
        [
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==0], '0'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==1], '1'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==2], '2'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==3], '3'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==4], '4'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==5], '5'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==6], '6'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==7], '7'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==8], '8'),
            ([((str(i), str(j)), (1, 1)) for i in range(10) for j in range(10) if (i+j)%10==9], '9'),

            # default
            # ([(('X', 'X'), (0, 0))]*10, 'X'),
        ], 10, confidence=5)

    # model.post_w_choice
    # model.post_w_outputs

    # set_weights(
    #     model.post_w_choice,
    #     model.post_w_outputs,
    #     [
    #         ([((), (1, 1), )]),
    #     ])


    # SELECT
    # model.select_out_choice

    pass


# @@@@@@@@@@
# SET CYBORG SYMBOLS (bc they're frozen)

# for (stack, sharp, pop_op, push_op, push_fn) in model.stacks:
# model.calc_out

with torch.no_grad():
    n_sym = model.syms.size(0)
    for i, c in enumerate(all_symbols):
        ii = i % n_sym
        model.syms[ii] = project(c)


# @@@@@@@@@@

# print("TESTING CHEAT")
# reset_cheat(model)

# for i in [1, 2, 3]:
#     for j in [1, 2, 3]:
#         inp = torch.cat([project(str(i)), project(str(j))])
#         choice = model.post_w_val_choice(inp.unsqueeze(0))
#         out = einsum('bcr, vc -> bv',
#                      choice.view(1, model.post_w_val_outputs.size(1), 10),
#                      model.post_w_val_outputs)
#         v, sim = unproject(out.squeeze(0), return_sim=True)
#         print(f'{i}+{j}={v}, {sim.item():>.3f}') # , {choice.cpu().detach()}')

# # WEIGHT VALS
# for i in range(10):
#     for vi in [0, VEC_SIZE]:
#         v, sim = unproject(model.post_w_val_choice.weight[i, vi:vi+VEC_SIZE], return_sim=True)
#         print(f'{v}, {sim.item():>.3f}', end='  |  ')
#     print(model.post_w_val_choice.nand_weight[i].cpu().tolist())
#     # v, sim = unproject(model.post_w_val_outputs[:, i], return_sim=True)
#     # print(f'{v}, {sim.item():>.3f}')

# # OUTPUT VALS
# for i in range(12):
#     print(unproject(model.post_w_val_outputs[:, i], return_sim=True))

# BRK

# @@@@@@@@@@




##########
# Go

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

SHARP = 5.0
with torch.no_grad():
    model.control_sharp[:] = SHARP
    model.work_sharp[:] = SHARP


# RESET WEIGHTS
with torch.no_grad():
    model.pre_c_nop[0].ff.weight[:] = torch.randn_like(model.pre_c_nop[0].ff.weight) * 1e-3
    model.pre_w_nop[0].ff.weight[:] = torch.randn_like(model.pre_w_nop[0].ff.weight) * 1e-3
    model.pre_c_pop[0].ff.weight[:] = torch.randn_like(model.pre_c_pop[0].ff.weight) * 1e-3
    model.pre_w_pop[0].ff.weight[:] = torch.randn_like(model.pre_w_pop[0].ff.weight) * 1e-3


USE_LOSS = True
REG_OP = False
REG_VAL = False
NOISE_LVL = 1e-0

SHARP_SCHED = 'random'
SHARP_SCHED = 'anneal'
SHARP_SCHED = None

optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-1)
optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-2)
optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-3)
optimizer = optim.Adam(opt_params, weight_decay=WD, lr=1e-4)

'''

for epoch in range(NUM_EPOCHS):
    if CHEAT:
        print('cheat ', end='')
        reset_cheat(model)

    train_loss = run_epoch(epoch, model, train_dl, optimizer, DEVICE, train_mode=True)

    # Experiment changing sharpen param
    if SHARP_SCHED == 'random':
        with torch.no_grad():
            model.control_sharp[:] = torch.randint(5, 30, (1,))
            model.work_sharp[:]    = torch.randint(5, 30, (1,))

    if SHARP_SCHED == 'anneal':
        LO = 5
        HI = 20
        with torch.no_grad():
            a = epoch / NUM_EPOCHS
            model.control_sharp[:] = (1-a) * LO + a * HI
            model.work_sharp[:]    = (1-a) * LO + a * HI

    if epoch % 10 == 0:
        val_loss = run_epoch(epoch, model, val_dl, None, DEVICE, train_mode=False)
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
# tacc = accuracy(model, train_dl, DEVICE, debug=True)
