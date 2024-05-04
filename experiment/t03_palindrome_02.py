'''

1. Moving toward a common interface for building and testing models

2. Get palindrome to work on untrained embeddings


TODO:
- [x] trace stack ops
- [x] emit outputs from embedding vocab + symbolic (trace)
- [ ] playground that others can use

'''

import torch
import random

SEED = 43
torch.manual_seed(SEED)
random.seed(SEED)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import Dataset
from functools import partial
from tokenizers import Tokenizer, AddedToken
from torch import cosine_similarity, einsum
from torch.nn.functional import elu, selu, gelu, leaky_relu
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from typing import List, Any, Dict
import matplotlib.pyplot as plt
import neurallambda.latch as L
import neurallambda.stack as S
import numpy as np

import tokenizers


##################################################
# Data concerns (this work is duplicated in `demo/common.py`

def print_grid(data):
    # maximum width for each column
    column_widths = [max(len(str(item)) for item in column) for column in zip(*data)]
    for row in data:
        formatted_row = "  ".join(str(item).ljust(width) for item, width in zip(row, column_widths))
        print(formatted_row)


def print_model_info(model):
    print('------------')
    print('MODEL PARAMS')
    info = []
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        info.append((name, param_count, f'{list(param.shape)}', f'grad:{param.requires_grad}'))
    print_grid(info)
    print(f'Total Parameters: {total_params:,}')


def iterator(all_datasets: List[Dataset], keys: List[str], special_tokens):
    ''' Iterate over a list of datasets and build a vocab for them. '''
    for tok in special_tokens:
        yield tok
    for dataset in all_datasets:
        for xs in dataset:
            for key in keys:
                for x in xs[key]:
                    yield x


def collate_fn(pad_token, batch):
    ''' Tokenize and pad batches of data. '''

    # is_pretokenized expects `isinstance(sample['inputs'], List[str])`
    input_ids  = [tokenizer.encode(sample['inputs'], is_pretokenized=True).ids for sample in batch]
    output_ids = [tokenizer.encode(sample['outputs'], is_pretokenized=True).ids for sample in batch]

    # Get the maximum sequence length in the batch
    max_length = max(len(ids) for ids in input_ids + output_ids)

    # Pad the sequences on the left side
    input_ids  = [F.pad(torch.tensor(ids), (max_length - len(ids), 0), value=tokenizer.token_to_id(pad_token)) for ids in input_ids]
    output_ids = [F.pad(torch.tensor(ids), (max_length - len(ids), 0), value=tokenizer.token_to_id(pad_token)) for ids in output_ids]

    # Stack the padded sequences into tensors
    input_ids = torch.stack(input_ids)
    output_ids = torch.stack(output_ids)

    return input_ids, output_ids


def dataloader_info(dataloader, tokenizer):
    ''' Print debug info about data, and get histogram of sequence length. '''
    import matplotlib.pyplot as plt

    # PRINT EXAMPLES
    for batch_idx, batch in enumerate(dataloader):
        input_ids, output_ids = batch

        # a random sample from the batch
        sample_idx = random.randint(0, input_ids.size(0) - 1)
        sample_input_ids = input_ids[sample_idx].tolist()
        sample_output_ids = output_ids[sample_idx].tolist()

        # decode
        #   NOTE: map decode since these ids are symbols in a list, not a string
        sample_input_tokens  = [tokenizer.decode([x], skip_special_tokens=True) for x in sample_input_ids]
        sample_output_tokens = [tokenizer.decode([x], skip_special_tokens=True) for x in sample_output_ids]

        print(f"Batch {batch_idx + 1}:")
        print_grid([
            ['Input Tokens:'] + sample_input_tokens,
            ['Output Tokens:'] + sample_output_tokens,
            ['Input IDs:'] + list(map(str, sample_input_ids)),
            ['Output IDs:'] + list(map(str, sample_output_ids))
        ])
        print()
        if batch_idx >= 4:
            break

    # STATISTICS
    sequence_lengths = []
    batch_sizes = []
    for batch in dataloader:
        input_ids, _ = batch
        batch_sizes.append(input_ids.size(0))
        sequence_lengths.append(input_ids.size(1))

    print("DATALOADER STATISTICS:")
    print(f"number of batches: {len(batch_sizes)}")
    print(f"avg batch size: {sum(batch_sizes) / len(batch_sizes):.2f}")
    print(f"min padded sequence length: {min(sequence_lengths)}")
    print(f"max padded sequence length: {max(sequence_lengths)}")
    print(f"avg padded sequence length: {sum(sequence_lengths) / len(sequence_lengths):.2f}")

    # histogram of sequence lengths
    plt.figure(figsize=(8, 6))
    plt.hist(sequence_lengths, bins=20, edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sequence Lengths')
    plt.show()


def build_tokenizer_dataloader(
        raw_datasets: List[ # multiple datasets
            List[Dict[str, List[str]]] # a single dataset
        ]):
    for data in raw_datasets:
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        assert 'inputs' in data[0]
        assert 'outputs' in data[0]

    # Convert to HF Dataset
    hf_datasets = [Dataset.from_list(x) for x in raw_datasets]

    # Make Tokenizer from Data
    UNK_TOKEN = '[UNK]'
    PAD_TOKEN = '[PAD]'
    special_tokens = [UNK_TOKEN, PAD_TOKEN]
    # init tokenizer
    tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab={}, unk_token=UNK_TOKEN))
    print('training tokenizer')
    tokenizer.train_from_iterator(iterator(hf_datasets, ['inputs', 'outputs'], special_tokens))
    tokenizer.add_special_tokens([UNK_TOKEN, PAD_TOKEN])
    dataloaders = [
        DataLoader(x, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, PAD_TOKEN))
        for x in hf_datasets
    ]

    return tokenizer, dataloaders


class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0, hard=False, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard
        self.dim = dim

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=self.dim)


##################################################
# Palindrome

DEVICE = 'cuda'
VEC_SIZE = 256

MIN_LENGTH = 8

TRAIN_NUM_SAMPLES = 2000
TRAIN_MAX_SEQUENCE_LENGTH = 10

VAL_NUM_SAMPLES = 200
VAL_MAX_SEQUENCE_LENGTH = 20

N_STACK = 20
INITIAL_SHARPEN = 5.0

LR = 1e-2
BATCH_SIZE = 32

NUM_EPOCHS = 12
GRAD_CLIP = None

# Symbols that go into palindrome
train_lang = 'a b c d e f g h i j'.split(' ')
# val_lang   = 'a b c d e f g h i j'.split(' ')
val_lang   = 'k l m n o p q r s t'.split(' ')


##########
# DATA

BOS_SYMBOL = '^'
REFLECT_SYMBOL = '|'
PAUSE_SYMBOL = '.'

def palindrome(num_samples, min_length, max_length, lang) -> Dict[str, List[str]]:
    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        hlength = length // 2
        seed = [random.choice(lang) for _ in range(hlength)]
        # add pauses to inputs and outputs
        inputs  = [BOS_SYMBOL] + seed + [REFLECT_SYMBOL] + [PAUSE_SYMBOL] * hlength
        outputs = [PAUSE_SYMBOL] * (hlength + 2) + seed[::-1]
        # convert all symbols to str
        inputs  = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'seed': seed,
            'inputs': inputs,
            'outputs': outputs,
        })
    return data

train_raw = palindrome(TRAIN_NUM_SAMPLES, MIN_LENGTH, TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang)
train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))

val_raw = palindrome(VAL_NUM_SAMPLES, MIN_LENGTH, VAL_MAX_SEQUENCE_LENGTH, lang=val_lang)
val_raw = sorted(val_raw, key=lambda x: len(x['inputs']))

tokenizer, (train_dl, val_dl) = build_tokenizer_dataloader([train_raw, val_raw])

# dataloader_info(train_dl, tokenizer)


##########
# LSTM

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tokenizer):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, input_dim)

        self.lc1 = nn.LSTMCell(input_dim, hidden_dim)
        self.lc2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x_ids):
        x = model.embeddings(x_ids)
        # init
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        c1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []
        for i in range(x.size(1)):
            h1, c1 = self.lc1(x[:, i, :], (h1, c1))
            h1s.append(h1)
        h1s = torch.stack(h1s, dim=1) # [batch, seq, dim]
        h2 = torch.zeros(x.size(0), self.hidden_dim).to(x.device) # [batch, dim]
        c2 = torch.zeros(x.size(0), self.hidden_dim).to(x.device) # [batch, dim]
        outputs = []
        for i in range(x.size(1)):
            h2, c2 = self.lc2(h1s[:, i], (h2, c2))
            output = self.fc(h2)
            output = F.softmax(output, dim=1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tokenizer):
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, input_dim)

        self.lc1 = nn.RNNCell(input_dim, hidden_dim)
        self.lc2 = nn.RNNCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x_ids):
        x = model.embeddings(x_ids)
        # init
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []
        for i in range(x.size(1)):
            h1 = self.lc1(x[:, i, :], h1)
            h1s.append(h1)
        h1s = torch.stack(h1s, dim=1) # [batch, seq, dim]
        h2 = torch.zeros(x.size(0), self.hidden_dim).to(x.device) # [batch, dim]
        outputs = []
        for i in range(x.size(1)):
            h2 = self.lc2(h1s[:, i], h2)
            output = self.fc(h2)
            output = F.softmax(output, dim=1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class RNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tokenizer):
        super(RNNStack, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, input_dim)

        self.rnn_i = nn.Linear(input_dim, hidden_dim, bias=False)
        self.rnn_h = nn.Linear(hidden_dim, hidden_dim, bias=False)


        ##########
        # Outputs

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.vocab_size, bias=False),
            nn.Softmax(dim=1)
            # GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        )

        self.choose = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2, bias=False),
            # nn.Softmax(dim=1),
            # GumbelSoftmax(dim=1, temperature=1.0, hard=True),
            GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        )

        ##########
        # Stack
        self.n_stack = N_STACK
        self.initial_sharpen = INITIAL_SHARPEN
        self.zero_offset = 0
        self.stack_vec_size = input_dim
        self.sharp = nn.Parameter(torch.tensor([self.initial_sharpen]))

        self.ops = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim * 2, 3, bias=False), # rnn hidden -> push,pop,nop
            # nn.Softmax(dim=1)
            # GumbelSoftmax(dim=1, temperature=1.0, hard=True),
            GumbelSoftmax(dim=1, temperature=1.0, hard=False),
        )

    def forward(self, x_ids, debug=False):
        x = model.embeddings(x_ids)
        batch_size, device, dtype = x.size(0), x.device, x.dtype
        # init
        ss = S.initialize(self.input_dim, self.n_stack, batch_size, self.zero_offset + 1e-3, device, dtype=dtype)
        h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        outputs = []
        if debug:
            ops_trace = []
        for i in range(x.size(1)):
            ri = self.rnn_i(F.normalize(x[:, i], dim=1))
            rh = self.rnn_h(h)
            h = (ri + rh).tanh()

            hmod = 1 - h.relu()
            hmod = torch.cat([hmod, hmod], dim=1)

            # Stack
            ops = self.ops(hmod)
            if debug:
                ops_trace.append(ops)
            push, pop, nop = [o.squeeze(1) for o in torch.chunk(ops, chunks=3, dim=-1)]
            ss, pop_val = S.push_pop_nop(ss, self.sharp, push, pop, nop, x[:, i])

            # Outputs
            semantic = self.fc(hmod)
            # OPTIM: will this be memory hog?
            syntactic = torch.cosine_similarity(pop_val.unsqueeze(1), # [B, 1, D]
                                                self.embeddings.weight.unsqueeze(0), # [1, V, D]
                                                dim=2) # [B, V]
            choice = self.choose(hmod)
            output = torch.einsum('bc, bcv -> bv', choice, torch.stack([semantic, syntactic], dim=1))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        if debug:
            ops_trace = torch.stack(ops_trace, dim=1)
            debug_out = {'ops_trace': ops_trace}
            return outputs, debug_out
        else:
            return outputs



##################################################
# Training

def train_epoch(model, dl, optimizer, device, clip=None):
    '''
    Args:
      warm_up: ignore loss for this many steps
    '''
    model.train()
    epoch_loss = 0

    for i, (src_ids, trg_ids) in enumerate(dl):
        src_ids = src_ids.to(device)  # [batch, seq, vec_size]
        trg_ids = trg_ids.to(device)

        optimizer.zero_grad()
        output = model(src_ids) # [batch, seq, vec_size]

        # loss = ((1 - torch.cosine_similarity(output, trg, dim=2))).mean()
        loss = F.cross_entropy(output.flatten(0, 1), trg_ids.flatten(), reduction='mean')
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dl)


def evaluate(model, dl, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src_ids, trg_ids) in enumerate(dl):
            src_ids = src_ids.to(device)  # [batch, seq, vec_size]
            trg_ids = trg_ids.to(device)
            output = model(src_ids) # [batch, seq, vec_size]
            loss = F.cross_entropy(output.flatten(0, 1), trg_ids.flatten(), reduction='mean')
            epoch_loss += loss.item()

    return epoch_loss / len(dl)

def accuracy(model, val_dl, device, debug=False):
    model.eval()
    n_correct = 0
    n = 0
    outputs = []
    with torch.no_grad():
        for i, (src_ids, trg_ids) in enumerate(val_dl):
            src_ids = src_ids.to(device)  # [batch, seq, vec_size]
            trg_ids = trg_ids.to(device)
            output = model(src_ids) # [batch, seq, vec_size]
            output_ids = output.argmax(dim=2)
            n += trg_ids.shape[0] * trg_ids.shape[1] # total count
            n_correct += (output_ids == trg_ids).sum()
            outputs.append((src_ids, trg_ids, output_ids))
    acc = n_correct / n
    return acc.item(), outputs


##########
# Setup

# MyModel = LSTMModel
# MyModel = RNNModel
MyModel = RNNStack

model = MyModel(
    input_dim=VEC_SIZE,
    hidden_dim=32,
    output_dim=VEC_SIZE,
    tokenizer=tokenizer,
)
model.to(DEVICE)

no_trains = [model.embeddings]
for no_train in no_trains:
    for p in no_train.parameters():
        p.requires_grad = False

params = [x for x in model.parameters() if x.requires_grad]
print_model_info(model)

optimizer = optim.Adam(params, lr=LR, weight_decay=0)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_dl, optimizer, DEVICE, GRAD_CLIP)
    train_losses.append(train_loss)

    val_loss = evaluate(model, val_dl, DEVICE)
    val_losses.append(val_loss)

    tacc, _ = accuracy(model, train_dl, DEVICE)
    train_accuracies.append(tacc)

    vacc, _ = accuracy(model, val_dl, DEVICE)
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

tacc, outputs = accuracy(model, val_dl, DEVICE, debug=True)

print()
n = 0
for src_idss, trg_idss, output_idss  in reversed(outputs):
    for src_ids, trg_ids, output_ids in zip(src_idss, trg_idss, output_idss):
        s = [tokenizer.decode([x]) for x in src_ids.tolist()]
        t = [tokenizer.decode([x]) for x in trg_ids.tolist()]
        o = [tokenizer.decode([x]) for x in output_ids.tolist()]
        print('__________________________________________________')
        print_grid([['src'] + s,
                    ['out'] + o,
                    ['trg'] + t])
        n += 1
        if n > 5:
            break
    if n > 5:
        break




# Hand check some things

print()

inp = '^ a b c d e e e | . . . . . . .'.split(' ')
trg = '^ . . . . . . . . e e e d c b a'.split(' ')

inp = '^ k l m n o o o | . . . . . . .'.split(' ')
trg = '^ . . . . . . . . o o o n m l k'.split(' ')

inp_ids = torch.tensor(tokenizer.encode(inp, is_pretokenized=True).ids, device=DEVICE).unsqueeze(0)
trg_ids = torch.tensor(tokenizer.encode(trg, is_pretokenized=True).ids, device=DEVICE).unsqueeze(0)
logits, debug = model(inp_ids, debug=True)
out_ids = logits.argmax(dim=2)[0]
out_toks = [tokenizer.decode([x]) for x in out_ids]
print_grid([['inp:'] + inp,
            ['out:'] + out_toks,
            ['trg:'] + trg])


ops = debug['ops_trace']
push, pop, nop = [x.squeeze(1) for x in torch.chunk(ops, chunks=3, dim=-1)]

max_scores = logits.max(dim=2)
print_grid([[f'{x:>.2f}' for x in max_scores.values[0].tolist()],
            max_scores.indices[0].tolist()])

print(f'{F.cross_entropy(logits.flatten(0, 1), trg_ids.flatten())=}')

# Plot logits
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(logits[0].T.detach().cpu().numpy())
plt.title('Logits')
plt.xlabel('Time')
plt.ylabel('Token')

# Plot push, pop, nop operations over time
plt.subplot(1, 2, 2)
time_steps = range(push.shape[1])
plt.plot(time_steps, push[0].detach().cpu().numpy(), label='Push')
plt.plot(time_steps, pop[0].detach().cpu().numpy(), label='Pop')
plt.plot(time_steps, nop[0].detach().cpu().numpy(), label='Nop')
plt.title('Operations over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


# # convert cross-entropy by hand
# num_classes = logits.shape[-1]
# trg_one_hot = F.one_hot(trg_ids, num_classes=num_classes).float()
# loss = -torch.sum(trg_one_hot[0] * F.log_softmax(logits[0], dim=-1), dim=-1).mean()
# print(f'{loss=}')
