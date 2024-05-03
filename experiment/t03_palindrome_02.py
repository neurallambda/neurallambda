'''

Moving toward a common interface for building and testing models

'''


import tokenizers
from tokenizers import Tokenizer, AddedToken
from typing import List, Any, Dict
import datasets
import torch
import random
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from functools import partial
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
    for name, param in model.named_parameters():
        info.append((name, param.numel(), f'{list(param.shape)}', f'grad:{param.requires_grad}'))
    print_grid(info)

def iterator(all_datasets: List[Dataset], keys: List[str], special_tokens):
    ''' Iterate over a list of datasets and build a vocab for them. '''
    batch_size = 1000
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
    # a = tokenizer.encode(['1', '2', '3'], is_pretokenized=True)
    # print_grid([['tokens:'] + a.tokens,
    #             ['ids:'] + a.ids])
    dataloaders = [
        DataLoader(
            x,
            batch_size=BATCH_SIZE,
            collate_fn=partial(collate_fn, PAD_TOKEN)
        ) for x in hf_datasets]

    return tokenizer, dataloaders



##################################################
# Palindrome


torch.manual_seed(11)
DEVICE = 'cuda'
VEC_SIZE = 256

TRAIN_NUM_SAMPLES = 2000
TRAIN_MAX_SEQUENCE_LENGTH = 10

VAL_NUM_SAMPLES = 200
VAL_MAX_SEQUENCE_LENGTH = 20

N_STACK = 10

LR = 1e-2
BATCH_SIZE = 32

NUM_EPOCHS = 8
GRAD_CLIP = None

# Symbols that go into palindrome
train_lang = 'a b c d e f g h i j'.split(' ')
val_lang   = 'a b c d e f g h i j'.split(' ')
# val_lang   = 'k l m n o p q r s t'.split(' ')


##########
# DATA

REFLECT_SYMBOL = '|'
PAUSE_SYMBOL = '.'

def palindrome(num_samples, max_length, lang) -> Dict[str, List[str]]:
    data = []
    for _ in range(num_samples):
        length = random.randint(5, max_length)
        hlength = length // 2
        seed = [random.choice(lang) for _ in range(hlength)]
        # add pauses to inputs and outputs
        inputs  = seed + [REFLECT_SYMBOL] + [PAUSE_SYMBOL] * hlength
        outputs = [PAUSE_SYMBOL] * (hlength + 1) + seed[::-1]
        # convert all symbols to str
        inputs  = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'seed': seed,
            'inputs': inputs,
            'outputs': outputs,
        })
    return data

train_raw = palindrome(TRAIN_NUM_SAMPLES, TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang)
train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))

val_raw = palindrome(VAL_NUM_SAMPLES, VAL_MAX_SEQUENCE_LENGTH, lang=val_lang)
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

        self.lc1 = nn.RNNCell(input_dim, hidden_dim)
        self.lc2 = nn.RNNCell(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, self.vocab_size, bias=False),
            nn.Softmax(dim=1)
        )

        ##########
        # Stack
        self.n_stack = N_STACK
        self.initial_sharpen = 10
        self.zero_offset = 0
        self.stack_vec_size = input_dim
        self.sharp = nn.Parameter(torch.tensor([5.0]))

        self.ops = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3, bias=False), # rnn hidden -> push,pop,nop
            nn.Softmax(dim=1)
        )


    def forward(self, x_ids):
        x = model.embeddings(x_ids)
        batch_size, device, dtype = x.size(0), x.device, x.dtype
        # init
        ss = S.initialize(self.input_dim, self.n_stack, batch_size, self.zero_offset, device, dtype=dtype)
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        outputs = []
        for i in range(x.size(1)):
            h1 = self.lc1(x[:, i], h1)

            ops = self.ops(h1)
            push, pop, nop = [x.squeeze(1) for x in torch.chunk(ops, chunks=3, dim=-1)]
            ss, pop_val = S.push_pop_nop(ss, self.sharp, push, pop, nop, x[:, i])

            output = self.fc(torch.cat([h1, pop_val], dim=1))
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs



##########
# NeuralStack Only

class NeuralstackOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, tokenizer):
        super(NeuralstackOnly, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, input_dim)

        ##########
        # Stack
        self.n_stack = N_STACK
        self.initial_sharpen = 5
        self.zero_offset = 0
        self.stack_vec_size = input_dim
        self.sharpen_pointer = nn.Parameter(torch.tensor([5.0]))

        ##########
        # Latch
        self.latch = L.DataLatch(input_dim, init_scale=1e-2, dropout=None)

        ##########
        # NNs

        self.should_pop = nn.Parameter(torch.randn(input_dim))
        self.null_symbol = nn.Parameter(torch.randn(output_dim))

        self.proj_out = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, self.vocab_size, bias=False)
        )


    def forward(self, x_ids):
        # Stack Init

        x = model.embeddings(x_ids)

        device = x.device
        batch_size = x.shape[0]
        vec_size = x.shape[2]
        dtype = x.dtype

        # self.stack.init(batch_size, zero_offset, device)
        ss = S.initialize(self.input_dim, self.n_stack, batch_size, self.zero_offset, device, dtype=dtype)

        # Latch Init
        latch_state = torch.zeros((batch_size, self.input_dim), device=device) + self.zero_offset
        # latch_state = torch.randn((batch_size, self.input_dim), device=device)

        outputs = []

        # Debugging
        latch_states = []
        pops = []
        stack_tops = []

        # relay = [latch_state, ]

        # Loop over sequence
        for i in range(x.size(1)):
            inp = x[:, i]

            # lsr = relay[0]
            lsr = latch_state

            # Decide what to do
            pop = cosine_similarity(self.should_pop.unsqueeze(0), lsr)
            # pop = elu(pop)
            # pop = gelu(pop)
            # pop = leaky_relu(pop)
            pops.append(pop)

            push    = 1 - pop
            null_op = torch.zeros_like(pop)

            # Update Stack
            ss, pop_val = S.push_pop_nop(ss, self.sharpen_pointer, push, pop, null_op, inp)
            s = S.read(ss)

            out = self.proj_out(
                einsum('b, bv -> bv', pop, s) +
                einsum('b,  v -> bv', 1 - pop, self.null_symbol)
            )
            out = F.softmax(out, dim=1)
            outputs.append(out)

            # Update Latch
            latch_state = self.latch(latch_state, enable=inp, data=inp)
            latch_states.append(latch_state)
            # relay.append(latch_state)
            # relay = relay[1:]
            stack_tops.append(S.read(ss))

        outputs = torch.stack(outputs, dim=1)
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
# MyModel = NeuralstackOnly

model = MyModel(
    input_dim=VEC_SIZE,
    hidden_dim=8,
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

optimizer = optim.Adam(params, lr=LR)

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
