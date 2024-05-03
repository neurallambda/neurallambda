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
BATCH_SIZE = 32
VEC_SIZE = 256

TRAIN_NUM_SAMPLES = 1000
TRAIN_MAX_SEQUENCE_LENGTH = 20

VAL_NUM_SAMPLES = 100
VAL_MAX_SEQUENCE_LENGTH = 30

BATCH_SIZE = 10

LR = 1e-2

NUM_EPOCHS = 8
GRAD_CLIP = 10

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

# dataloader_info(train_loader, tokenizer)
# dataloader_info(val_loader, tokenizer)



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
    def __init__(self, input_dim, hidden_dim, output_dim, tokenizer):
        super(NeuralstackOnly, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embeddings = nn.Embedding(tokenizer.get_vocab_size(), input_dim)

        ##########
        # Stack
        self.n_stack = 32
        self.initial_sharpen = 5
        self.zero_offset = 1e-6
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
        ss = S.initialize(self.input_dim, self.n_stack, batch_size, self.zero_offset, device, dtype=dtype)

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

            # Update Latch
            latch_state = self.latch(latch_state, enable=inp, data=inp)
            latch_states.append(latch_state)
            relay.append(latch_state)
            relay = relay[1:]
            stack_tops.append(S.read(ss))

        outputs = torch.stack(outputs, dim=1)

        return outputs

    # , {
    #         'latch': {'data':torch.stack(latch_states, dim=1), 'fn': unproject_int},
    #         'pops': {'data':torch.stack(pops, dim=1), 'fn': lambda x: x.tolist()},
    #         'stack': {'data': torch.stack(stack_tops, dim=1), 'fn': unproject_int},
    #     }


##################################################
# Training

def train_epoch(model, dl, optimizer, device, clip):
    '''
    Args:
      warm_up: ignore loss for this many steps
    '''
    model.train()
    epoch_loss = 0

    for i, (src_ids, trg_ids) in enumerate(dl):
        src = model.embeddings(src_ids.to(device))  # [batch, seq, vec_size]
        trg = model.embeddings(trg_ids.to(device))  # [batch, seq, vec_size]

        optimizer.zero_grad()
        # output, latch_states = model(src) # shape=[batch, seq, vec_size]
        output = model(src) # shape=[batch, seq, vec_size]

        # loss = criterion(output, trg) * mask
        loss = ((1 - torch.cosine_similarity(output, trg, dim=2))).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dl)


def evaluate(model, dl, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src_ids, trg_ids) in enumerate(dl):
            src = model.embeddings(src_ids.to(device))
            trg = model.embeddings(trg_ids.to(device))

            # output, latch_states = model(src)
            output = model(src)

            # loss = criterion(output, trg) * mask
            loss = ((1 - torch.cosine_similarity(output, trg, dim=2))).mean()
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
    tokenizer=tokenizer,
)
model.to(DEVICE)


params = [x for name, x in model.named_parameters() if 'Embedding' not in name]
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

    # tacc = accuracy(model, train_dl, DEVICE)
    # train_accuracies.append(tacc)

    # vacc = accuracy(model, val_dl, DEVICE)
    # val_accuracies.append(vacc)

    # print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}')

# Plot training and validation loss
plt.figure()
n = np.arange(len(train_losses))
plt.plot(n, train_losses, label='Train Loss')
plt.plot(n, val_losses, label='Val Loss')
# plt.plot(n, train_accuracies, label='Train Acc')
# plt.plot(n, val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


##########

# tacc = accuracy(model, val_dl, DEVICE, debug=True)
