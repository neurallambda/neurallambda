'''

Compare architectures on this task using RayTune

'''


from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import neurallambda.stack as S
from neurallambda.lab.common import (
    build_tokenizer_dataloader,
    dataloader_info,
    print_grid,
    print_model_info,
    run_epoch,
)
from neurallambda.torch import GumbelSoftmax

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


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
val_lang = 'k l m n o p q r s t'.split(' ')  # NOTE: these tokens are never trained in any example


##########
# DATA

BOS_SYMBOL = '^'
REFLECT_SYMBOL = '|'
PAUSE_SYMBOL = '.'


def palindrome(num_samples,
               min_length,
               max_length,
               lang) -> Dict[str, List[str]]:
    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        hlength = length // 2
        seed = [random.choice(lang) for _ in range(hlength)]
        # add pauses to inputs and outputs
        inputs = [BOS_SYMBOL] + seed + [REFLECT_SYMBOL] + [PAUSE_SYMBOL] * hlength
        outputs = [PAUSE_SYMBOL] * (hlength + 2) + seed[::-1]
        # convert all symbols to str
        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'inputs': inputs,
            'outputs': outputs,
        })
    return data


train_raw = palindrome(TRAIN_NUM_SAMPLES, MIN_LENGTH,
                       TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang)
train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))
# >>> train_raw[0]
#   {'inputs': ['^', 'a', 'b', 'c', 'd', '|', '.', '.', '.', '.'],
#   'outputs': ['.', '.', '.', '.', '.', '.', 'd', 'c', 'b', 'a']}

val_raw = palindrome(VAL_NUM_SAMPLES, MIN_LENGTH,
                     VAL_MAX_SEQUENCE_LENGTH, lang=val_lang)
val_raw = sorted(val_raw, key=lambda x: len(x['inputs']))

tokenizer, (train_dl, val_dl) = build_tokenizer_dataloader(
    [train_raw, val_raw],
    data_keys=['inputs', 'outputs'],
    batch_size=BATCH_SIZE)

dataloader_info(train_dl, tokenizer)


##############################
# Transformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return x + self.pos_embedding(positions)

class AlibiPositionalBias(nn.Module):
    def __init__(self, num_heads, max_len=5000):
        super(AlibiPositionalBias, self).__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        slopes = torch.Tensor(self._get_slopes(num_heads))
        bias = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_len).unsqueeze(0).unsqueeze(0)
        self.register_buffer('bias', bias)

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.bias[:, :, :seq_len]

# Add the relative positional encoding from Transformer-XL
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.embedding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.embedding[:seq_len]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, tokenizer, pos_encoding='sine'):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, input_dim)

        if pos_encoding == 'sine':
            self.pos_encoder = PositionalEncoding(input_dim)
        elif pos_encoding == 'learned':
            self.pos_encoder = PositionEmbedding(input_dim)
        elif pos_encoding == 'alibi':
            self.pos_encoder = AlibiPositionalBias(num_heads)
        elif pos_encoding == 'transformerxl':
            self.pos_encoder = RelativePositionalEncoding(input_dim)
        else:
            raise ValueError(f"Unsupported position encoding: {pos_encoding}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x_ids):
        x = self.embeddings(x_ids)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        outputs = F.softmax(x, dim=-1)
        return outputs


##############################
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
        B = x_ids.size(0)
        x = model.embeddings(x_ids)
        # init
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        c1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []
        for i in range(x.size(1)):
            h1, c1 = self.lc1(x[:, i, :], (h1, c1))
            h1s.append(h1)
        h1s = torch.stack(h1s, dim=1)  # [batch, seq, dim]
        h2 = torch.zeros(B, self.hidden_dim).to(x.device)  # [batch, dim]
        c2 = torch.zeros(B, self.hidden_dim).to(x.device)  # [batch, dim]
        outputs = []
        for i in range(x.size(1)):
            h2, c2 = self.lc2(h1s[:, i], (h2, c2))
            output = self.fc(h2)
            output = F.softmax(output, dim=1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


##############################
# RNN

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
        B = x_ids.size(0)
        x = model.embeddings(x_ids)
        # init
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []
        for i in range(x.size(1)):
            h1 = self.lc1(x[:, i, :], h1)
            h1s.append(h1)
        h1s = torch.stack(h1s, dim=1)  # [batch, seq, dim]
        h2 = torch.zeros(B, self.hidden_dim).to(x.device)  # [batch, dim]
        outputs = []
        for i in range(x.size(1)):
            h2 = self.lc2(h1s[:, i], h2)
            output = self.fc(h2)
            output = F.softmax(output, dim=1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


##############################
# RNNStack

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
            # nn.Softmax(dim=1)
            # GumbelSoftmax(dim=1, temperature=1.0, hard=True)
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
            nn.Linear(hidden_dim * 2, 3, bias=False),  # [hidden, push+pop+nop]
            # nn.Softmax(dim=1)
            # GumbelSoftmax(dim=1, temperature=1.0, hard=True)
            GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        )

    def forward(self, x_ids, debug=False):
        x = model.embeddings(x_ids)
        batch_size, device, dtype = x.size(0), x.device, x.dtype
        # init
        ss = S.initialize(self.input_dim, self.n_stack, batch_size,
                          self.zero_offset + 1e-3, device, dtype=dtype)
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
            push, pop, nop = [o.squeeze(1)
                              for o in torch.chunk(ops, chunks=3, dim=-1)]
            ss, pop_val = S.push_pop_nop(ss, self.sharp,
                                         push, pop, nop, x[:, i])

            # Outputs
            semantic = self.fc(hmod)
            # OPTIM: will this be memory hog?
            syntactic = torch.cosine_similarity(
                pop_val.unsqueeze(1),  # [B, 1, D]
                self.embeddings.weight.unsqueeze(0),  # [1, V, D]
                dim=2)  # [B, V]
            choice = self.choose(hmod)
            output = torch.einsum('bc, bcv -> bv',
                                  choice,
                                  torch.stack([semantic, syntactic], dim=1))
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
    train_loss, tacc, _ = run_epoch(
        model, train_dl, optimizer, 'train', DEVICE, GRAD_CLIP,
        check_accuracy=True)
    train_losses.append(train_loss)

    val_loss, vacc, _ = run_epoch(
        model, val_dl, None, 'eval', DEVICE,
        check_accuracy=True)
    val_losses.append(val_loss)

    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')


##################################################
# RESULTS & VISUALIZATIONS

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

vloss, vacc, outputs = run_epoch(model, val_dl, None, 'eval', DEVICE, GRAD_CLIP,
                                 check_accuracy=True)

print()
n = 0
for src_idss, trg_idss, output_idss in reversed(outputs):
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
trg = '. . . . . . . . . e e e d c b a'.split(' ')

inp = '^ k l m n o o o | . . . . . . .'.split(' ')
trg = '. . . . . . . . . o o o n m l k'.split(' ')

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
