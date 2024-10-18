'''.

An RNN+Stack model that can learn to reverse strings of symbols, INCLUDING
symbols it was never trained on (!)

TODO:

- why the noise, that loss can't reduce more?

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
import neurallambda.queue as Q
import neurallambda.stack_joulin as SJ
from neurallambda.lab.common import (
    build_tokenizer_dataloader,
    dataloader_info,
    print_grid,
    print_model_info,
    run_epoch,
)
from neurallambda.torch import GumbelSoftmax
from neurallambda.lab.datasets import palindrome, arithmetic_expressions, binary_arithmetic


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


##################################################
# Palindrome

DEVICE = 'cuda'
VEC_SIZE = 256

MIN_LENGTH = 2

TRAIN_NUM_SAMPLES = 4000
TRAIN_MAX_SEQUENCE_LENGTH = 10

VAL_NUM_SAMPLES = 200
VAL_MAX_SEQUENCE_LENGTH = 10

N_STACKS = 1
STACK_DEPTH = 12
QUEUE_DEPTH = 12

HIDDEN_DIM = 32

LR = 5e-3
BATCH_SIZE = 32
BATCH_WARMUP = False

NUM_EPOCHS = 12
GRAD_CLIP = None

# LOSS_FN = 'cosine_distance'
LOSS_FN = 'cross_entropy'

PROBLEM = 'PALINDROME'
# PROBLEM = 'ARITHMETIC_SIMPLE'
# PROBLEM = 'BINARY_ARITHMETIC'


##################################################
# Problem Setup

if PROBLEM == 'PALINDROME':
    ##########
    # Palindrome
    #
    # >>> train_raw[0]
    #   {'inputs': ['^', 'a', 'b', 'c', 'd', '|', '.', '.', '.', '.'],
    #   'outputs': ['.', '.', '.', '.', '.', '.', 'd', 'c', 'b', 'a']}

    # Symbols that go into palindrome
    train_lang = 'a b c d e f g h i j'.split(' ')
    val_lang = 'a b c d e f g h i j'.split(' ')
    # val_lang = 'k l m n o p q r s t'.split(' ')  # NOTE: these tokens are never trained in any example
    train_raw = palindrome(TRAIN_NUM_SAMPLES, MIN_LENGTH, TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang)
    val_raw = palindrome(VAL_NUM_SAMPLES, MIN_LENGTH, VAL_MAX_SEQUENCE_LENGTH, lang=val_lang)

    # Palindrome Training
    viz_inp = '^ a b c d e e e | . . . . . . .'.split(' ')
    viz_trg = '. . . . . . . . . e e e d c b a'.split(' ')

    # # Palindrome Test
    # viz_inp = '^ k l m n o o o | . . . . . . .'.split(' ')
    # viz_trg = '. . . . . . . . . o o o n m l k'.split(' ')

elif PROBLEM == 'ARITHMETIC_SIMPLE':

    numbers = [0, 1, 2, 3, 4]
    modulus = 5

    # operations = ['+']
    # # brackets = []
    # brackets = [('(', ')')]

    operations = ['+', '*']
    brackets = [('(', ')')]

    train_raw = arithmetic_expressions(TRAIN_NUM_SAMPLES, MIN_LENGTH, TRAIN_MAX_SEQUENCE_LENGTH, numbers, modulus, operations, brackets)
    val_raw = arithmetic_expressions(VAL_NUM_SAMPLES, MIN_LENGTH, VAL_MAX_SEQUENCE_LENGTH, numbers, modulus, operations, brackets)

    viz_inp = '^ 2 + 4 .'.split(' ')
    viz_trg = '. . . . 1'.split(' ')


elif PROBLEM == 'BINARY_ARITHMETIC':

    op = '+'
    # len=7   2**6  64
    # len=8   2**7  128
    # len=9   2**8  256
    # len=10  2**9  512
    # len=11  2**10 1024
    train_raw = binary_arithmetic(TRAIN_NUM_SAMPLES, 0, 2**4, op)
    val_raw = binary_arithmetic(VAL_NUM_SAMPLES, 0, 2**5, op)

    viz_inp = '^ 1 1 1 1 0 + 1 0 1 0 1 . . . . . .'.split(' ')
    viz_trg = '. . . . . . . . . . . . 1 1 0 0 1 1'.split(' ')


##########
# Prep

train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))
val_raw = sorted(val_raw, key=lambda x: len(x['inputs']))
tokenizer, dataloader_fn = build_tokenizer_dataloader(
    [train_raw, val_raw],
    data_keys=['inputs', 'outputs'])

# dataloader_info(train_dl, tokenizer)


##############################
# RNNStack

class RNNStack(nn.Module):
    def __init__(self, emb_dim, hidden_dim, tokenizer):
        super(RNNStack, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_stacks = N_STACKS
        self.stack_depth = STACK_DEPTH

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        self.rnn_i = nn.Linear(emb_dim + self.n_stacks * emb_dim, hidden_dim, bias=True)
        self.rnn_h = nn.Linear(hidden_dim, hidden_dim, bias=True)

        ##########
        # Outputs

        self.sem = nn.Sequential(
            # nn.Linear(hidden_dim * 2, self.emb_dim, bias=True),

            nn.Linear(hidden_dim * 2, self.vocab_size, bias=True),

            # nn.Linear(hidden_dim * 2, 512, bias=False),
            # nn.ReLU(),
            # nn.Linear(512, self.emb_dim, bias=False),

            # nn.Linear(hidden_dim * 2, 256, bias=True),
            # nn.ReLU(),
            # nn.Linear(256, self.vocab_size, bias=False),

        )

        # self.unembed = nn.Linear(self.emb_dim, self.vocab_size, bias=False)
        # self.unembed.weight = self.embeddings.weight


        # self.syn = nn.Sequential(
        #     nn.Linear(emb_dim, self.vocab_size, bias=False),
        #     # nn.Softmax(dim=1)
        #     # GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        # )

        # self.choose = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, 2, bias=False),
        #     # nn.Softmax(dim=1)
        #     # GumbelSoftmax(dim=1, temperature=1.0, hard=True)
        #     GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        # )

        ##########
        # Stack

        self.stack_params = nn.ParameterList()
        for _ in range(self.n_stacks):
            push_val = nn.Sequential(
                nn.Linear(hidden_dim * 2, emb_dim, bias=True)

                # nn.Linear(hidden_dim * 2, 16, bias=False),  # [hidden, push+pop+nop]
                # nn.ReLU(),
                # nn.Linear(16, emb_dim, bias=False),  # [hidden, push+pop+nop]

            )
            ops = nn.Sequential(

                nn.Linear(hidden_dim * 2, 3, bias=True),  # [hidden, push+pop+nop]

                # nn.Linear(hidden_dim * 2, 16, bias=False),  # [hidden, push+pop+nop]
                # nn.ReLU(),
                # nn.Linear(16, 3, bias=False),  # [hidden, push+pop+nop]

                nn.Softmax(dim=1)
                # GumbelSoftmax(dim=1, temperature=1.0, hard=True)
                # GumbelSoftmax(dim=1, temperature=1.0, hard=False)
            )
            self.stack_params.append(nn.ParameterList([push_val, ops]))

        # A non-linear transformation baked in
        self.weird = torch.randn(hidden_dim * 2, device='cuda')

    def forward(self, x_ids, debug=False):
        xs = self.embeddings(x_ids)
        # xs = F.normalize(xs, dim=1)

        batch_size, device, dtype = xs.size(0), xs.device, xs.dtype
        # stack states
        sss = []
        for _ in range(self.n_stacks):
            ss = SJ.initialize(self.emb_dim, self.stack_depth, batch_size, device, dtype=dtype)
            sss.append(ss)
        h = torch.zeros(xs.size(0), self.hidden_dim).to(xs.device)
        outputs = []
        if debug:
            ops_traces = []
        for i in range(xs.size(1)):
            x = xs[:, i]
            peeks = []
            for ss in sss:
                peeks.append(SJ.read(ss))

            ri = self.rnn_i(torch.cat([x] + peeks, dim=1))
            rh = self.rnn_h(h)
            # h = (ri + rh).relu()  # only relu works
            h1 = (ri+rh).relu()
            # h2 = (ri+rh).relu()
            h2 = (ri-rh).relu()  # for some reason this works great
            h = h1
            hh = torch.cat([h1, h2], dim=1)

            # Update Stacks
            nsss = []
            ops_trace = []
            for (ss, (push_val, ops)) in zip(sss, self.stack_params):
                pv = push_val(hh)
                ops = ops(hh)
                if debug:
                    ops_trace.append(ops)
                push, pop, nop = [o.squeeze(1) for o in torch.chunk(ops, chunks=3, dim=-1)]
                nss, pop_val = SJ.push_pop_nop(ss, push, pop, nop, pv)
                nsss.append(nss)
            if debug:
                ops_traces.append(torch.stack(ops_trace, dim=1))

            output = self.sem(hh)
            # output = self.sem(torch.roll(hh, 1, dims=-1))
            # output = self.sem(torch.einsum('h, bh -> bh', self.weird, hh))

            # output = torch.einsum('ve, be -> bv',
            #                       # F.normalize(self.embeddings.weight, dim=1),
            #                       self.embeddings.weight,
            #                       # F.normalize(self.sem(hh), dim=1)
            #                       self.sem(hh)
            #                       )
            # output = torch.log(output.abs())



            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        if debug:
            ops_traces = torch.stack(ops_traces, dim=1)  # [B, T, Stack, 3], i think
            debug_out = {'ops_traces': ops_traces}
            return outputs, debug_out
        else:
            return outputs


##################################################
# Training

##########
# Setup

Model = RNNStack

model = Model(
    emb_dim=VEC_SIZE,
    hidden_dim=HIDDEN_DIM,
    tokenizer=tokenizer,
)
model.to(DEVICE)

# skip training embeddings
print('skipping training embeddings')
no_trains = [model.embeddings]
for no_train in no_trains:
    for p in no_train.parameters():
        p.requires_grad = False

params = [x for x in model.parameters() if x.requires_grad]
print_model_info(model)

warmup_optimizer = optim.Adam(params, lr=LR*0.1, weight_decay=0)
main_optimizer = optim.Adam(params, lr=LR, weight_decay=0)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


# START_BLOCK_1
for epoch in range(NUM_EPOCHS):
    if BATCH_WARMUP and epoch == 0:
        print('Running at batch_size==1 for first epoch')
        train_dl, val_dl = dataloader_fn(1)
        optimizer = warmup_optimizer
    else:
        train_dl, val_dl = dataloader_fn(BATCH_SIZE)
        optimizer = main_optimizer

    train_loss, tacc, _ = run_epoch(
        model, train_dl, optimizer, 'train', DEVICE, GRAD_CLIP,
        check_accuracy=True, loss_fn=LOSS_FN)
    train_losses.append(train_loss)
    train_accuracies.append(tacc)

    val_loss, vacc, _ = run_epoch(
        model, val_dl, None, 'eval', DEVICE,
        check_accuracy=True, loss_fn=LOSS_FN)
    val_losses.append(val_loss)
    val_accuracies.append(vacc)

    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')
# END_BLOCK_1


##################################################
# RESULTS & VISUALIZATIONS

# START_BLOCK_2

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
                                 check_accuracy=True, loss_fn=LOSS_FN)

print()
n = 0
for src_idss, trg_idss, acc_masks, output_idss in reversed(outputs):
    for src_ids, trg_ids, acc_mask, output_ids in zip(src_idss, trg_idss, acc_masks, output_idss):
        s = [tokenizer.decode([x]) for x in src_ids.tolist()]
        t = [tokenizer.decode([x]) for x in trg_ids.tolist()]
        o = [tokenizer.decode([x]) for x in output_ids.tolist()]
        print('__________________________________________________')
        print_grid([['src'] + s,
                    ['out'] + o,
                    ['trg'] + t,
                    ['acc'] + acc_mask.to(dtype=torch.int32).tolist()])
        n += 1
        if n > 5:
            break
    if n > 5:
        break



model.eval()
inp_ids = torch.tensor(tokenizer.encode(viz_inp, is_pretokenized=True).ids, device=DEVICE).unsqueeze(0)
trg_ids = torch.tensor(tokenizer.encode(viz_trg, is_pretokenized=True).ids, device=DEVICE).unsqueeze(0)
out, debug = model(inp_ids, debug=True)
if LOSS_FN == 'cosine_distance':
    scores = torch.einsum('vd, btd -> btv',
                          F.normalize(model.embeddings.weight, dim=1),  # [VOCAB, DIM]
                          F.normalize(out, dim=2))
    out_ids = scores.argmax(dim=2)[0]  # [BATCH, TIME, DIM]
    max_scores = scores.max(dim=2)
else:
    scores = torch.softmax(out, dim=-1)
    out_ids = scores.argmax(dim=2)[0]
    max_scores = scores.max(dim=2)

out_toks = [tokenizer.decode([x]) for x in out_ids]
print_grid([['inp:'] + viz_inp,
            ['out:'] + out_toks,
            ['trg:'] + viz_trg])


ops = debug['ops_traces']
ops = ops[:, :, 0, :]  # select stack 0
push, pop, nop = [x.squeeze(1) for x in torch.chunk(ops, chunks=3, dim=-1)]

print_grid([[f'{x:>.2f}' for x in max_scores.values[0].tolist()],
            max_scores.indices[0].tolist()])

# print(f'{F.cosine_distance(out.flatten(0, 1), trg_ids.flatten())=}')

# Plot out
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(scores[0].T.detach().cpu().numpy())
plt.title('Logits')
plt.xlabel('Time')
plt.xticks(range(inp_ids.shape[1]))
plt.ylabel('Token')

# Plot push, pop, nop operations over time
plt.subplot(1, 2, 2)
time_steps = range(push.shape[1])
plt.plot(time_steps, push[0].detach().cpu().numpy(), label='Push')
plt.plot(time_steps, pop[0].detach().cpu().numpy(), label='Pop')
plt.plot(time_steps, nop[0].detach().cpu().numpy(), label='Nop')
plt.title('Operations over Time')
plt.xlabel('Time')
plt.xticks(range(inp_ids.shape[1]))
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# END_BLOCK_2
