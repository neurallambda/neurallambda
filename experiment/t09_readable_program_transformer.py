'''.

'neural programs' that can be read like embeddings

----------
SCAN dataset:
  https://huggingface.co/datasets/scan
  https://github.com/brendenlake/SCAN

'simple' :

'length' : train is shorter sequences than test


'addprim_jump' : training set includes all of the compositional tasks that do
                 not include "jump", as well as just the primitive "jump"
                 command in isolation (over-represented to consist of 10% of
                 the samples). The test set includes all of the compositional
                 commands that use jump.

'addprim_turn_left' : This is similar to the split above, but the "turn left"
                      command is added instead of jump.


Primitive Filler: the training set starts out having no commands containing
                  "around right", but is gradually increased across conditions
                  to include commands containing the expression "Primitive
                  around right" for 0, 1, 2 or 3 different primit ive
                  fillers. The test set is held constant, including only
                  examples with the subcommand "jump around right".
  'filler_num0'
  'filler_num1'
  'filler_num2'
  'filler_num3'

Templates: all sequences except those containing a certain template (a complex
           subcommand, like "jump around right"), to which it must generalize at test
           time. There are 4 different splits corresponding to 4 different held-out
           templates: "jump around right", "Primitive right", "opposite right" and "around
           right"
  'template_around_right'
  'template_jump_around_right'
  'template_opposite_right'
  'template_right'

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# os.environ['HF_HUB_OFFLINE'] = '0'
import datasets
from neurallambda.lab.common import build_tokenizer_dataloader
from neurallambda.lab.datasets import BOS_SYMBOL, PAUSE_SYMBOL

import matplotlib.pyplot as plt

from neurallambda.lab.common import (
    build_tokenizer_dataloader,
    dataloader_info,
    print_grid,
    print_model_info,
    run_epoch,
)
from neurallambda.torch import GumbelSoftmax
from neurallambda.lab.datasets import palindrome, arithmetic_expressions, binary_arithmetic
import math
import random

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


##################################################
# Data


# all_splits = [
#     'simple',
#     'length',
#     'addprim_jump',
#     'addprim_turn_left',
#     'filler_num0',
#     'filler_num1',
#     'filler_num2',
#     'filler_num3',
#     'template_around_right',
#     'template_jump_around_right',
#     'template_opposite_right',
#     'template_right',
# ]
# for s in all_splits:
#     data = datasets.load_dataset(path='scan', name=s)



def prep(data, split):
    commands = [x.split() for x in data[split]['commands']]
    actions = [x.split() for x in data[split]['actions']]
    out = []
    for inputs, outputs in zip(commands, actions):
        inputs, outputs, accuracy_mask = (  # depend on each others' lengths. also add BOS.
            [BOS_SYMBOL] + inputs + [PAUSE_SYMBOL] * len(outputs),
            [PAUSE_SYMBOL] + [PAUSE_SYMBOL] * len(inputs) + outputs,
            [0] * (len(inputs) + 1) + [1] * len(outputs)
        )
        out.append({'inputs': inputs,
                    'outputs': outputs,
                    'accuracy_mask': accuracy_mask})
    return out


try:
    already_loaded
except:
    data = datasets.load_dataset(path='scan', name='addprim_jump')
    ptrain = prep(data, 'train')

    # double the number of jump primitives
    jump_prim = {'inputs': ['^', 'jump', '.'], 'outputs': ['.', '.', 'I_JUMP'], 'accuracy_mask': [0, 0, 1]}
    for _ in range(1467):
        ptrain.append(jump_prim)

    ptest = prep(data, 'test')
    tokenizer, dataloader_fn = build_tokenizer_dataloader([ptrain, ptest])
    already_loaded = True


##################################################
# Classes

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, q, k, v, p=None, mask=None):
        batch_size, device = q.size(0), q.device

        # Linear projections
        q = self.query(q)  # [B, S, D]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        k = self.key(k)  # [B, S, D]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        k = k.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        v = self.value(v)  # [B, S, D]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        v = v.transpose(1, 2)  # [B, NUM_HEADS, S, HEAD_DIM]

        if p is None:
            # Scaled dot-product attention
            scores = torch.einsum('bnse, bnte -> bnst', q, k)  # [B, N_HEADS, S, S]
        else:
            # SDPA with extra Program
            scores = torch.einsum('bnse, bsnef, bntf -> bnst', q, p, k)  # [B, N_HEADS, S, S]

        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))  # [B, NUM_HEADS, S, S]

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)  # [B, N_HEADS, S, S]
            scores += mask  # [B, NUM_HEADS, S, S]

        # scores = self.extra_attn(scores)

        attention_weights = F.softmax(scores, dim=-1)  # [B, NUM_HEADS, S, S]
        # attention_weights = F.dropout(attention_weights, p=dropout_prob, training=self.training)  # [B, NUM_HEADS, S, S]
        attended_values = torch.matmul(attention_weights, v)  # [B, N_HEADS, S, HEAD_DIM]

        # Concatenation and linear transformation
        concatenated = attended_values.transpose(1, 2)  # [B, S, N_HEADS, HEAD_DIM]
        concatenated = concatenated.contiguous().view(batch_size, -1, self.emb_dim)  # [B, S, D]
        output = self.out(concatenated)  # [B, S, D]

        return output

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward=512, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(emb_dim, num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(dim_feedforward, emb_dim, bias=False)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, xs, p=None, xs_mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(xs, xs, xs, p=p, mask=xs_mask)
        xs = self.norm1(xs + attn_output)

        # Feed-forward network with residual connection
        ffnn_output = self.ffnn(xs)
        xs = self.norm2(xs + ffnn_output)

        return xs

def create_identity_tensor(prefix_dims, dim):
    identity_matrix = torch.eye(dim)
    identity_tensor = identity_matrix.repeat(*(prefix_dims + (1, 1)))
    return identity_tensor

class Model(nn.Module):
    def __init__(self, Attention, tokenizer, emb_dim, num_heads, num_layers, dropout=0.1):
        super(Model, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, emb_dim)
        self.pos_encoding = self.positional_encoding(emb_dim)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        ##########
        # Programs: softmax(QPK^T)V
        self.n_progs = 16
        self.n_syntax_layers = 2
        head_dim = emb_dim // num_heads
        self.all_progs = nn.ParameterList([])
        for _ in range(num_layers):

            # self.progs = nn.Parameter(torch.randn((self.n_progs, num_heads, head_dim, head_dim)) / (head_dim**2))
            progs = nn.Parameter(torch.randn((self.n_progs, num_heads, head_dim, head_dim)))
            # self.progs = nn.Parameter(torch.rand((self.n_progs, num_heads, head_dim, head_dim)))
            # self.progs = nn.Parameter(torch.rand((self.n_progs, num_heads, head_dim, head_dim)))

            # with torch.no_grad():
            #     self.progs *= 1e-2
            #     i = create_identity_tensor((self.n_progs, num_heads,), head_dim)
            #     self.progs += i

            syntax_layers = nn.ModuleList([DecoderLayer(emb_dim, num_heads, dropout=dropout) for _ in range(self.n_syntax_layers)])
            norm = nn.LayerNorm(emb_dim)
            prog_select = nn.Sequential(nn.Linear(emb_dim, self.n_progs, bias=False),
                                        nn.Softmax(dim=1)
                                        # GumbelSoftmax(dim=1, hard=True)
                                        # GumbelSoftmax(dim=1, hard=False)
                                        )
            self.all_progs.append(nn.ParameterList([progs, syntax_layers, norm, prog_select]))

    def positional_encoding(self, emb_dim, max_len=5000):
        pos_enc = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz))).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        # return torch.zeros(sz, sz)

    def generate_syntax_mask(self, sz):
        # also don't attend to self. Try to make program selection not dependent on current word.
        mask = (torch.triu(torch.ones(sz, sz, dtype=torch.int))).transpose(0, 1)
        mask -= torch.eye(sz, dtype=torch.int)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask[0][0] = 0  # can pay attn to bos
        return mask

    def forward(self, x_ids):
        B, S, device = x_ids.size(0), x_ids.size(1), x_ids.device
        xs_mask = self.generate_square_subsequent_mask(S).to(device)
        syn_mask = self.generate_syntax_mask(S).to(device)

        # semantics
        xs = self.embedding(x_ids) * math.sqrt(self.emb_dim)
        xs += self.pos_encoding[:S, :].to(xs.device)
        # xs = self.dropout(xs)

        for layer, (progs, syntax_layers, norm, prog_select) in zip(self.layers, self.all_progs):

            # syntax
            prog = self.embedding(x_ids) * math.sqrt(self.emb_dim)
            prog += self.pos_encoding[:S, :].to(prog.device)
            # prog = self.dropout(prog)

            for syntax_layer in syntax_layers:
                prog = syntax_layer(prog, p=None, xs_mask=syn_mask)
            prog = norm(prog)
            prog_select = prog_select(prog)
            prog = torch.einsum('bsp, phde -> bshde ', prog_select, progs)

            # couple prog to xs
            # prog = torch.einsum('bshde, bshe -> bshde', prog, xs.view(B, S, self.num_heads, self.emb_dim//self.num_heads))
            # prog = prog + xs.view(B, S, self.num_heads, self.emb_dim//self.num_heads).unsqueeze(-1).repeat(1, 1, 1, 1, self.emb_dim//self.num_heads)

            xs = layer(xs, p=prog, xs_mask=xs_mask)

            # xs = layer(xs, p=None, xs_mask=xs_mask)

            # if i in {0, 1, 2, 3}:
            #     xs = layer(xs, p=prog, xs_mask=xs_mask)
            # else:
            #     xs = layer(xs, p=None, xs_mask=xs_mask)

        xs = self.norm(xs)
        output = self.fc_out(xs)
        return output


##################################################

EMB_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DEVICE = 'cuda'
LR = 1e-3
NUM_EPOCHS = 30
BATCH_SIZE = 128
GRAD_CLIP = None
LOSS_FN = 'cross_entropy'
# LOSS_FN = 'nllloss'

model = Model(
    Attention=MultiHeadAttention,
    tokenizer=tokenizer,
    emb_dim=EMB_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
)
model.to(DEVICE)

# # skip training embeddings
# print('skipping training embeddings')
# no_trains = [model.embedding]
# for no_train in no_trains:
#     for p in no_train.parameters():
#         p.requires_grad = False

# # only train extra_attn
# for x in model.parameters():
#     x.requires_grad = False
# for x in model.layers:
#     x.self_attn.extra_attn.requires_grad = True

params = [x for x in model.parameters() if x.requires_grad]
print_model_info(model)

optimizer = optim.Adam(params, lr=LR, weight_decay=0)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


def regularization_fn(model, outputs):
    loss = 0

    # # round the extra_attn
    # alpha = 0.000001
    # scale = 2
    # # 10  -> concentrates around 0.1s
    # # 5   -> concentrates around 0.2s
    # # 1   -> concentrates around 1s
    # # 0.1 -> concentrates around 0 only, poor performance
    # loss = 0
    # w = model.progs * scale
    # loss += alpha * torch.sum(torch.abs(w - torch.round(w))) / scale

    return loss


# START_BLOCK_1
for epoch in range(NUM_EPOCHS):
    train_dl, val_dl = dataloader_fn(BATCH_SIZE)

    train_loss, tacc, _ = run_epoch(
        model, train_dl, optimizer, 'train', DEVICE, GRAD_CLIP,
        check_accuracy=True, loss_fn=LOSS_FN, regularization_fn=regularization_fn)
    train_losses.append(train_loss)
    train_accuracies.append(tacc)

    val_loss, vacc, _ = run_epoch(
        model, val_dl, None, 'eval', DEVICE,
        check_accuracy=True, loss_fn=LOSS_FN)
    val_losses.append(val_loss)
    val_accuracies.append(vacc)

    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.9f} | Train Acc: {tacc:.3f} | Val Acc: {vacc:.3f}')

    # Log parameter histograms
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

# END_BLOCK_1

# START_BLOCK_2
print()
for inps, trgs, attn_masks in val_dl:
    for i, (inp, trg, attn_mask) in enumerate(zip(inps, trgs, attn_masks)):
        if i>8:
            break
        out = model(inp.unsqueeze(0).to(DEVICE))
        print()
        print(f'{tokenizer.decode(inp.tolist(), skip_special_tokens=True)}')
        print(f'{tokenizer.decode(trg.tolist(), skip_special_tokens=True)}')
        print(f'{tokenizer.decode(torch.argmax(out[0], dim=1).tolist(), skip_special_tokens=True)}')
    break
# END_BLOCK_2


# plt.hist(model.progs.flatten().tolist(), bins=1000)
# plt.show()
