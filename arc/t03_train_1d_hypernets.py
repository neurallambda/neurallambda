'''

Train ARC-like 1D using hypernets

TRAIN TINY NETS FOR USE IN DOWNSTREAM TASKS


'''

import os
import warnings
import random
from typing import Callable, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, LayerNorm
import torch.optim as optim
from einops.layers.torch import Rearrange

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as Q
import tokenizers
from tokenizers import Tokenizer

from neurallambda.lab.common import print_model_info
import t02_train_1d_serial_attn as Attention
import importlib
try:
    importlib.reload(arc_like)
except NameError:
    print('RELOADING DATA MODULE')
    import t01_data_arc_like as arc_like

SEED = 152
torch.manual_seed(152)
random.seed(SEED)

NUM_SAMPLES = 4000
SEQ_LEN = 24
N_SHOTS = 2

BATCH_SIZE = 1024

NUM_EPOCHS = 200
LR = 2e-3
WD = 0
GRAD_CLIP = None
LOSS_FN = 'cross_entropy'
DEVICE = 'cuda'


##################################################
# Tokenizer

EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
tokens = '= -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'.split(' ')

special_tokens = [UNK_TOKEN, EOS_TOKEN, PAD_TOKEN]
tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab={},
                                                  unk_token=UNK_TOKEN))
tokenizer.train_from_iterator(tokens + special_tokens)
tokenizer.add_special_tokens(special_tokens)

def token_to_id(tok):
    assert isinstance(tok, str)
    i = tokenizer.token_to_id(tok)
    if i is None:
        raise ValueError(f'token "{tok}" not found in tokenizer vocabulary')
    return i

EOS_TOKEN_ID = token_to_id(EOS_TOKEN)
PAD_TOKEN_ID = token_to_id(PAD_TOKEN)
UNK_TOKEN_ID = token_to_id(UNK_TOKEN)
EQ_TOKEN_ID = token_to_id('=')

tokenizer.pad_token_id = PAD_TOKEN_ID

# Datasets
a = arc_like

def puzzles_fn():
    return {
        # 'some translate(3)': a.compose([a.gen_some_blocks(a.colors, SEQ_LEN), a.translate(3)]),

        # 'some denoise (0.3)': a.compose([a.gen_some_blocks(a.colors, SEQ_LEN), a.swap, a.add_bg_noise(0.3, a.colors), a.swap]),
        # 'one denoise (0.3)': a.compose([a.gen_one_block(a.colors, SEQ_LEN), a.swap, a.add_bg_noise(0.3, a.colors), a.swap]),
        # 'three denoise (0.3)': a.compose([a.gen_three_blocks(a.colors, SEQ_LEN), a.swap, a.add_bg_noise(0.3, a.colors), a.swap]),

        # 'endpoints': a.compose([a.gen_some_blocks(a.colors, SEQ_LEN), a.endpoints]),
        'infill': a.compose([a.gen_some_blocks(a.colors, SEQ_LEN), a.endpoints, a.swap]),

        # 'magnets': a.compose([a.gen_n_blocks(a.colors, 2, SEQ_LEN), a.magnets()]),

        # 'sort_pixels': a.compose([a.gen_some_pixels(a.colors[:3], p=0.1, seq_length=SEQ_LEN), a.sort_pixels()]),

        # 'repaint-from-max-block': a.compose([a.gen_three_blocks(a.colors, SEQ_LEN), a.repaint_max_block]),
        # 'move_to_pivot': a.compose([a.gen_one_block(list(set(a.colors) - {5}), SEQ_LEN), a.add_pivot, a.move_to_pivot]),
        # 'extend_to_pivot': a.compose([a.gen_one_block(list(set(a.colors) - {5}), SEQ_LEN), a.add_pivot, a.extend_to_pivot]),
}

train_puzzles = puzzles_fn()
val_puzzles = puzzles_fn()

train_dl, val_dl = arc_like.build_dataloaders(tokenizer,
                                              batch_size=BATCH_SIZE,
                                              seq_len=SEQ_LEN,
                                              num_samples=NUM_SAMPLES,
                                              device=DEVICE,
                                              train_puzzles=train_puzzles,
                                              val_puzzles=val_puzzles,)

# Visualize Data
if False:
    from arc_like.puzzles import Sequence
    from torch.utils.data import TensorDataset

    datasets = {}
    num_samples = 10
    grid_width = 2
    grid_height = 2
    for (name, transformer) in train_puzzles.items():
        all_inputs, all_outputs = [], []
        for _ in range(num_samples):
            seq = Sequence([], [], None)
            seq = transformer(seq)
            all_inputs.append(seq.inputs)
            all_outputs.append(seq.outputs)
            inputs_tensor, outputs_tensor = torch.tensor(all_inputs), torch.tensor(all_outputs)
            datasets[name] = TensorDataset(inputs_tensor, outputs_tensor)

    arc_like.visualize_datasets(datasets, grid_width=grid_width, grid_height=grid_height, num_samples=num_samples)
    BRK


##################################################
#

def run_epoch(model,
              dl, optimizer, mode, device,
              clip=None,
              check_accuracy=False,
              loss_fn='cross_entropy'):
    ''' Run an epoch over a DataLoader, and optionally perform greedy sampling
    to check accuracy.

    Args:
      if loss_fn == 'cosine_distance', model must return probabilities (after sigmoid activation)
      if loss_fn == 'cross_entropy', model must return unnormalized logits (ie final output comes from Linear layer)
      if loss_fn == 'nllloss', model must return logprobs, ie `F.log_softmax(..., dim=-1)`
    '''
    assert mode in ['eval', 'train'], "mode must be either 'eval' or 'train'"
    assert loss_fn in ['cosine_distance', 'nllloss', 'cross_entropy'], "loss_fn must be 'cosine_distance', 'nllloss', or 'cross_entropy'"

    if mode == 'train':
        model.train()
    else:
        model.eval()

    epoch_loss = 0

    if check_accuracy:
        outputs = []
        total_correct = 0
        total_samples = 0

    with torch.set_grad_enabled(mode == 'train'):
        for i, batch in enumerate(dl):
            input_ids = batch['inputs']  # [B, S]
            attention_mask = batch['inputs_mask']  # [B, S]
            target_ids = batch['outputs']  # [B, S]
            B, S = input_ids.shape
            outs = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         )  # [B, S, DIM]

            if loss_fn == 'cosine_distance':

                # trg_tags = model.embeddings(trg_tags_ids)
                # trg_col1 = model.embeddings(trg_col1_ids)
                # trg_col2 = model.embeddings(trg_col2_ids)

                # loss = 0
                # for t, o in zip([trg_tags, trg_col1, trg_col2],
                #                 [out_tags, out_col1, out_col2]):
                #     # t: [B, S, D]
                #     # o: [B, S, D]
                #     assert o.shape[0] == t.shape[0]
                #     assert o.shape[1] == t.shape[1]
                #     assert o.shape[2] == t.shape[2]
                #     loss += (1 - torch.cosine_similarity(o, t, dim=2).mean())
                raise Exception('cosine_distance not implemented')

            elif loss_fn == 'cross_entropy':
                loss = 0

                # target_ids: [B, S]
                # outs      : [B, S, Vocab]
                assert outs.shape[0] == target_ids.shape[0], f'{outs.shape[0]=} == {target_ids.shape[0]=}'
                assert outs.shape[1] == target_ids.shape[1], f'{outs.shape[1]=} == {target_ids.shape[1]=}'
                assert outs.shape[2] == model.embedding.weight.shape[0], f'{outs.shape[2]=} == {model.embedding.weight.shape[0]=}'
                loss += F.cross_entropy(outs.flatten(0, 1),
                                        target_ids.flatten(),
                                        reduction='mean')


            elif loss_fn == 'nllloss':
                # trg_ids = torch.cat([trg_tags_ids, trg_col1_ids, trg_col2_ids], dim=1)
                # # log_probs = F.log_softmax(output, dim=-1)
                # loss = F.nll_loss(output.flatten(0, 1),
                #                   trg_ids.flatten(),
                #                   reduction='mean')
                pass

            # if regularization_fn is not None:
            #     loss += regularization_fn(model, output)

            epoch_loss += loss.item()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            if check_accuracy:
                with torch.no_grad():
                    if loss_fn == 'cosine_distance':
                        # # find the embedding closest to the output, consider
                        # # that the output_id
                        # out_ids = torch.cosine_similarity(model.embeddings.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, VOCAB, EMB_DIM]
                        #                                   outs.logits.unsqueeze(2),  # [BATCH, TIME, 1, EMB_DIM]
                        #                                   dim=3).argmax(dim=2)
                        pass
                    else:
                        out_ids = outs.argmax(dim=2)

                    correct_outs = (out_ids == target_ids).sum().item()

                    total_correct += correct_outs
                    total_samples += out_ids.numel()
                    outputs.append((out_ids))

    if check_accuracy:
        accuracy = total_correct / total_samples
        # scores: (precision, recall, f1_score, support)
        # f1_tags = precision_recall_fscore_support(y_true_tags, y_pred_tags, average='weighted', zero_division=torch.nan)
        # f1_col1 = precision_recall_fscore_support(y_true_col1, y_pred_col1, average='weighted', zero_division=torch.nan)
        # f1_col2 = precision_recall_fscore_support(y_true_col2, y_pred_col2, average='weighted', zero_division=torch.nan)

        return epoch_loss / len(dl), {
            # 'f1_tags': f1_tags,
            # 'f1_col1': f1_col1,
            # 'f1_col2': f1_col2,
            'accuracy': accuracy,
        }, outputs

    else:
        return epoch_loss / len(dl)


##################################################
# Models

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # To add dropout, consider `Recurrent Neural Network Regularization`
        # self.dropout = nn.Dropout(dropout)

        # Create a ModuleList of LSTMCells
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size=embedding_dim if i == 0 else hidden_dim, hidden_size=hidden_dim)
            for i in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        embedded = self.embedding(input_ids)

        # Initialize hidden state and cell state
        h = [torch.zeros(batch_size, self.hidden_dim).to(input_ids.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(input_ids.device) for _ in range(self.num_layers)]

        outputs = []

        for t in range(seq_length):
            x = embedded[:, t, :]

            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer], c[layer] = self.lstm_cells[layer](x, (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.lstm_cells[layer](h[layer - 1], (h[layer], c[layer]))

            outputs.append(h[-1])
        output = torch.stack(outputs, dim=1)
        logits = self.fc(output)
        return logits


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits


##############################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SINTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, dim_feedforward, num_layers=3, dropout=0.1):
        super(SINTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim,
                                    nhead=nhead,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout,
                                    batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        embedded = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        output = self.pos_encoder(embedded)

        # make sure attention_mask is all in {0, 1}
        # assert torch.allclose(torch.logical_or(attention_mask==0, attention_mask==1), torch.ones_like(attention_mask, dtype=torch.bool))

        # src_key_padding_mask = attention_mask == 0  # False means not padding, True means is padding ignore
        src_key_padding_mask = None

        # src_key_padding_mask = attention_mask.float().masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        # breakpoint()
        # src_key_padding_mask = attention_mask.float()

        for encoder in self.transformer_encoders:
            output = encoder(output, src_key_padding_mask=src_key_padding_mask)
        # output = self.norm(output)
        logits = self.fc(output)
        return logits



##############################

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = self._attention(q, k, v, attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output

    def _attention(self, q, k, v, attention_mask=None):
        d_k = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, v)
        return output

class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, attention_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ROPETransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, dim_feedforward, num_layers=3, dropout=0.1):
        super(ROPETransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(d_model=embedding_dim, nhead=nhead,
                                            dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        embedded = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        output = embedded
        for encoder in self.transformer_encoders:
            output = encoder(output, src_key_padding_mask=attention_mask)
        output = self.norm(output)
        logits = self.fc(output)
        return logits


##############################

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, seq_len, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(seq_len, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, seq_len, embedding_dim, vocab_size, n_layers, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.mixer_blocks.append(MixerBlock(embedding_dim, self.seq_len, token_dim, channel_dim, dropout))
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_dim, vocab_size)
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        return self.mlp_head(x)


##################################################
# Go

# Hyperparameters
VOCAB_SIZE = len(tokenizer.get_vocab())
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.0

# LSTM
model = LSTMModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)

# # Sin Transformer
# NUM_HEADS = 4
# DIM_FEEDFORWARD = 64
# DROPOUT = 0.1
# model = SINTransformer(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, DIM_FEEDFORWARD, NUM_LAYERS, DROPOUT).to(DEVICE)

# # Transformer
# NUM_HEADS = 4
# DIM_FEEDFORWARD = 64
# NUM_LAYERS = 2
# DROPOUT = 0.1
# model = ROPETransformer(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, DIM_FEEDFORWARD, NUM_LAYERS, DROPOUT).to(DEVICE)

# # MLP Mixer
# token_dim = 16
# channel_dim = 16
# model = MLPMixer(seq_len=SEQ_LEN,
#                  embedding_dim=EMBEDDING_DIM,
#                  vocab_size=VOCAB_SIZE,
#                  n_layers=NUM_LAYERS,
#                  token_dim=token_dim,
#                  channel_dim=channel_dim,
#                  dropout=DROPOUT).to(DEVICE)


# Initialize the optimizer
print_model_info(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

# Training loop
for epoch in range(NUM_EPOCHS):
    train_loss = run_epoch(model, train_dl, optimizer, mode='train', device=DEVICE, clip=GRAD_CLIP, loss_fn=LOSS_FN)
    val_loss, val_metrics, val_outputs = run_epoch(model, val_dl, optimizer, mode='eval', device=DEVICE, check_accuracy=True, loss_fn=LOSS_FN)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
