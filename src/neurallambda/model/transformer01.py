'''.

Reference implementation, sin/cos position encoding (not embedding).

PROVENANCE:
- experiment/t10_indirection.py

'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, use_wq=True, use_wk=True, use_wv=True, use_wout=True):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wv = use_wv
        self.use_wout = use_wout

        if use_wq:
            self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wk:
            self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wv:
            self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        if use_wout:
            self.out = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, q, k, v, mask=None, attn_nonlin=None, alibi_bias=None):
        '''
        Args:
          attn_nonlin: {None, 'softmax', 'linear', 'sigmoid', 'tanh', 'relu'}
          use_wq, use_wk, use_wv: whether or not to project QKV first
          alibi_bias: for ALiBI
        '''
        assert attn_nonlin in {None, 'none', 'softmax', 'sigmoid', 'tanh', 'relu'}
        batch_size, device = q.size(0), q.device

        # Linear projections
        q = self.query(q) if self.use_wq else q  # [B, S, D]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        k = self.key(k) if self.use_wk else k  # [B, S, D]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        k = k.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        v = self.value(v) if self.use_wv else v  # [B, S, D]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)  # [B, S, N_HEADS, HEAD_DIM]
        v = v.transpose(1, 2)  # [B, NUM_HEADS, S, HEAD_DIM]


        # Scaled dot-product attention
        scores = einsum('bhse, bhte -> bhst', q, k)  # [B, N_HEADS, S, S]
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, device=device))  # [B, NUM_HEADS, S, S]

        if mask is not None and mask.ndim == 2:  # [S, S]
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)  # [B, N_HEADS, S, S]
            scores += mask  # [B, S, S]
        elif mask is not None and mask.ndim == 3:  # [B, S, S]
            mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)  # [B, N_HEADS, S, S]
            scores += mask  # [B, NUM_HEADS, S, S]

        if alibi_bias is not None:
            scores += alibi_bias

        match attn_nonlin:
            case 'none' | None:
                attention_weights = scores
            case 'softmax':
                # correct softmax dim is -1:
                #   https://github.com/pytorch/pytorch/blob/03725a05127296c581a434cebe727e95caf5c14f/torch/nn/functional.py#L5026
                attention_weights = F.softmax(scores, dim=-1)
            case 'sigmoid':
                attention_weights = scores.sigmoid()
            case 'tanh':
                attention_weights = scores.tanh()
            case 'relu':
                attention_weights = scores.relu()

        # attention_weights = F.dropout(attention_weights, p=dropout_prob, training=self.training)  # [B, NUM_HEADS, S, S]
        # attended_values = torch.matmul(attention_weights, v)  # [B, N_HEADS, S, HEAD_DIM]
        attended_values = torch.einsum('bhst, bhtd -> bshd', attention_weights, v)

        # Concatenation and linear transformation
        # concatenated = attended_values.transpose(1, 2)  # [B, S, N_HEADS, HEAD_DIM]
        concatenated = attended_values.contiguous().view(batch_size, -1, self.emb_dim)  # [B, S, D]
        output = self.out(concatenated) if self.use_wout else concatenated  # [B, S, D]

        if 'DEBUG_MULTIHEAD_ATTENTION' in os.environ and os.environ['DEBUG_MULTIHEAD_ATTENTION'] in {'true', 'True', '1'}:
            import matplotlib.pyplot as plt

            def visualize_matrices(tensor):
                tensor = tensor.cpu().detach().numpy()
                N = tensor.shape[0]
                grid_size = int(math.ceil(math.sqrt(N)))

                fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
                fig.subplots_adjust(hspace=0.4, wspace=0.4)

                for i, ax in enumerate(axes.flat):
                    if i < N:
                        matrix = tensor[i]
                        ax.imshow(matrix, cmap='viridis')
                        ax.set_title(f'Matrix {i+1}')
                        ax.axis('off')
                    else:
                        ax.axis('off')

                plt.tight_layout()
                plt.show()


            breakpoint()

        return output


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward=512, dropout=0.1, use_wq=True, use_wk=True, use_wv=True, use_wout=True):
        super(DecoderLayer, self).__init__()
        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wv = use_wv
        self.use_wout = use_wout

        self.self_attn = MultiHeadAttention(emb_dim, num_heads, self.use_wq, self.use_wk, self.use_wv, self.use_wout)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_dim, bias=False)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, attn_nonlin=None, alibi_bias=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(q, k, v, mask, attn_nonlin, alibi_bias)
        attn_output = self.dropout(attn_output)
        xs = self.norm1(q + attn_output)

        # Feed-forward network with residual connection
        ffnn_output = self.ffnn(xs)
        ffnn_output = self.dropout(ffnn_output)
        xs = self.norm2(xs + ffnn_output)

        return xs


def positional_encoding(emb_dim, max_len=5000):
    pos_enc = torch.zeros(max_len, emb_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc


def generate_square_subsequent_mask(sz):
    ''' causal masking '''
    mask = (torch.triu(torch.ones(sz, sz))).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_syntax_mask(sz):
    ''' causal mask. also don't attend to self. Try to make "program" selection not dependent on current word. '''
    mask = (torch.triu(torch.ones(sz, sz, dtype=torch.int))).transpose(0, 1)
    mask -= torch.eye(sz, dtype=torch.int)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask[0][0] = 0  # can pay attn to bos
    return mask



##################################################
# Implementation Example
#
# NOTE: this is currently rather specialized to a specific experiment (one using
#       3 inputs to the transformer: tags, col1, col2. It also includes things like recurrence

class TransformerModel(nn.Module):
    def __init__(self, tokenizer, emb_dim, num_heads, num_layers, dim_feedforward, num_recurrence, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_recurrence = num_recurrence

        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)
        self.pos_encoding = positional_encoding(emb_dim)
        self.layers = nn.ModuleList([DecoderLayer(emb_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, self.vocab_size, bias=False)
        # self.fc_out = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, addresses_ids, inp_tags_ids, inp_col1_ids, inp_col2_ids):
        B, S = addresses_ids.shape

        addresses = self.embeddings(addresses_ids) * math.sqrt(self.emb_dim)
        inp_tags  = self.embeddings(inp_tags_ids) * math.sqrt(self.emb_dim)
        inp_col1  = self.embeddings(inp_col1_ids) * math.sqrt(self.emb_dim)
        inp_col2  = self.embeddings(inp_col2_ids) * math.sqrt(self.emb_dim)

        embs = torch.cat([inp_tags, inp_col1, inp_col2], dim=1)
        pos = self.pos_encoding[:S, :].to('cuda')
        # pos3 = torch.cat([pos] * 3, dim=0)
        pos3 = torch.cat([addresses] * 3, dim=1) * 1e-1
        xs = embs
        # xs = self.dropout(xs)

        for i in range(self.num_recurrence):
            for j, layer in enumerate(self.layers):
                # xs = layer(q=xs, k=xs, v=xs, p=None, xs_mask=None)
                # xs = layer(q=xs + pos3, k=xs + pos3, v=xs, p=None, xs_mask=None)
                if j == 0:  # only add at first layer (and each recurrence)
                    xs = layer(q=xs + pos3, k=xs + pos3, v=xs + pos3, p=None, xs_mask=None)
                else:
                    xs = layer(q=xs, k=xs, v=xs, p=None, xs_mask=None)

        tags, col1, col2 = torch.chunk(xs, 3, dim=1)
        # xs = self.norm(xs)
        # output = self.fc_out(xs)

        # selection = einsum('bsd, bsd -> bsd',
        #                          F.normalize(output, dim=2),
        #                          F.normalize(keys, dim=2),
        #                          )
        # output = einsum('bsd, bsd -> bsd',
        #                       selection,
        #                       embs,
        #                       )

        return (
            self.fc_out(self.norm(tags)),
            self.fc_out(self.norm(col1)),
            self.fc_out(self.norm(col2)),
        )
