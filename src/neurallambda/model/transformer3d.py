'''.

Transformer, but instead of outer pdt of QK, cubic outer pdt of QKL

PROVENANCE:
- experiment/t10_indirection
- experiment/t10_swap_03

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math
import warnings




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


'''

DecodeLayer moved to t10_swap02 temporarily


'''

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
