'''.

Binding Attention

Replace self attention's QK weights with an LSTM.

PROVENANCE:
- model/transformer01

'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math


class FullLSTM(nn.Module):
    ''' Transductive LLM, records and stacks each output, so, same sequence length as input. '''
    def __init__(self, input_size, hidden_size, dropout):
        super(FullLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h, c = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(x.device), torch.zeros(batch_size, self.lstm_cell.hidden_size).to(x.device)
        outputs = []

        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        # output = self.layer_norm(output)
        return output


class FullRNN(nn.Module):
    ''' Transductive RNN, records and stacks each output, so, same sequence length as input. '''
    def __init__(self, input_size, hidden_size, dropout):
        super(FullRNN, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.rnn_cell.hidden_size).to(x.device)
        outputs = []

        for t in range(seq_len):
            h = self.rnn_cell(x[:, t, :], h)
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        # output = self.layer_norm(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, use_wq=True, use_wk=True, use_wv=True, use_wout=True):
        super(MultiHeadAttention, self).__init__()
        DROPOUT = 0.0  # todo

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        self.use_wq = use_wq
        self.use_wk = use_wk
        self.use_wv = use_wv
        self.use_wout = use_wout

        if use_wq:
            self.query = FullLSTM(emb_dim, hidden_size=emb_dim, dropout=DROPOUT)
        if use_wk:
            self.key = FullLSTM(emb_dim, hidden_size=emb_dim, dropout=DROPOUT)
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
