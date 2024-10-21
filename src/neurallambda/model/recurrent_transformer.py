'''

Recurrent Transformer:
Run a transformer, do some stuff, resume, repeat.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

# SEED = 152
# torch.manual_seed(SEED)
# random.seed(SEED)


##################################################
# Attention

class BasicMultiHeadAttention(nn.Module):
    ''' Vanilla MultiHead Attention '''

    def __init__(self, emb_dim, num_heads, bias):
        super(BasicMultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum('bhse, bhte -> bhst', q, k)  # [B, N_HEADS, S, S]
        scores = scores / self.head_dim ** 0.5  # [B, NUM_HEADS, S, S]

        if attn_mask is not None:
            scores += attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attended_values = torch.einsum('bhst, bhtd -> bshd', attn_weights, v)
        concatenated = attended_values.contiguous().view(batch_size, -1, self.emb_dim)  # [B, S, D]
        output = self.out_proj(concatenated)  # [B, S, D]
        return output


class MultiHeadAttention(nn.Module):
    '''MultiHeadAttention that retains interim calculations (KV calcs) so that
    transformers can be run in blocks across a sequence, and those blocks can
    be concattenated together to yield an identical output to a non-horizontal
    transformer.

    '''

    def __init__(self, emb_dim, num_heads, bias):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        # KV-cache for use even during training
        self.k_state = []
        self.v_state = []

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        # QKV for just new chunk
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, N_HEADS, CHUNK, D//N_HEADS]
        k_ = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, N_HEADS, S, D//N_HEADS]
        v_ = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, N_HEADS, S, D//N_HEADS]

        # Extend KV with cache
        k = torch.concat(self.k_state + [k_], dim=2)
        v = torch.concat(self.v_state + [v_], dim=2)

        # Save KV to cache
        self.k_state.append(k_)
        self.v_state.append(v_)

        scores = torch.einsum('bhse, bhte -> bhst', q, k)  # [B, N_HEADS, CHUNK, S]
        scores = scores / self.head_dim ** 0.5  # [B, NUM_HEADS, CHUNK, S]


        if attn_mask is not None:
            scores += attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attended_values = torch.einsum('bhst, bhtd -> bshd', attn_weights, v)
        # concat heads, mlp projection over current chunk
        concatenated = attended_values.contiguous().view(batch_size, -1, self.emb_dim)  # [B, CHUNK, D]
        return self.out_proj(concatenated)  # [B, CHUNK, D]


##################################################
# Decoder

def positional_encoding(emb_dim, max_len=5000):
    pos_enc = torch.zeros(max_len, emb_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc


class BasicDecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward, dropout=0.0):
        super(BasicDecoderLayer, self).__init__()
        self.self_attn = BasicMultiHeadAttention(emb_dim, num_heads, bias=True)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_dim, bias=False)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(q, k, v, mask)
        attn_output = self.dropout(attn_output)
        xs = self.norm1(q + attn_output)

        # Feed-forward network with residual connection
        ffnn_output = self.ffnn(xs)
        ffnn_output = self.dropout(ffnn_output)
        xs = self.norm2(xs + ffnn_output)

        return xs


class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(emb_dim, num_heads, bias=True)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_dim, bias=False)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(q, k, v, mask)
        attn_output = self.dropout(attn_output)
        xs = self.norm1(q + attn_output)

        # Feed-forward network with residual connection
        ffnn_output = self.ffnn(xs)
        ffnn_output = self.dropout(ffnn_output)
        xs = self.norm2(xs + ffnn_output)

        return xs


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Test accuracy of Decoders


if False:
    batch_size = 16
    seq_length = 20
    emb_dim = 256
    num_heads = 8
    dim_feedforward = 1024
    dropout = 0.1

    # Create random weights for the layers
    weights = {
        'self_attn.q_proj.weight': torch.randn((emb_dim, emb_dim)),
        'self_attn.k_proj.weight': torch.randn((emb_dim, emb_dim)),
        'self_attn.v_proj.weight': torch.randn((emb_dim, emb_dim)),
        'self_attn.out_proj.weight': torch.randn((emb_dim, emb_dim)),
        'self_attn.q_proj.bias': torch.randn(emb_dim),
        'self_attn.k_proj.bias': torch.randn(emb_dim),
        'self_attn.v_proj.bias': torch.randn(emb_dim),
        'self_attn.out_proj.bias': torch.randn(emb_dim),
        'ffnn.0.weight': torch.randn((dim_feedforward, emb_dim)),
        'ffnn.3.weight': torch.randn((emb_dim, dim_feedforward)),
        'norm1.weight': torch.randn(emb_dim),
        'norm1.bias': torch.randn(emb_dim),
        'norm2.weight': torch.randn(emb_dim),
        'norm2.bias': torch.randn(emb_dim),
    }

    # Instantiate layers
    basic_layer = BasicDecoderLayer(emb_dim, num_heads, dim_feedforward, dropout)
    recurrent_layer = DecoderLayer(emb_dim, num_heads, dim_feedforward, dropout)

    # Set weights for both layers
    with torch.no_grad():
        for name, param in basic_layer.named_parameters():
            param.copy_(weights[name])
        for name, param in recurrent_layer.named_parameters():
            param.copy_(weights[name])

    # Set both layers to evaluation mode to disable dropout
    basic_layer.eval()
    recurrent_layer.eval()

    # Convert to float64 for higher precision
    basic_layer.to(dtype=torch.float64)
    recurrent_layer.to(dtype=torch.float64)

    # Create input tensor
    x = torch.randn(batch_size, seq_length, emb_dim, dtype=torch.float64)

    # Create attention mask
    attn_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)

    with torch.no_grad():
        ##########
        # Basic layer forward pass
        basic_output = basic_layer(x, x, x, attn_mask)

        ##########
        # Recurrent layer forward pass (in chunks)
        n = 4  # Number of chunks
        chunk_size = seq_length // n
        recurrent_outputs = []
        for i in range(n):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            x_chunk = x[:, start_idx:end_idx]
            attn_mask_chunk = attn_mask[start_idx:end_idx, :end_idx]
            output_chunk = recurrent_layer(x_chunk, x_chunk, x_chunk, attn_mask_chunk)
            recurrent_outputs.append(output_chunk)

        recurrent_output = torch.cat(recurrent_outputs, dim=1)

    # Compare outputs
    max_diff = torch.max(torch.abs(basic_output - recurrent_output)).item()
    print(f"Maximum difference between BasicDecoderLayer and DecoderLayer outputs: {max_diff}")

    # Check if outputs are close enough
    result = torch.allclose(basic_output, recurrent_output, atol=1e-5)
    print(f"Decoder layers produce equivalent outputs: {result}")




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Test accuracy of Attention Implementations

if False:
    batch_size = 16
    seq_length = 10
    emb_dim = 256
    num_heads = 8

    weights = {
        'q_proj.weight': torch.randn((emb_dim, emb_dim)),
        'k_proj.weight': torch.randn((emb_dim, emb_dim)),
        'v_proj.weight': torch.randn((emb_dim, emb_dim)),
        'out_proj.weight': torch.randn((emb_dim, emb_dim)),
        'q_proj.bias': torch.randn(emb_dim),
        'k_proj.bias': torch.randn(emb_dim),
        'v_proj.bias': torch.randn(emb_dim),
        'out_proj.bias': torch.randn(emb_dim)
    }

    basic_attn = BasicMultiHeadAttention(emb_dim, num_heads, bias=True)
    with torch.no_grad():
        # weights
        basic_attn.q_proj.weight.copy_(weights['q_proj.weight'])
        basic_attn.k_proj.weight.copy_(weights['k_proj.weight'])
        basic_attn.v_proj.weight.copy_(weights['v_proj.weight'])
        basic_attn.out_proj.weight.copy_(weights['out_proj.weight'])
        # biases
        basic_attn.q_proj.bias.copy_(weights['q_proj.bias'])
        basic_attn.k_proj.bias.copy_(weights['k_proj.bias'])
        basic_attn.v_proj.bias.copy_(weights['v_proj.bias'])
        basic_attn.out_proj.bias.copy_(weights['out_proj.bias'])
    basic_attn.to(dtype=torch.float64)


    horizontal_attn = MultiHeadAttention(emb_dim, num_heads, bias=True)
    with torch.no_grad():
        # weights
        horizontal_attn.q_proj.weight.copy_(weights['q_proj.weight'])
        horizontal_attn.k_proj.weight.copy_(weights['k_proj.weight'])
        horizontal_attn.v_proj.weight.copy_(weights['v_proj.weight'])
        horizontal_attn.out_proj.weight.copy_(weights['out_proj.weight'])
        # biases
        horizontal_attn.q_proj.bias.copy_(weights['q_proj.bias'])
        horizontal_attn.k_proj.bias.copy_(weights['k_proj.bias'])
        horizontal_attn.v_proj.bias.copy_(weights['v_proj.bias'])
        horizontal_attn.out_proj.bias.copy_(weights['out_proj.bias'])
    horizontal_attn.to(dtype=torch.float64)

    # Note: add_bias_kv is not bias at all, it concats a learned term to the start of the sequence
    #    https://discuss.pytorch.org/t/how-is-add-bias-kv-implemented-in-multiheadattention/181841/4
    pytorch_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, bias=True, add_bias_kv=False)
    with torch.no_grad():
        # weights
        pytorch_attn.in_proj_weight.copy_(torch.cat([
            weights['q_proj.weight'],
            weights['k_proj.weight'],
            weights['v_proj.weight']
        ]))
        pytorch_attn.out_proj.weight.copy_(weights['out_proj.weight'])
        # biases
        pytorch_attn.in_proj_bias.copy_(torch.cat([
            weights['q_proj.bias'],
            weights['k_proj.bias'],
            weights['v_proj.bias']
        ]))
        pytorch_attn.out_proj.bias.copy_(weights['out_proj.bias'])
    pytorch_attn.to(dtype=torch.float64)

    # random input
    x = torch.randn(batch_size, seq_length, emb_dim, dtype=torch.float64)
    attn_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)

    with torch.no_grad():
        ##########
        # Basic Attention
        basic_output = basic_attn(x, x, x, attn_mask)

        ##########
        # Horizontal Attn
        #   Test the Chunking ability of the HorizontalMultHeadAttention
        n = 5
        chunk_size = seq_length // n

        outputs = []
        for i in range(n):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            x_chunk = x[:, start_idx:end_idx]

            # Slice the attention mask
            # We take all rows up to the current chunk, and columns up to the end of the current chunk
            attn_mask_chunk = attn_mask[start_idx:end_idx, :end_idx]
            output_chunk = horizontal_attn(x_chunk, x_chunk, x_chunk, attn_mask_chunk)
            outputs.append(output_chunk)

        # Concatenate all the outputs
        horizontal_output = torch.cat(outputs, dim=1)

        ##########
        # Pytorch Attn
        pytorch_output, _ = pytorch_attn(x, x, x, attn_mask=attn_mask, need_weights=False, is_causal=True)

    # Compare outputs
    diffs = f'''
DIFFS
{torch.max(torch.abs(horizontal_output - pytorch_output)).item()=}
{torch.max(torch.abs(basic_output - pytorch_output)).item()=}
{torch.max(torch.abs(horizontal_output - basic_output)).item()=}
'''
    print(diffs)

    result = torch.allclose(horizontal_output, pytorch_output, atol=1e-5)

    print(f"Attention blocks produce identical outputs: {result}")
