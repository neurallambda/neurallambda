'''

A  that handles both a "forward" and "backward" pass, similar to pytorch. Unlike pytorch, it calculates gradients by hand.

Each core function in the library is a class, like Linear, that has a forward class function that also returns a cache used by the backward pass, and the backward class function.


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

torch.manual_seed(42)

DO_TEST = False

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return torch.mean((inputs - targets) ** 2)

    def backward(self, inputs, targets):
        n = inputs.numel()
        return 2 * (inputs - targets) / n

def test_mse_loss():
    for _ in range(10):
        inputs = torch.randn(10, 5, requires_grad=True)
        targets = torch.randn(10, 5)

        # Custom implementation
        mse_loss = MSELoss()
        loss = mse_loss(inputs, targets)
        grad = mse_loss.backward(inputs, targets)

        # PyTorch implementation
        mse_loss_torch = nn.MSELoss()
        loss_torch = mse_loss_torch(inputs, targets)
        loss_torch.backward()

        assert torch.isclose(loss, loss_torch, rtol=1e-5, atol=1e-8), \
            f"Loss mismatch: Custom={loss:.6f}, PyTorch={loss_torch:.6f}"
        assert torch.allclose(grad, inputs.grad, rtol=1e-5, atol=1e-8), \
            "Gradient mismatch"

    print("MSELoss tests passed successfully!")

if DO_TEST:
    test_mse_loss()

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_init = nn.Parameter(torch.zeros(out_features)) if bias else None

        # online-trained "params" need to be kept separate from pytorch-trained
        # params, and also can't be set during model init or they end up on
        # separate devices.
        self.weight = None
        self.bias = None


    def oforward(self, x):
        # on first forward pass, set these
        if self.weight is None:
            self.weight = self.weight_init + 0
            self.bias = self.bias_init + 0

        out = torch.einsum('...i,oi->...o', x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out, {'x': x}

    def obackward(self, grad_output, cache):
        x = cache['x']

        grad_input = torch.einsum('...o,oi->...i', grad_output, self.weight)

        grad_weight = torch.einsum('...i,...o->...oi', x, grad_output)
        if grad_weight.ndim > 2:
            grad_weight = grad_weight.sum(dim=tuple(range(grad_weight.ndim - 2)))

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if self.bias is not None else None

        return grad_input, {
            'weight': grad_weight,
            'bias': grad_bias
        }

    def grad_step(self, grads, lr):
        self.weight = self.weight + grads['weight'] * lr
        self.bias = self.bias + grads['bias'] * lr

def test_linear_layer():
    in_features, out_features, batch_size = 5, 3, 10

    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)

    x = torch.randn(batch_size, in_features, requires_grad=True)

    # Custom implementation
    linear_custom = Linear(in_features, out_features, bias=True)
    linear_custom.set_parameters(weight, bias)
    y_custom, cache = linear_custom(x)

    # PyTorch implementation
    linear_torch = nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear_torch.weight.copy_(weight)
        linear_torch.bias.copy_(bias)
    y_torch = linear_torch(x)

    assert torch.allclose(y_custom, y_torch, rtol=1e-5, atol=1e-8), "Forward pass mismatch"

    grad_output = torch.randn(*y_custom.shape)

    # Custom backward
    grads_custom = linear_custom.backward(grad_output, cache)
    grad_x_custom = grads_custom['grad_x']
    grad_weight_custom = grads_custom['grad_weight']
    grad_bias_custom = grads_custom['grad_bias']

    # PyTorch backward
    y_torch.backward(grad_output)

    assert torch.allclose(grad_x_custom, x.grad, rtol=1e-5, atol=1e-8), "Input gradient mismatch"
    assert torch.allclose(grad_weight_custom, linear_torch.weight.grad, rtol=1e-5, atol=1e-8), "Weight gradient mismatch"
    assert torch.allclose(grad_bias_custom, linear_torch.bias.grad, rtol=1e-5, atol=1e-8), "Bias gradient mismatch"

    print("All Linear layer tests passed successfully!")

if DO_TEST:
    test_linear_layer()

class MatMul(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        output = torch.einsum('...ij,...jk->...ik', x1, x2)
        cache = {'x1': x1, 'x2': x2}
        return output, cache

    def backward(self, grad_output, cache):
        x1, x2 = cache['x1'], cache['x2']

        grad_x1 = torch.einsum('...ik,...jk->...ij', grad_output, x2)
        grad_x2 = torch.einsum('...ij,...ik->...jk', x1, grad_output)

        return grad_x1, grad_x2

def test_matmul():
    x1 = torch.randn(2, 3, 4, requires_grad=True)
    x2 = torch.randn(2, 4, 5, requires_grad=True)

    # Custom implementation
    matmul_custom = MatMul()
    y_custom, cache = matmul_custom(x1, x2)

    # PyTorch implementation
    y_torch = torch.matmul(x1, x2)

    assert torch.allclose(y_custom, y_torch, rtol=1e-5, atol=1e-8), "MatMul forward pass mismatch"

    grad_output = torch.randn(*y_custom.shape)

    grad_x1_custom, grad_x2_custom = matmul_custom.backward(grad_output, cache)
    y_torch.backward(grad_output)

    assert torch.allclose(grad_x1_custom, x1.grad, rtol=1e-5, atol=1e-8), "MatMul grad_x1 mismatch"
    assert torch.allclose(grad_x2_custom, x2.grad, rtol=1e-5, atol=1e-8), "MatMul grad_x2 mismatch"

    print("MatMul test passed successfully!")

if DO_TEST:
    test_matmul()

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

    def backward(self, grad_output, x):
        return grad_output * (x > 0)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def backward(self, grad_output, previous_output):
        return grad_output * previous_output * (1 - previous_output)

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)

    def backward(self, grad_output, previous_output):
        return previous_output * (grad_output - (grad_output * previous_output).sum(dim=self.dim, keepdim=True))

def test_activations():
    x = torch.randn(10, 5, requires_grad=True)

    # ReLU
    relu_custom = ReLU()
    y_relu_custom = relu_custom(x)
    y_relu_torch = F.relu(x)
    assert torch.allclose(y_relu_custom, y_relu_torch, rtol=1e-5, atol=1e-8), "ReLU forward pass mismatch"

    grad_output = torch.randn(*y_relu_custom.shape)
    grad_input_custom = relu_custom.backward(grad_output, x)
    y_relu_torch.backward(grad_output)
    assert torch.allclose(grad_input_custom, x.grad, rtol=1e-5, atol=1e-8), "ReLU backward pass mismatch"

    # Sigmoid
    x.grad = None
    sigmoid_custom = Sigmoid()
    y_sigmoid_custom = sigmoid_custom(x)
    y_sigmoid_torch = torch.sigmoid(x)
    assert torch.allclose(y_sigmoid_custom, y_sigmoid_torch, rtol=1e-5, atol=1e-8), "Sigmoid forward pass mismatch"

    grad_output = torch.randn(*y_sigmoid_custom.shape)
    grad_input_custom = sigmoid_custom.backward(grad_output, y_sigmoid_custom)
    y_sigmoid_torch.backward(grad_output)
    assert torch.allclose(grad_input_custom, x.grad, rtol=1e-5, atol=1e-8), "Sigmoid backward pass mismatch"

    # Softmax
    x.grad = None
    softmax_custom = Softmax()
    y_softmax_custom = softmax_custom(x)
    y_softmax_torch = F.softmax(x, dim=-1)
    assert torch.allclose(y_softmax_custom, y_softmax_torch, rtol=1e-5, atol=1e-8), "Softmax forward pass mismatch"

    grad_output = torch.randn(*y_softmax_custom.shape)
    grad_input_custom = softmax_custom.backward(grad_output, y_softmax_custom)
    y_softmax_torch.backward(grad_output)
    assert torch.allclose(grad_input_custom, x.grad, rtol=1e-5, atol=1e-8), "Softmax backward pass mismatch"

    print("All activation function tests passed successfully!")

if DO_TEST:
    test_activations()

class FFNNCustom(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = Linear(input_size, hidden_size, bias=True)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        l1_output, linear1_cache = self.linear1(x)
        r_output = self.relu(l1_output)
        l2_output, linear2_cache = self.linear2(r_output)
        cache = {
            'linear1_cache': linear1_cache,
            'r_output': r_output,
            'linear2_cache': linear2_cache,
            'l2_output': l2_output
        }
        return l2_output, cache

    def backward(self, grad_loss, cache):
        linear1_cache = cache['linear1_cache']
        r_output = cache['r_output']
        linear2_cache = cache['linear2_cache']
        l2_output = cache['l2_output']
        grads_l2 = self.linear2.backward(grad_loss, linear2_cache)
        grad_r_output = grads_l2['grad_x']
        grad_l1_output = self.relu.backward(grad_r_output, r_output)
        grads_l1 = self.linear1.backward(grad_l1_output, linear1_cache)

        return {
            'linear1.weight': grads_l1['grad_weight'],
            'linear1.bias': grads_l1['grad_bias'],
            'linear2.weight': grads_l2['grad_weight'],
            'linear2.bias': grads_l2['grad_bias']
        }

    def get_parameters(self):
        return {
            'linear1.weight': self.linear1.weight,
            'linear1.bias': self.linear1.bias,
            'linear2.weight': self.linear2.weight,
            'linear2.bias': self.linear2.bias
        }

    def set_parameters(self, params):
        self.linear1.weight = params['linear1.weight']
        self.linear1.bias = params['linear1.bias']
        self.linear2.weight = params['linear2.weight']
        self.linear2.bias = params['linear2.bias']


class FFNNPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def test_integrated_model():
    input_size, hidden_size, output_size = 10, 5, 1
    batch_size = 32

    # Initialize models
    model_custom = FFNNCustom(input_size, hidden_size, output_size)
    model_torch = FFNNPyTorch(input_size, hidden_size, output_size)

    # Generate random parameters
    params = {
        'linear1.weight': torch.randn(hidden_size, input_size),
        'linear1.bias': torch.randn(hidden_size),
        'linear2.weight': torch.randn(output_size, hidden_size),
        'linear2.bias': torch.randn(output_size)
    }

    # Set parameters for both models
    model_custom.set_parameters(params)
    with torch.no_grad():
        for name, param in model_torch.named_parameters():
            param.copy_(params[name])

    # Generate input data
    x = torch.randn(batch_size, input_size, requires_grad=True)

    # Forward pass
    y_custom, ffnn_cache = model_custom(x)
    y_torch = model_torch(x)

    assert torch.allclose(y_custom, y_torch, rtol=1e-5, atol=1e-8), "Forward pass mismatch"

    # Generate target data
    target = torch.randn(batch_size, output_size)

    # Compute loss
    mse_loss = MSELoss()
    loss_custom = mse_loss(y_custom, target)
    loss_torch = nn.MSELoss()(y_torch, target)

    assert torch.allclose(loss_custom, loss_torch, rtol=1e-5, atol=1e-8), "Loss mismatch"

    # Backward pass
    loss_torch.backward()
    grad_loss = mse_loss.backward(y_custom, target)
    grads_custom = model_custom.backward(grad_loss, ffnn_cache)

    # Mapping between custom and PyTorch parameter names
    param_mapping = {
        'linear1.weight': 'linear1.weight',
        'linear1.bias': 'linear1.bias',
        'linear2.weight': 'linear2.weight',
        'linear2.bias': 'linear2.bias'
    }

    # Compare gradients
    for custom_name, torch_name in param_mapping.items():
        custom_grad = grads_custom[custom_name]
        torch_grad = getattr(model_torch, torch_name.split('.')[0]).weight.grad if 'weight' in torch_name else getattr(model_torch, torch_name.split('.')[0]).bias.grad

        assert torch.allclose(custom_grad, torch_grad, rtol=1e-5, atol=1e-8), f"Gradient mismatch for {custom_name}"

    print("All integrated model tests passed successfully!")

if DO_TEST:
    test_integrated_model()


##################################################
# Self Attention


class Transpose(Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return torch.permute(x, self.axes)

    def backward(self, grad_output):
        reverse_axes = tuple(torch.argsort(torch.tensor(self.axes)).tolist())
        return torch.permute(grad_output, reverse_axes)

def test_transpose():
    x = torch.randn(2, 3, 4, 5, requires_grad=True)

    transpose_custom = Transpose((0, 2, 1, 3))
    y_custom = transpose_custom(x)
    y_torch = x.permute(0, 2, 1, 3)

    assert torch.allclose(y_custom, y_torch)

    grad_output = torch.randn(2, 4, 3, 5)

    grad_input_custom = transpose_custom.backward(grad_output)
    y_torch.backward(grad_output)

    assert torch.allclose(grad_input_custom, x.grad)
    print('Transpose tests passed')

if DO_TEST:
    test_transpose()

class View(Module):
    def __init__(self, from_shape, to_shape):
        super().__init__()
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x):
        return x.view(self.to_shape)

    def backward(self, grad_output):
        return grad_output.view(self.from_shape)

class Reshape(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, shape):
        cache = {
            'from_shape': x.shape
        }
        return x.reshape(shape), cache

    def backward(self, grad_output, cache):
        return grad_output.reshape(cache['from_shape'])

def test_view():
    x = torch.randn(2, 3, 4, requires_grad=True)

    view_custom = View((2, 3, 4), (2, 12))
    y_custom = view_custom(x)
    y_torch = x.view(2, 12)

    assert torch.allclose(y_custom, y_torch)

    grad_output = torch.randn(2, 12)

    grad_input_custom = view_custom.backward(grad_output)
    y_torch.backward(grad_output)

    assert torch.allclose(grad_input_custom, x.grad)
    print('View tests passed')

def test_reshape():
    x = torch.randn(2, 3, 4, requires_grad=True)

    reshape_custom = Reshape()
    y_custom, reshape_cache = reshape_custom(x, (2, 12))
    y_torch = x.reshape(2, 12)
    assert torch.allclose(y_custom, y_torch)

    grad_output = torch.randn(2, 12)

    grad_input_custom = reshape_custom.backward(grad_output, reshape_cache)
    y_torch.backward(grad_output)

    assert torch.allclose(grad_input_custom, x.grad)
    print('Reshape tests passed')

if DO_TEST:
    test_view()
    test_reshape()

##################################################


class SelfAttention(Module):
    '''Self-attention with a custom (but equivalent) backward pass. This was for
work on TTT, but really this should all be calculated by torch's graph-tracking
abilities.'''

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = torch.randn(3 * embed_dim, embed_dim)
        self.in_proj_bias = torch.randn(3 * embed_dim) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        self.batch_first = batch_first

        self.reshape = Reshape()
        self.softmax = Softmax(dim=-1)

        self.transpose_1_0_2 = Transpose((1, 0, 2))
        self.transpose_0_2_1 = Transpose((0, 2, 1))

        self.matmul = MatMul()

    def forward(self, query, key, value, need_weights=True, attn_mask=None):
        cache = {}

        if self.batch_first:
            query = self.transpose_1_0_2(query)
            key = self.transpose_1_0_2(key)
            value = self.transpose_1_0_2(value)

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        w = self.in_proj_weight
        b = self.in_proj_bias
        E = query.shape[-1]

        if torch.equal(query, key) and torch.equal(key, value):
            qkv_proj, qkv_proj_cache = self.matmul(query, w.T)
            cache['qkv_proj_cache'] = qkv_proj_cache
            if b is not None:
                qkv_proj += b
            q, k, v = torch.chunk(qkv_proj, 3, dim=-1)
        else:
            w_q, w_k, w_v = w[:E], w[E:2*E], w[2*E:]
            b_q, b_k, b_v = (b[:E], b[E:2*E], b[2*E:]) if b is not None else (None, None, None)
            q, q_proj_cache = self.matmul(query, w_q.T)
            k, k_proj_cache = self.matmul(key, w_k.T)
            v, v_proj_cache = self.matmul(value, w_v.T)
            cache['q_proj_cache'] = q_proj_cache
            cache['k_proj_cache'] = k_proj_cache
            cache['v_proj_cache'] = v_proj_cache

            if b is not None:
                q += b_q
                k += b_k
                v += b_v

        # Reshaping
        q, q_reshape_cache = self.reshape(q, (tgt_len, bsz * self.num_heads, self.head_dim))
        q = self.transpose_1_0_2(q)

        k, k_reshape_cache = self.reshape(k, (src_len, bsz * self.num_heads, self.head_dim))
        k = self.transpose_1_0_2(k)

        v, v_reshape_cache = self.reshape(v, (src_len, bsz * self.num_heads, self.head_dim))
        v = self.transpose_1_0_2(v)

        k_t = self.transpose_0_2_1(k)

        scores, scores_cache = self.matmul(q, k_t)
        cache['scores_cache'] = scores_cache
        scores = scores / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_output_weights = self.softmax(scores)
        attn_output, attn_output_cache = self.matmul(attn_output_weights, v)
        cache['attn_output_cache'] = attn_output_cache

        attn_output = self.transpose_1_0_2(attn_output)
        attn_output, attn_output_reshape_cache = self.reshape(attn_output, (tgt_len, bsz, embed_dim))
        attn_output, out_proj_cache = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = self.transpose_1_0_2(attn_output)

        cache.update({
            'q_reshape_cache': q_reshape_cache,
            'k_reshape_cache': k_reshape_cache,
            'v_reshape_cache': v_reshape_cache,
            'attn_output_reshape_cache': attn_output_reshape_cache,
            'out_proj_cache': out_proj_cache,
            'attn_output_weights': attn_output_weights,
        })

        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.mean(dim=1)  # Average over heads
            return attn_output, attn_output_weights, cache
        else:
            return attn_output, None, cache

    def backward(self, grad_output, cache):
        if self.batch_first:
            grad_output = self.transpose_1_0_2(grad_output)

        out_proj_grads = self.out_proj.backward(grad_output, cache['out_proj_cache'])
        grad_attn_output = out_proj_grads['grad_x']
        grad_out_proj_weight, grad_out_proj_bias = out_proj_grads['grad_weight'], out_proj_grads['grad_bias']

        grad_attn_output = self.reshape.backward(grad_attn_output, cache['attn_output_reshape_cache'])
        grad_attn_output = self.transpose_1_0_2(grad_attn_output)

        grad_attn_output_weights, grad_v = self.matmul.backward(grad_attn_output, cache['attn_output_cache'])

        grad_scores = self.softmax.backward(grad_attn_output_weights, cache['attn_output_weights'])

        grad_scores = grad_scores / (self.head_dim ** 0.5)

        grad_q, grad_k_t = self.matmul.backward(grad_scores, cache['scores_cache'])
        grad_k = self.transpose_0_2_1(grad_k_t)

        grad_q = self.transpose_1_0_2(grad_q)
        grad_k = self.transpose_1_0_2(grad_k)
        grad_v = self.transpose_1_0_2(grad_v)

        grad_q = self.reshape.backward(grad_q, cache['q_reshape_cache'])
        grad_k = self.reshape.backward(grad_k, cache['k_reshape_cache'])
        grad_v = self.reshape.backward(grad_v, cache['v_reshape_cache'])

        if 'qkv_proj_cache' in cache:
            grad_qkv_proj = torch.cat([grad_q, grad_k, grad_v], dim=-1)
            if self.in_proj_bias is not None:
                grad_in_proj_bias = grad_qkv_proj.sum(dim=(0, 1))
            grad_query, grad_in_proj_weight = self.matmul.backward(grad_qkv_proj, cache['qkv_proj_cache'])
            grad_in_proj_weight = grad_in_proj_weight.sum(dim=0).T
        else:
            grad_query, grad_wq = self.matmul.backward(grad_q, cache['q_proj_cache'])
            grad_key, grad_wk = self.matmul.backward(grad_k, cache['k_proj_cache'])
            grad_value, grad_wv = self.matmul.backward(grad_v, cache['v_proj_cache'])
            grad_in_proj_weight = torch.cat([
                grad_wq.sum(dim=0).T,
                grad_wk.sum(dim=0).T,
                grad_wv.sum(dim=0).T
            ], dim=0)
            if self.in_proj_bias is not None:
                grad_in_proj_bias = torch.cat([
                    grad_q.sum(dim=(0, 1)),
                    grad_k.sum(dim=(0, 1)),
                    grad_v.sum(dim=(0, 1))
                ])

        if self.batch_first:
            grad_query = self.transpose_1_0_2(grad_query)
            if 'qkv_proj_cache' not in cache:
                grad_key = self.transpose_1_0_2(grad_key)
                grad_value = self.transpose_1_0_2(grad_value)

        grads = {
            'in_proj_weight': grad_in_proj_weight,
            'out_proj.weight': grad_out_proj_weight,
        }
        if self.in_proj_bias is not None:
            grads['in_proj_bias'] = grad_in_proj_bias
        if self.out_proj.bias is not None:
            grads['out_proj.bias'] = grad_out_proj_bias

        if 'qkv_proj_cache' in cache:
            return grad_query, grads
        else:
            return grad_query, grad_key, grad_value, grads

def test_multihead_attention_equivalence():
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_length = 10

    custom_attn = SelfAttention(embed_dim, num_heads, bias=True, batch_first=True)
    torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, bias=True, batch_first=True)

    in_proj_weight = torch.randn(3 * embed_dim, embed_dim)
    in_proj_bias = torch.randn(3 * embed_dim)
    out_proj_weight = torch.randn(embed_dim, embed_dim)
    out_proj_bias = torch.randn(embed_dim)

    custom_attn.in_proj_weight = in_proj_weight
    custom_attn.in_proj_bias = in_proj_bias
    custom_attn.out_proj.weight = out_proj_weight
    custom_attn.out_proj.bias = out_proj_bias

    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(in_proj_weight)
        torch_mha.in_proj_bias.copy_(in_proj_bias)
        torch_mha.out_proj.weight.copy_(out_proj_weight)
        torch_mha.out_proj.bias.copy_(out_proj_bias)

    x = torch.randn(batch_size, seq_length, embed_dim)

    custom_output, custom_attn, _ = custom_attn(x, x, x, need_weights=True)
    torch_output, torch_attn = torch_mha(x, x, x, need_weights=True)

    output_close = torch.allclose(custom_output, torch_output, rtol=1e-3, atol=1e-3)
    attn_close = torch.allclose(custom_attn, torch_attn, rtol=1e-4, atol=1e-4)

    print(f"Outputs close: {output_close}")
    print(f"Attention weights close: {attn_close}")

    print(f'{torch_output[0,0,:3]=}')
    print(f'{custom_output[0,0,:3]=}')

    assert output_close and attn_close, "Outputs or attention weights do not match"

    print("Test passed: custom and PyTorch implementations produce equivalent results.")


def test_self_attention_gradients():
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_length = 10

    custom_attn = SelfAttention(embed_dim, num_heads, bias=True, batch_first=True)
    torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, bias=True, batch_first=True)

    in_proj_weight = torch.randn(3 * embed_dim, embed_dim, dtype=torch.float64)
    in_proj_bias = torch.randn(3 * embed_dim, dtype=torch.float64)
    out_proj_weight = torch.randn(embed_dim, embed_dim, dtype=torch.float64)
    out_proj_bias = torch.randn(embed_dim, dtype=torch.float64)

    custom_attn.in_proj_weight = in_proj_weight
    custom_attn.in_proj_bias = in_proj_bias
    custom_attn.out_proj.weight = out_proj_weight
    custom_attn.out_proj.bias = out_proj_bias

    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(in_proj_weight)
        torch_mha.in_proj_bias.copy_(in_proj_bias)
        torch_mha.out_proj.weight.copy_(out_proj_weight)
        torch_mha.out_proj.bias.copy_(out_proj_bias)
    torch_mha.to(dtype=torch.float64)

    x = torch.randn(batch_size, seq_length, embed_dim, dtype=torch.float64, requires_grad=True)

    custom_output, _, cache = custom_attn(x, x, x)
    torch_output, _ = torch_mha(x, x, x)

    output_grad = torch.randn_like(custom_output)

    # Handle the case where query, key, and value are the same
    custom_grad_input, custom_grads = custom_attn.backward(output_grad, cache)

    torch_output.backward(output_grad)

    rtol, atol = 1e-3, 1e-3

    print('Input grad diff:', (custom_grad_input - x.grad).abs().max().item())
    print('in_proj_weight diff:', (custom_grads['in_proj_weight'] - torch_mha.in_proj_weight.grad).abs().max().item())
    print('in_proj_bias diff:', (custom_grads['in_proj_bias'] - torch_mha.in_proj_bias.grad).abs().max().item())
    print('out_proj.weight diff:', (custom_grads['out_proj.weight'] - torch_mha.out_proj.weight.grad).abs().max().item())
    print('out_proj.bias diff:', (custom_grads['out_proj.bias'] - torch_mha.out_proj.bias.grad).abs().max().item())

    assert torch.allclose(custom_grad_input, x.grad, rtol=rtol, atol=atol), "Input gradients do not match"
    assert torch.allclose(custom_grads['in_proj_weight'], torch_mha.in_proj_weight.grad, rtol=rtol, atol=atol), "in_proj_weight gradients do not match"
    assert torch.allclose(custom_grads['in_proj_bias'], torch_mha.in_proj_bias.grad, rtol=rtol, atol=atol), "in_proj_bias gradients do not match"
    assert torch.allclose(custom_grads['out_proj.weight'], torch_mha.out_proj.weight.grad, rtol=rtol, atol=atol), "out_proj.weight gradients do not match"
    assert torch.allclose(custom_grads['out_proj.bias'], torch_mha.out_proj.bias.grad, rtol=rtol, atol=atol), "out_proj.bias gradients do not match"

    print("All gradient checks passed!")

if DO_TEST:
    test_multihead_attention_equivalence()
    test_self_attention_gradients()
    print('ALL DONE')


##################################################

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm

def test_layer_norm():
    batch_size, seq_len, hidden_size = 2, 3, 4
    eps = 1e-5

    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

    custom_ln = LayerNorm(hidden_size, eps=eps)
    torch_ln = nn.LayerNorm(hidden_size, eps=eps)

    # Set the same weights and biases for both implementations
    with torch.no_grad():
        torch_ln.weight.copy_(custom_ln.weight)
        torch_ln.bias.copy_(custom_ln.bias)

    custom_output = custom_ln(x)
    torch_output = torch_ln(x)

    assert torch.allclose(custom_output, torch_output, rtol=1e-4, atol=1e-4)

    # Test backward pass
    grad_output = torch.randn_like(custom_output)
    custom_output.backward(grad_output)
    torch_output.backward(grad_output)

    assert torch.allclose(x.grad, x.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(custom_ln.weight.grad, torch_ln.weight.grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(custom_ln.bias.grad, torch_ln.bias.grad, rtol=1e-4, atol=1e-4)

    print("LayerNorm test passed successfully!")

if DO_TEST:
    test_layer_norm()


##################################################
# Sandbox

# def sgd_update(params: Dict[str, torch.Tensor], grads: Dict[str, torch.Tensor], learning_rate: float) -> Dict[str, torch.Tensor]:
#     """Simple SGD update"""
#     return {k: v - learning_rate * grads[k] for k, v in params.items()}

def sgd_update(params: Dict[str, torch.Tensor], grads: Dict[str, torch.Tensor], learning_rate: float) -> Dict[str, torch.Tensor]:
    """Simple SGD update with detailed key assertion"""
    params_keys = set(params.keys())
    grads_keys = set(grads.keys())
    assert params_keys == grads_keys, f"Keys mismatch. Missing in grads: {params_keys - grads_keys}. Extra in grads: {grads_keys - params_keys}"
    return {k: v - learning_rate * grads[k] for k, v in params.items()}

def test_ffnn_custom_overfitting():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate training data
    input_size = 32
    hidden_size = 64
    output_size = 10

    num_samples = 10
    X = torch.randn(num_samples, input_size)
    Y = torch.randn(num_samples, output_size)

    # Instantiate the model
    model = FFNNCustom(input_size, hidden_size, output_size)
    mse_loss = MSELoss()

    # Training loop
    learning_rate = 0.01
    num_epochs = 1000

    for iteration in range(num_epochs):
        # Forward pass
        output, cache = model.forward(X)

        # Backward pass
        loss = mse_loss(output, Y)
        grad_output = mse_loss.backward(output, Y)
        grads = model.backward(grad_output, cache)

        # Update weights
        params = model.get_parameters()
        updated_params = sgd_update(params, grads, learning_rate)
        model.set_parameters(updated_params)

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss.item()}")

    # Final evaluation
    final_output, _ = model.forward(X)
    final_loss = mse_loss(final_output, Y)
    print(f"Final Loss: {final_loss.item()}")
    print("FFNNCustom overfitting test passed successfully!")

if DO_TEST:
    test_ffnn_custom_overfitting()
