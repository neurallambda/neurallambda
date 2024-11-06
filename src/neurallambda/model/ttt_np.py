'''

A numpy library that handles both a "forward" and "backward" pass, similar to pytorch. Unlike pytorch, it calculates gradients by hand.

Each core function in the library is a class, like Linear, that has a forward class function that also returns a cache used by the backward pass, and the backward class function.


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any

import numpy as np

np.random.seed(42)
torch.manual_seed(42)

DO_TEST = False

class Module:
    ''' Create a similar calling syntax as pytorch, but for numpy derived modules '''

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


##########
# MSELoss

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return np.mean((inputs - targets) ** 2)

    def backward(self, inputs, targets):
        n = inputs.size
        return 2 * (inputs - targets) / n


def test_mse_loss():
    for _ in range(10):
        inputs = torch.randn(10, 5, requires_grad=True)
        targets = torch.randn(10, 5)

        # NumPy implementation
        mse_loss = MSELoss()
        inputs_np = inputs.detach().numpy()
        targets_np = targets.numpy()
        loss_np = mse_loss(inputs_np, targets_np)
        grad_np = mse_loss.backward(inputs_np, targets_np)

        # PyTorch implementation
        mse_loss_torch = nn.MSELoss()
        loss_torch = mse_loss_torch(inputs, targets)
        loss_torch.backward()

        # Assert results are close
        assert np.isclose(loss_np, loss_torch.item(), rtol=1e-5, atol=1e-8), \
            f"Loss mismatch: NumPy={loss_np:.6f}, PyTorch={loss_torch.item():.6f}"
        assert np.allclose(grad_np, inputs.grad.numpy(), rtol=1e-5, atol=1e-8), \
            "Gradient mismatch"

    print("MSELoss tests passed successfully!")

if DO_TEST:
    test_mse_loss()


##########
# Linear

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.zeros(out_features) if bias else None

    def forward(self, x):
        out = np.einsum('...i,oi->...o', x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out, {'x': x}

    def backward(self, grad_output, cache):
        x = cache['x']

        grad_input = np.einsum('...o,oi->...i', grad_output, self.weight)

        # # OPTION 1
        # grad_weight = np.einsum('...i,...o->oi', x, grad_output, optimize='optimal')
        # # optimize=optimal helps avoid the error:
        # #   ValueError: output has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions.

        # OPTION 2
        grad_weight = np.einsum('...i,...o->...oi', x, grad_output)
        if grad_weight.ndim > 2:
            # collapse all but last 2 dimensions
            grad_weight = grad_weight.sum(axis=tuple(range(grad_weight.ndim - 2)))

        grad_bias = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1))) if self.bias is not None else None

        return {
            'grad_x': grad_input,
            'grad_weight': grad_weight,
            'grad_bias': grad_bias
        }

    def set_parameters(self, weight, bias):
        self.weight = weight
        self.bias = bias


def test_linear_layer():
    in_features, out_features, batch_size = 5, 3, 10

    # Initialize weight and biases
    weight = np.random.randn(out_features, in_features)
    bias = np.random.randn(out_features)

    # Create inputs
    x_np = np.random.randn(batch_size, in_features)
    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

    # NumPy implementation
    linear_np = Linear(in_features, out_features, bias=True)
    linear_np.set_parameters(weight, bias)
    y_np, cache = linear_np(x_np)

    # PyTorch implementation
    linear_torch = nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear_torch.weight.copy_(torch.tensor(weight))
        linear_torch.bias.copy_(torch.tensor(bias))
    y_torch = linear_torch(x_torch)

    # Forward pass test
    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch"

    # Backward pass
    grad_output_np = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output_np, dtype=torch.float32)

    # NumPy backward
    grads_np = linear_np.backward(grad_output_np, cache)
    grad_x_np = grads_np['grad_x']
    grad_weight_np = grads_np['grad_weight']
    grad_bias_np = grads_np['grad_bias']

    # PyTorch backward
    y_torch.backward(grad_output_torch)

    # Compare gradients
    assert np.allclose(grad_x_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Input gradient mismatch"
    assert grad_weight_np.shape == linear_torch.weight.grad.numpy().shape, f"Shape mismatch: {grad_weight_np.shape=}, {linear_torch.weight.grad.numpy().shape=}"
    assert np.allclose(grad_weight_np, linear_torch.weight.grad.numpy(), rtol=1e-5, atol=1e-8), "Weight gradient mismatch"
    assert np.allclose(grad_bias_np, linear_torch.bias.grad.numpy(), rtol=1e-5, atol=1e-8), "Bias gradient mismatch"

    print("All Linear layer tests passed successfully!")

if DO_TEST:
    test_linear_layer()



class MatMul(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        output = np.einsum('...ij,...jk->...ik', x1, x2)
        cache = {'x1': x1, 'x2': x2}
        return output, cache

    def backward(self, grad_output, cache):
        x1, x2 = cache['x1'], cache['x2']

        # grad_x1 = grad_output * x2^T
        grad_x1 = np.einsum('...ik,...jk->...ij', grad_output, x2)

        # grad_x2 = x1^T * grad_output
        grad_x2 = np.einsum('...ij,...ik->...jk', x1, grad_output)

        return grad_x1, grad_x2

def test_matmul():
    x1_np = np.random.randn(2, 3, 4)
    x2_np = np.random.randn(2, 4, 5)
    x1_torch = torch.tensor(x1_np, requires_grad=True)
    x2_torch = torch.tensor(x2_np, requires_grad=True)

    # Forward pass
    matmul_np = MatMul()
    y_np, cache = matmul_np(x1_np, x2_np)
    y_torch = torch.matmul(x1_torch, x2_torch)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "MatMul forward pass mismatch"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_x1_np, grad_x2_np = matmul_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_x1_np, x1_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "MatMul grad_x1 mismatch"
    assert np.allclose(grad_x2_np, x2_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "MatMul grad_x2 mismatch"

    print("MatMul test passed successfully!")

if DO_TEST:
    test_matmul()


##########
# Activation Functions

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, grad_output, x):
        """
        Compute the gradient of ReLU.

        Args:
        grad_output: The gradient of the loss with respect to the output of ReLU.
        x: The input to the ReLU function in the forward pass.
        """
        return grad_output * (x > 0)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, grad_output, previous_output):
        """
        Compute the gradient of Sigmoid.

        Args:
        grad_output: The gradient of the loss with respect to the output of Sigmoid.
        previous_output: The output of the Sigmoid function in the forward pass.
        """
        return grad_output * previous_output * (1 - previous_output)

class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

    def backward(self, grad_output, previous_output):
        """
        Compute the gradient of Softmax.

        Args:
        grad_output: The gradient of the loss with respect to the output of Softmax.
        previous_output: The output of the Softmax function in the forward pass.
        """
        return previous_output * (grad_output - np.sum(grad_output * previous_output, axis=self.axis, keepdims=True))

def test_relu():
    x_np = np.random.randn(10, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # NumPy implementation
    relu_np = ReLU()
    y_np = relu_np(x_np)

    # PyTorch implementation
    y_torch = F.relu(x_torch)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "ReLU forward pass mismatch"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = relu_np.backward(grad_output, x_np)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "ReLU backward pass mismatch"

    print("ReLU test passed successfully!")

def test_sigmoid():
    np.random.seed(43)
    torch.manual_seed(43)

    x_np = np.random.randn(10, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # NumPy implementation
    sigmoid_np = Sigmoid()
    y_np = sigmoid_np(x_np)

    # PyTorch implementation
    y_torch = torch.sigmoid(x_torch)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Sigmoid forward pass mismatch"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = sigmoid_np.backward(grad_output, y_np)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Sigmoid backward pass mismatch"

    print("Sigmoid test passed successfully!")

def test_softmax():
    np.random.seed(44)
    torch.manual_seed(44)

    x_np = np.random.randn(10, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # NumPy implementation
    softmax_np = Softmax()
    y_np = softmax_np(x_np)

    # PyTorch implementation
    y_torch = F.softmax(x_torch, dim=-1)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Softmax forward pass mismatch"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = softmax_np.backward(grad_output, y_np)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Softmax backward pass mismatch"

    print("Softmax test passed successfully!")

if DO_TEST:
    test_relu()
    test_sigmoid()
    test_softmax()


##########
# Simple FFNN

class FFNNNumpy(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = Linear(input_size, hidden_size, bias=True)
        self.sigmoid = ReLU()
        self.linear2 = Linear(hidden_size, output_size, bias=True)
        self.mse_loss = MSELoss()

    def forward(self, x):
        l1_output, linear1_cache = self.linear1(x)
        s_output = self.sigmoid(l1_output)
        l2_output, linear2_cache = self.linear2(s_output)
        cache = {
            'linear1_cache': linear1_cache,
            's_output': s_output,
            'linear2_cache': linear2_cache,
            'l2_output': l2_output
        }
        return l2_output, cache

    def backward(self, targets, cache):
        # Extract the cached values
        linear1_cache = cache['linear1_cache']
        s_output = cache['s_output']
        linear2_cache = cache['linear2_cache']
        l2_output = cache['l2_output']

        # Compute the gradient of the loss with respect to the output
        grad_loss = self.mse_loss.backward(l2_output, targets)

        # Backprop through the second linear layer
        grads_l2 = self.linear2.backward(grad_loss, linear2_cache)
        grad_s_output = grads_l2['grad_x']  # Gradient w.r.t the output of the sigmoid

        # Backprop through the sigmoid layer
        grad_l1_output = self.sigmoid.backward(grad_s_output, s_output)

        # Backprop through the first linear layer
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
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x

def test_integrated_model():
    input_size, hidden_size, output_size = 10, 5, 1
    batch_size = 32

    # Initialize models
    model_np = FFNNNumpy(input_size, hidden_size, output_size)
    model_torch = FFNNPyTorch(input_size, hidden_size, output_size)

    # Generate random parameters
    params = {
        'linear1.weight': np.random.randn(hidden_size, input_size),
        'linear1.bias': np.random.randn(hidden_size),
        'linear2.weight': np.random.randn(output_size, hidden_size),
        'linear2.bias': np.random.randn(output_size)
    }

    # Set parameters for both models
    model_np.set_parameters(params)
    with torch.no_grad():
        for name, param in model_torch.named_parameters():
            param.copy_(torch.tensor(params[name]))

    # Generate input data
    x_np = np.random.randn(batch_size, input_size)
    x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)

    # Forward pass
    y_np, ffnn_cache = model_np(x_np)
    y_torch = model_torch(x_torch)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch"

    # Generate target data
    target_np = np.random.randn(batch_size, output_size)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Compute loss
    loss_np = model_np.mse_loss(y_np, target_np)
    loss_torch = nn.MSELoss()(y_torch, target_torch)

    assert np.allclose(loss_np, loss_torch.item(), rtol=1e-5, atol=1e-8), "Loss mismatch"

    # Backward pass
    loss_torch.backward()
    grads_np = model_np.backward(target_np, ffnn_cache)

    # Mapping between NumPy and PyTorch parameter names
    param_mapping = {
        'linear1.weight': 'linear1.weight',
        'linear1.bias': 'linear1.bias',
        'linear2.weight': 'linear2.weight',
        'linear2.bias': 'linear2.bias'
    }

    # Compare gradients
    for np_name, torch_name in param_mapping.items():
        np_grad = grads_np[np_name]
        torch_grad = getattr(model_torch, torch_name.split('.')[0]).weight.grad.numpy() if 'weight' in torch_name else getattr(model_torch, torch_name.split('.')[0]).bias.grad.numpy()

        assert np.allclose(np_grad, torch_grad, rtol=1e-5, atol=1e-8), f"Gradient mismatch for {np_name}"

    print("All integrated model tests passed successfully!")

if DO_TEST:
    test_integrated_model()


##########
# Self Attention

class Transpose(Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return np.transpose(x, self.axes)

    def backward(self, grad_output):
        # Reverse the transposition for backpropagation
        reverse_axes = np.argsort(self.axes)
        return np.transpose(grad_output, reverse_axes)


class View(Module):
    def __init__(self, from_shape, to_shape):
        super().__init__()
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x):
        return np.reshape(x, self.to_shape)

    def backward(self, grad_output):
        return np.reshape(grad_output, self.from_shape)


class Reshape(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, shape):
        cache = {
            'from_shape': x.shape
        }
        return np.reshape(x, shape), cache

    def backward(self, grad_output, cache):
        return np.reshape(grad_output, cache['from_shape'])


def test_view():
    x_np = np.random.randn(2, 3, 4)
    x_torch = torch.tensor(x_np, requires_grad=True)

    view_np = View((2, 3, 4), (2, 12))
    y_np = view_np(x_np)
    y_torch = x_torch.view(2, 12)

    assert np.allclose(y_np, y_torch.detach().numpy())

    grad_output_np = np.random.randn(2, 12)
    grad_output_torch = torch.tensor(grad_output_np)

    grad_input_np = view_np.backward(grad_output_np)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy())
    print('View tests passed')

def test_transpose():
    x_np = np.random.randn(2, 3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    transpose_np = Transpose((0, 2, 1, 3))
    y_np = transpose_np(x_np)
    y_torch = x_torch.permute(0, 2, 1, 3)

    assert np.allclose(y_np, y_torch.detach().numpy())

    grad_output_np = np.random.randn(2, 4, 3, 5)
    grad_output_torch = torch.tensor(grad_output_np)

    grad_input_np = transpose_np.backward(grad_output_np)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy())
    print('Transpose tests passed')

def test_reshape():
    x_np = np.random.randn(2, 3, 4)
    x_torch = torch.tensor(x_np, requires_grad=True)

    reshape_np = Reshape()
    y_np, reshape_cache = reshape_np(x_np, (2, 12))
    y_torch = x_torch.reshape(2, 12)
    assert np.allclose(y_np, y_torch.detach().numpy())

    grad_output_np = np.random.randn(2, 12)
    grad_output_torch = torch.tensor(grad_output_np)

    grad_input_np = reshape_np.backward(grad_output_np, reshape_cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy())
    print('Reshape tests passed')

if DO_TEST:
    test_view()
    test_transpose()
    test_reshape()


class SelfAttention(Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = np.random.randn(3 * embed_dim, embed_dim)
        self.in_proj_bias = np.random.randn(3 * embed_dim) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        self.batch_first = batch_first

        self.reshape = Reshape()
        self.softmax = Softmax(axis=-1)

        self.transpose_1_0_2 = Transpose((1, 0, 2))
        self.transpose_0_2_1 = Transpose((0, 2, 1))

        self.matmul = MatMul()

    def forward(self, query, key, value, need_weight=True, attn_mask=None):
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

        if query is key and key is value:
            qkv_proj, qkv_proj_cache = self.matmul(query, w.T)
            cache['qkv_proj_cache'] = qkv_proj_cache
            if b is not None:
                qkv_proj += b
            q, k, v = np.split(qkv_proj, 3, axis=-1)
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
        scores = scores / np.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_output_weight = self.softmax(scores)
        attn_output, attn_output_cache = self.matmul(attn_output_weight, v)
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
            'attn_output_weight': attn_output_weight,
        })

        if need_weight:
            attn_output_weight = attn_output_weight.reshape(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weight, cache
        else:
            return attn_output, None, cache


    def backward(self, grad_output, cache):
        if self.batch_first:
            grad_output = self.transpose_1_0_2(grad_output)

        # Backward through out_proj
        # grad_attn_output, out_proj_grads = self.out_proj.backward(grad_output, cache['out_proj_cache'])
        # grad_out_proj_weight, grad_out_proj_bias = out_proj_grads['grad_weight'], out_proj_grads['grad_bias']
        out_proj_grads = self.out_proj.backward(grad_output, cache['out_proj_cache'])
        grad_attn_output = out_proj_grads['grad_x']
        grad_out_proj_weight, grad_out_proj_bias = out_proj_grads['grad_weight'], out_proj_grads['grad_bias']

        # Reshape grad_attn_output
        grad_attn_output = self.reshape.backward(grad_attn_output, cache['attn_output_reshape_cache'])
        grad_attn_output = self.transpose_1_0_2(grad_attn_output)

        # Backward through attention mechanism
        grad_attn_output_weight, grad_v = self.matmul.backward(grad_attn_output, cache['attn_output_cache'])

        # Backward through softmax
        grad_scores = self.softmax.backward(grad_attn_output_weight, cache['attn_output_weight'])

        # Scale gradient
        grad_scores = grad_scores / np.sqrt(self.head_dim)

        # Backward through matmul for scores
        grad_q, grad_k_t = self.matmul.backward(grad_scores, cache['scores_cache'])
        grad_k = self.transpose_0_2_1(grad_k_t)

        # Reshape gradients
        grad_q = self.transpose_1_0_2(grad_q)
        grad_k = self.transpose_1_0_2(grad_k)
        grad_v = self.transpose_1_0_2(grad_v)

        grad_q = self.reshape.backward(grad_q, cache['q_reshape_cache'])
        grad_k = self.reshape.backward(grad_k, cache['k_reshape_cache'])
        grad_v = self.reshape.backward(grad_v, cache['v_reshape_cache'])

        # Backward through linear projections
        if 'qkv_proj_cache' in cache:
            # Case: query is key and key is value
            grad_qkv_proj = np.concatenate([grad_q, grad_k, grad_v], axis=-1)
            if self.in_proj_bias is not None:
                grad_in_proj_bias = np.sum(grad_qkv_proj, axis=(0, 1))
            grad_query, grad_in_proj_weight = self.matmul.backward(grad_qkv_proj, cache['qkv_proj_cache'])
            grad_in_proj_weight = np.sum(grad_in_proj_weight, axis=0).T  # Sum over batch, then transpose
        else:
            # Case: query, key, and value are different
            grad_query, grad_wq = self.matmul.backward(grad_q, cache['q_proj_cache'])
            grad_key, grad_wk = self.matmul.backward(grad_k, cache['k_proj_cache'])
            grad_value, grad_wv = self.matmul.backward(grad_v, cache['v_proj_cache'])
            grad_in_proj_weight = np.concatenate([
                np.sum(grad_wq, axis=0).T,
                np.sum(grad_wk, axis=0).T,
                np.sum(grad_wv, axis=0).T
            ], axis=0)  # Sum over batch for each, transpose, then concatenate
            if self.in_proj_bias is not None:
                grad_in_proj_bias = np.concatenate([
                    np.sum(grad_q, axis=(0, 1)),
                    np.sum(grad_k, axis=(0, 1)),
                    np.sum(grad_v, axis=(0, 1))
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


##################################################

def test_multihead_attention_equivalence():
    # Parameters
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_length = 10

    # Initialize numpy implementation
    np_self_attention = SelfAttention(embed_dim, num_heads, bias=True, batch_first=True)

    # Initialize PyTorch implementation
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, bias=True, batch_first=True)

    # Manually set weight and biases
    in_proj_weight = np.random.randn(3 * embed_dim, embed_dim)
    in_proj_bias = np.random.randn(3 * embed_dim)
    out_proj_weight = np.random.randn(embed_dim, embed_dim)
    out_proj_bias = np.random.randn(embed_dim)

    # Set weight for numpy implementation
    np_self_attention.in_proj_weight = in_proj_weight
    np_self_attention.in_proj_bias = in_proj_bias
    np_self_attention.out_proj.weight = out_proj_weight
    np_self_attention.out_proj.bias = out_proj_bias

    # Set weight for PyTorch implementation
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(torch.FloatTensor(in_proj_weight))
        torch_mha.in_proj_bias.copy_(torch.FloatTensor(in_proj_bias))
        torch_mha.out_proj.weight.copy_(torch.FloatTensor(out_proj_weight))
        torch_mha.out_proj.bias.copy_(torch.FloatTensor(out_proj_bias))

    # Create input tensors
    np_input = np.random.randn(batch_size, seq_length, embed_dim)
    torch_input = torch.FloatTensor(np_input)

    # Forward pass for numpy implementation
    np_output, np_attn, cache = np_self_attention(np_input, np_input, np_input)  #, average_attn_weights=False, need_weights=True)

    # Forward pass for PyTorch implementation
    torch_output, torch_attn = torch_mha(torch_input, torch_input, torch_input, average_attn_weights=False, need_weights=True)

    # Convert PyTorch outputs to numpy for comparison
    torch_output_np = torch_output.detach().numpy()
    torch_attn_np = torch_attn.detach().numpy()

    # Compare outputs
    output_close = np.allclose(np_output, torch_output_np, rtol=1e-3, atol=1e-3)
    attn_close = np.allclose(np_attn, torch_attn_np, rtol=1e-4, atol=1e-4)

    print(f"Outputs close: {output_close}")
    print(f"Attention weight close: {attn_close}")

    print(f'{torch_output[0,0,:3]=}')
    print(f'{np_output[0,0,:3]=}')

    assert output_close and attn_close, "Outputs or attention weight do not match"

    print("Test passed: numpy and PyTorch implementations produce equivalent results.")


def test_self_attention_gradients():
    # Parameters
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_length = 10

    # Initialize numpy implementation
    np_self_attention = SelfAttention(embed_dim, num_heads, bias=True, batch_first=True)

    # Initialize PyTorch implementation
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, bias=True, batch_first=True)

    # Manually set weights and biases with high precision
    in_proj_weight = np.random.randn(3 * embed_dim, embed_dim).astype(np.float64)
    in_proj_bias = np.random.randn(3 * embed_dim).astype(np.float64)
    out_proj_weight = np.random.randn(embed_dim, embed_dim).astype(np.float64)
    out_proj_bias = np.random.randn(embed_dim).astype(np.float64)

    # Set weights for numpy implementation
    np_self_attention.in_proj_weight = in_proj_weight
    np_self_attention.in_proj_bias = in_proj_bias
    np_self_attention.out_proj.weight = out_proj_weight
    np_self_attention.out_proj.bias = out_proj_bias

    # Set weights for PyTorch implementation
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(torch.DoubleTensor(in_proj_weight))
        torch_mha.in_proj_bias.copy_(torch.DoubleTensor(in_proj_bias))
        torch_mha.out_proj.weight.copy_(torch.DoubleTensor(out_proj_weight))
        torch_mha.out_proj.bias.copy_(torch.DoubleTensor(out_proj_bias))
    torch_mha.to(dtype=torch.float64)

    # Create input tensors with high precision
    np_input = np.random.randn(batch_size, seq_length, embed_dim).astype(np.float64)
    torch_input = torch.DoubleTensor(np_input).requires_grad_()

    # Forward pass for numpy implementation
    np_output, _, cache = np_self_attention(np_input, np_input, np_input)

    # Forward pass for PyTorch implementation
    torch_output, _ = torch_mha(torch_input, torch_input, torch_input)

    # Create gradient for output with high precision
    output_grad = np.random.randn(*np_output.shape).astype(np.float64)
    torch_output_grad = torch.DoubleTensor(output_grad)

    # Backward pass for numpy implementation
    grad_query, grad_key, grad_value, np_grads = np_self_attention.backward(output_grad, cache)

    # Backward pass for PyTorch implementation
    torch_output.backward(torch_output_grad)

    # Compare gradients
    rtol, atol = 1e-3, 1e-3

    # Check shapes
    assert grad_query.shape == torch_input.grad.shape, "Input gradient shapes do not match"
    assert np_grads['in_proj_weight'].shape == torch_mha.in_proj_weight.grad.shape, f"in_proj_weight gradient shapes do not match: {np_grads['in_proj_weight'].shape=}, {torch_mha.in_proj_weight.grad.shape=}"
    assert np_grads['in_proj_bias'].shape == torch_mha.in_proj_bias.grad.shape, "in_proj_bias gradient shapes do not match"
    assert np_grads['out_proj.weight'].shape == torch_mha.out_proj.weight.grad.shape, f"out_proj.weight gradient shapes do not match: {np_grads['out_proj.weight'].shape=}, {torch_mha.out_proj.weight.grad.shape=}"
    assert np_grads['out_proj.bias'].shape == torch_mha.out_proj.bias.grad.shape, "out_proj.bias gradient shapes do not match"

    # Sanity check gradients
    grad_input = grad_query + grad_key + grad_value
    print('Input grad diff:', (grad_input - torch_input.grad.numpy()).max())
    print('in_proj_weight diff:', (np_grads['in_proj_weight'] - torch_mha.in_proj_weight.grad.numpy()).max())
    print('in_proj_bias diff:', (np_grads['in_proj_bias'] - torch_mha.in_proj_bias.grad.numpy()).max())
    print('out_proj.weight diff:', (np_grads['out_proj.weight'] - torch_mha.out_proj.weight.grad.numpy()).max())
    print('out_proj.bias diff:', (np_grads['out_proj.bias'] - torch_mha.out_proj.bias.grad.numpy()).max())

    # Check input gradients
    assert np.allclose(grad_input, torch_input.grad.numpy(), rtol=rtol, atol=atol), "Input gradients do not match"

    # Check weight gradients
    assert np.allclose(np_grads['in_proj_weight'], torch_mha.in_proj_weight.grad.numpy(), rtol=rtol, atol=atol), "in_proj_weight gradients do not match"
    assert np.allclose(np_grads['in_proj_bias'], torch_mha.in_proj_bias.grad.numpy(), rtol=rtol, atol=atol), "in_proj_bias gradients do not match"
    assert np.allclose(np_grads['out_proj.weight'], torch_mha.out_proj.weight.grad.numpy(), rtol=rtol, atol=atol), "out_proj.weight gradients do not match"
    assert np.allclose(np_grads['out_proj.bias'], torch_mha.out_proj.bias.grad.numpy(), rtol=rtol, atol=atol), "out_proj.bias gradients do not match"

    print("All gradient checks passed!")

if DO_TEST:
    test_multihead_attention_equivalence()

    test_self_attention_gradients()


##################################################
# LayerNorm

class Sum(Module):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return np.sum(x, axis=self.axis, keepdims=self.keepdims), {'input_shape': x.shape}

    def backward(self, grad_output, cache):
        input_shape = cache['input_shape']
        if not self.keepdims:
            if self.axis is not None:
                # Expand dimensions for proper broadcasting
                expand_dims = [slice(None)] * len(input_shape)
                if isinstance(self.axis, int):
                    expand_dims[self.axis] = np.newaxis
                else:
                    for ax in self.axis:
                        expand_dims[ax] = np.newaxis
                grad_output = grad_output[tuple(expand_dims)]
            else:
                grad_output = np.full(input_shape, grad_output)
        return np.broadcast_to(grad_output, input_shape)

def test_sum():
    # Test case 1: Sum along axis 0
    x_np = np.random.randn(3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    sum_np = Sum(axis=0)
    y_np, cache = sum_np(x_np)

    y_torch = torch.sum(x_torch, dim=0)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for axis 0"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = sum_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for axis 0"

    # Test case 2: Sum all elements
    x_np = np.random.randn(3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    sum_np = Sum()
    y_np, cache = sum_np(x_np)

    y_torch = torch.sum(x_torch)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for sum all"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = sum_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for sum all"

    # Test case 3: Sum along multiple axes with keepdims=True
    x_np = np.random.randn(3, 4, 5, 6)
    x_torch = torch.tensor(x_np, requires_grad=True)

    sum_np = Sum(axis=(1, 2), keepdims=True)
    y_np, cache = sum_np(x_np)

    y_torch = torch.sum(x_torch, dim=(1, 2), keepdim=True)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for multiple axes"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = sum_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for multiple axes"

    print("All Sum tests passed successfully!")

if DO_TEST:
    test_sum()



class Mean(Module):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        result = np.mean(x, axis=self.axis, keepdims=self.keepdims)
        cache = {'input_shape': x.shape, 'output_shape': result.shape}
        return result, cache

    def backward(self, grad_output, cache):
        input_shape = cache['input_shape']
        output_shape = cache['output_shape']

        # Reshape grad_output if dimensions were reduced and not kept
        if not self.keepdims:
            if self.axis is not None:
                # Expand dimensions for proper broadcasting
                expand_dims = [slice(None)] * len(input_shape)
                if isinstance(self.axis, int):
                    expand_dims[self.axis] = np.newaxis
                else:
                    for ax in self.axis:
                        expand_dims[ax] = np.newaxis
                grad_output = grad_output[tuple(expand_dims)]
            else:
                grad_output = np.full(input_shape, grad_output)

        # Calculate the number of elements that were averaged
        if self.axis is None:
            num_elements = np.prod(input_shape)
        else:
            num_elements = np.prod([input_shape[ax] for ax in np.atleast_1d(self.axis)])

        # Distribute the gradient
        return np.broadcast_to(grad_output, input_shape) / num_elements

def test_mean():
    # Test case 1: Mean along axis 0
    x_np = np.random.randn(3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    mean_np = Mean(axis=0)
    y_np, cache = mean_np(x_np)

    y_torch = torch.mean(x_torch, dim=0)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for axis 0"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = mean_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for axis 0"

    # Test case 2: Mean of all elements
    x_np = np.random.randn(3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    mean_np = Mean()
    y_np, cache = mean_np(x_np)

    y_torch = torch.mean(x_torch)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for mean all"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = mean_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for mean all"

    # Test case 3: Mean along multiple axes with keepdims=True
    x_np = np.random.randn(3, 4, 5, 6)
    x_torch = torch.tensor(x_np, requires_grad=True)

    mean_np = Mean(axis=(1, 2), keepdims=True)
    y_np, cache = mean_np(x_np)

    y_torch = torch.mean(x_torch, dim=(1, 2), keepdim=True)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for multiple axes"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = mean_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for multiple axes"

    print("All Mean tests passed successfully!")

if DO_TEST:
    test_mean()




class Var(Module):
    def __init__(self, axis=None, keepdims=False, correction=0):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self.correction = correction  # Bessel's correction (0 for population variance, 1 for sample variance)

    def forward(self, x):
        mean = np.mean(x, axis=self.axis, keepdims=True)
        squared_diff = (x - mean) ** 2
        var = np.sum(squared_diff, axis=self.axis, keepdims=self.keepdims)

        if self.axis is None:
            n = x.size
        else:
            n = np.prod([x.shape[ax] for ax in np.atleast_1d(self.axis)])

        var /= (n - self.correction)

        cache = {
            'input_shape': x.shape,
            'output_shape': var.shape,
            'mean': mean,
            'x': x,
            'n': n
        }
        return var, cache

    def backward(self, grad_output, cache):
        x = cache['x']
        mean = cache['mean']
        input_shape = cache['input_shape']
        n = cache['n']

        if not self.keepdims:
            if self.axis is not None:
                # Expand dimensions for proper broadcasting
                expand_dims = [slice(None)] * len(input_shape)
                if isinstance(self.axis, int):
                    expand_dims[self.axis] = np.newaxis
                else:
                    for ax in self.axis:
                        expand_dims[ax] = np.newaxis
                grad_output = grad_output[tuple(expand_dims)]
            else:
                grad_output = np.full(input_shape, grad_output)

        grad_x = 2 * (x - mean) * grad_output / (n - self.correction)
        return grad_x

def test_var():
    # Test case 1: Variance along axis 0
    x_np = np.random.randn(3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    var_np = Var(axis=0)
    y_np, cache = var_np(x_np)

    y_torch = torch.var(x_torch, dim=0, unbiased=False)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for axis 0"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = var_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for axis 0"

    # Test case 2: Variance of all elements
    x_np = np.random.randn(3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)

    var_np = Var()
    y_np, cache = var_np(x_np)

    y_torch = torch.var(x_torch, unbiased=False)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for var all"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = var_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for var all"

    # Test case 3: Variance along multiple axes with keepdims=True and correction=1 (sample variance)
    x_np = np.random.randn(3, 4, 5, 6)
    x_torch = torch.tensor(x_np, requires_grad=True)

    var_np = Var(axis=(1, 2), keepdims=True, correction=1)
    y_np, cache = var_np(x_np)

    y_torch = torch.var(x_torch, dim=(1, 2), keepdim=True, unbiased=True)

    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch for multiple axes"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np = var_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for multiple axes"

    print("All Var tests passed successfully!")

if DO_TEST:
    test_var()



class Div(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        result = np.divide(x, y)
        cache = {'x': x, 'y': y}
        return result, cache

    def backward(self, grad_output, cache):
        x, y = cache['x'], cache['y']
        grad_x = grad_output / y
        grad_y = -grad_output * x / (y ** 2)
        return grad_x, grad_y

def test_div():
    x_np = np.random.randn(3, 4, 5)
    y_np = np.random.randn(3, 4, 5)
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)

    div_np = Div()
    z_np, cache = div_np(x_np, y_np)

    z_torch = torch.div(x_torch, y_torch)

    assert np.allclose(z_np, z_torch.detach().numpy(), rtol=1e-5, atol=1e-8), "Forward pass mismatch"

    # Backward pass
    grad_output = np.random.randn(*z_np.shape)
    grad_output_torch = torch.tensor(grad_output)

    grad_x_np, grad_y_np = div_np.backward(grad_output, cache)
    z_torch.backward(grad_output_torch)

    assert np.allclose(grad_x_np, x_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for x"
    assert np.allclose(grad_y_np, y_torch.grad.numpy(), rtol=1e-5, atol=1e-8), "Backward pass mismatch for y"

    print("Div test passed successfully!")

if DO_TEST:
    test_div()


##################################################
# LayerNorm

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = np.ones(normalized_shape)
            self.bias = np.zeros(normalized_shape)

    def forward(self, x):
        # Calculate mean and variance
        mean = Mean(axis=-1, keepdims=True)
        var = Var(axis=-1, keepdims=True)

        mean_x, mean_cache = mean(x)
        var_x, var_cache = var(x)

        # Normalize
        x_norm = (x - mean_x) / np.sqrt(var_x + self.eps)

        if self.elementwise_affine:
            y = self.weight * x_norm + self.bias
        else:
            y = x_norm

        cache = {
            'x': x,
            'mean': mean_x,
            'var': var_x,
            'x_norm': x_norm,
            'mean_cache': mean_cache,
            'var_cache': var_cache
        }

        return y, cache

    def backward(self, grad_output, cache):
        x = cache['x']
        mean = cache['mean']
        var = cache['var']
        x_norm = cache['x_norm']
        mean_cache = cache['mean_cache']
        var_cache = cache['var_cache']

        if self.elementwise_affine:
            grad_weight = np.sum(grad_output * x_norm, axis=tuple(range(len(x.shape) - len(self.normalized_shape))))
            grad_bias = np.sum(grad_output, axis=tuple(range(len(x.shape) - len(self.normalized_shape))))
            grad_x_norm = grad_output * self.weight
        else:
            grad_x_norm = grad_output

        # Backward pass for x_norm
        grad_var = -0.5 * np.sum(grad_x_norm * (x - mean) * (var + self.eps)**(-1.5), axis=-1, keepdims=True)
        grad_mean = -np.sum(grad_x_norm / np.sqrt(var + self.eps), axis=-1, keepdims=True)
        grad_x = grad_x_norm / np.sqrt(var + self.eps) + \
                 2 * grad_var * (x - mean) / x.shape[-1] + \
                 grad_mean / x.shape[-1]

        # Backward pass for mean and var
        mean_module = Mean(axis=-1, keepdims=True)
        var_module = Var(axis=-1, keepdims=True)
        grad_x += mean_module.backward(grad_mean, mean_cache)
        grad_x += var_module.backward(grad_var, var_cache)

        if self.elementwise_affine:
            return grad_x, {'weight': grad_weight, 'bias': grad_bias}
        else:
            return grad_x, {}

def test_layer_norm():
    # Test parameters
    batch_size, seq_len, hidden_size = 2, 3, 4
    eps = 1e-5

    # Create input tensor
    x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # Initialize LayerNorm
    ln_np = LayerNorm(hidden_size, eps=eps)
    ln_torch = nn.LayerNorm(hidden_size, eps=eps)

    # Set the same weights and biases for both implementations
    ln_torch.weight.data = torch.tensor(ln_np.weight)
    ln_torch.bias.data = torch.tensor(ln_np.bias)
    ln_torch.to(dtype=torch.float32)

    # Forward pass
    y_np, cache = ln_np(x_np)
    y_torch = ln_torch(x_torch)

    # Check forward pass
    assert np.allclose(y_np, y_torch.detach().numpy(), rtol=1e-4, atol=1e-4), \
        f"Forward pass mismatch. First few elements: np={y_np.flatten()[:5]}, torch={y_torch.detach().numpy().flatten()[:5]}"

    # Backward pass
    grad_output = np.random.randn(*y_np.shape).astype(np.float32)
    grad_output_torch = torch.tensor(grad_output)

    grad_input_np, grad_params_np = ln_np.backward(grad_output, cache)
    y_torch.backward(grad_output_torch)

    # Check input gradients
    assert np.allclose(grad_input_np, x_torch.grad.numpy(), rtol=1e-4, atol=1e-4), \
        f"Input gradient mismatch. First few elements: np={grad_input_np.flatten()[:5]}, torch={x_torch.grad.numpy().flatten()[:5]}"

    # Check weight gradients
    assert np.allclose(grad_params_np['weight'], ln_torch.weight.grad.numpy(), rtol=1e-4, atol=1e-4), \
        f"Weight gradient mismatch. First few elements: np={grad_params_np['weight'].flatten()[:5]}, torch={ln_torch.weight.grad.numpy().flatten()[:5]}"

    # Check bias gradients
    assert np.allclose(grad_params_np['bias'], ln_torch.bias.grad.numpy(), rtol=1e-4, atol=1e-4), \
        f"Bias gradient mismatch. First few elements: np={grad_params_np['bias'].flatten()[:5]}, torch={ln_torch.bias.grad.numpy().flatten()[:5]}"

    print("All LayerNorm tests passed successfully!")

if DO_TEST:
    test_layer_norm()


##########

if DO_TEST:
    print('ALL DONE')






##################################################
# Sandbox

def sgd_update(params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], learning_rate: float) -> Dict[str, np.ndarray]:
    """Simple SGD update"""
    return {k: v - learning_rate * grads[k] for k, v in params.items()}

def test_ffnn_numpy_overfitting():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate training data
    input_size = 32
    hidden_size = 64
    output_size = 10

    num_samples = 1
    X = np.random.randn(num_samples, input_size)
    Y = np.random.randn(num_samples, output_size)

    # Instantiate the model
    model = FFNNNumpy(input_size, hidden_size, output_size)

    # Training loop
    learning_rate = 0.01
    num_epochs = 1000

    for iteration in range(num_epochs):
        # Forward pass
        output, cache = model.forward(X)

        # Compute loss
        loss = model.mse_loss(output, Y)

        # Backward pass
        grad_output = model.mse_loss.backward(output, Y)
        grads = model.backward(grad_output, cache)

        # Update weights
        params = model.get_parameters()
        updated_params = sgd_update(params, grads, learning_rate)
        model.set_parameters(updated_params)

        print(f"Iteration {iteration + 1}, Loss: {loss}")

    # Final evaluation
    final_output, _ = model.forward(X)
    final_loss = model.mse_loss(final_output, Y)
    print(f"Final Loss: {final_loss}")

    print("FFNNNumpy overfitting test passed successfully!")

# Run the test
test_ffnn_numpy_overfitting()
