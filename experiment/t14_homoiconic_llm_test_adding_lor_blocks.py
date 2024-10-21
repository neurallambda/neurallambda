'''

Given a linear layer y = W @ x, what is the right way of adding a new complementary projection y = W @ x + V @ x?

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(0)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)

def test_method(method_name, method, W, V, x):
    y_w = x @ W
    y_v = x @ V
    y = method(W, V, x)

    sim_w = cosine_similarity(y, y_w)
    sim_v = cosine_similarity(y, y_v)

    print(f"{method_name}:")
    print(f"  Similarity with W@x: {sim_w.mean():.4f}")
    print(f"  Similarity with V@x: {sim_v.mean():.4f}")
    print(f"  Norm of y: {torch.norm(y, dim=1).mean():.4f} ± {torch.norm(y, dim=1).std():.4f}")
    print(f"  Mean of y: {y.mean():.4f}")
    print(f"  Variance of y: {y.var():.4f}")
    print(f"  Std of y: {y.std():.4f}")
    print()

def ignore_v(W, V, x):
    return x @ W

def naive(W, V, x):
    return x @ W + x @ V

def scaling(W, V, x, alpha=0.5):
    return x @ W * (1 - alpha) + alpha * (x @ V)

def normalization(W, V, x):
    V_norm = V * (torch.norm(W) / torch.norm(V))
    return x @ W + x @ V_norm

def preserve_stats(W, V, x):
    y_w = x @ W
    y_v = x @ V

    # Calculate means and standard deviations
    mean_w = y_w.mean(dim=0, keepdim=True)
    std_w = y_w.std(dim=0, keepdim=True)
    mean_v = y_v.mean(dim=0, keepdim=True)
    std_v = y_v.std(dim=0, keepdim=True)

    # Normalize both outputs
    y_w_norm = (y_w - mean_w) / std_w
    y_v_norm = (y_v - mean_v) / std_v

    # Combine normalized outputs
    y_combined = (y_w_norm + y_v_norm) / torch.sqrt(torch.tensor(2.0))

    # Scale back to original distribution
    y = y_combined * std_w + mean_w

    return y


def preserve_rms_norm(W, V, x):
    y_w = x @ W
    y_v = x @ V

    # Calculate the RMS norm of the original output
    rms_norm_w = torch.sqrt(torch.mean(y_w**2))

    # Combine the outputs
    y_combined = y_w + y_v

    # Calculate the RMS norm of the combined output
    rms_norm_combined = torch.sqrt(torch.mean(y_combined**2))

    # Scale the combined output to preserve the original RMS norm
    y = y_combined * (rms_norm_w / rms_norm_combined)

    return y


def preserve_layernorm_trained(W, V, x):
    output_dim = W.shape[1]

    optimizer = torch.optim.Adam(layer_norm.parameters(), lr=1e-2)

    # Train LayerNorm to reproduce x@W faithfully
    for _ in tqdm(range(1000)):
        y_w = x @ W
        y = layer_norm(y_w)
        loss = F.mse_loss(y, y_w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Combine with V
    with torch.no_grad():
        y_w = x @ W
        y_v = x @ V
        y_combined = y_w + y_v
        y = layer_norm(y_combined)

    return y


def preserve_rms_norm_trained(W, V, x):
    output_dim = W.shape[1]

    optimizer = torch.optim.Adam(rms_norm.parameters(), lr=1e-2)

    # Train LayerNorm to reproduce x@W faithfully
    for _ in tqdm(range(0)):
        y_w = x @ W
        y = rms_norm(y_w)
        loss = F.mse_loss(y, y_w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Combine with V
    with torch.no_grad():
        y_w = x @ W
        y_v = x @ V
        y_combined = y_w + y_v
        y = rms_norm(y_combined)

    return y


# Initialization of Layernorm experiments:
#   RESULTS: input_dim**0.5 is good init
#   track the mean layer_norm.weight given different in/out dims
#
# i = 128
#   o = 128, ln -> 11.2764
#   o = 256, ln -> 11.25
#   o = 1024, ln -> 11.30
#
# o = 256
#   i = 64, ln -> 7.98
#   i = 128, ln -> 11.25
#   i = 256, ln -> 15.723
#   i = 1024, ln -> 31.84


# Test setup
input_dim = 128
output_dim = 256
batch_size = 1000

W = torch.randn(input_dim, output_dim)
V = torch.randn(input_dim, output_dim)
x = torch.randn(batch_size, input_dim)

# Run tests
test_method("Ignore V", ignore_v, W, V, x)
test_method("Naive", naive, W, V, x)
test_method("Scaling", scaling, W, V, x)
test_method("Normalization", normalization, W, V, x)
test_method("Preserve Stats", preserve_stats, W, V, x)
test_method("Preserve RMS Norm", preserve_rms_norm, W, V, x)

layer_norm = nn.LayerNorm(output_dim)
with torch.no_grad():
    layer_norm.weight[:] = torch.zeros_like(layer_norm.weight) + input_dim ** 0.5
    layer_norm.bias[:] = torch.zeros_like(layer_norm.bias)
test_method("Preserve LayerNorm", preserve_layernorm_trained, W, V, x)
print(layer_norm.weight.mean())

rms_norm = nn.LayerNorm(output_dim)
with torch.no_grad():
    rms_norm.weight[:] = torch.zeros_like(rms_norm.weight) + input_dim ** 0.5
    rms_norm.bias[:] = torch.zeros_like(rms_norm.bias)
test_method("Preserve RMSNorm", preserve_rms_norm_trained, W, V, x)
print(rms_norm.weight.mean())
