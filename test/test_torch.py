'''

Tests for neurallambda.torch

'''

from neurallambda.torch import NuLinear, Choice, roll_without_wrap
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################
# roll_without_wrap

def test_roll_without_wrap():
    # Test case 1: No shift
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    shift = 0
    expected_output = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    assert torch.equal(roll_without_wrap(tensor, shift), expected_output)

    # Test case 2: Positive shift
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    shift = 1
    expected_output = torch.tensor([[[0, 0, 0], [1, 2, 3], [4, 5, 6]]])
    assert torch.equal(roll_without_wrap(tensor, shift), expected_output)

    # Test case 3: Negative shift
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    shift = -1
    expected_output = torch.tensor([[[4, 5, 6], [7, 8, 9], [0, 0, 0]]])
    assert torch.equal(roll_without_wrap(tensor, shift), expected_output)

    # Test case 4: Shift greater than sequence length
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    shift = 4
    expected_output = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert torch.equal(roll_without_wrap(tensor, shift), expected_output)

    # Test case 5: Shift less than negative sequence length
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    shift = -4
    expected_output = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert torch.equal(roll_without_wrap(tensor, shift), expected_output)

    # Test case 6: Multiple batches
    tensor = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    ])
    shift = 1
    expected_output = torch.tensor([
        [[0, 0, 0], [1, 2, 3], [4, 5, 6]],
        [[0, 0, 0], [10, 11, 12], [13, 14, 15]]
    ])
    assert torch.equal(roll_without_wrap(tensor, shift), expected_output)

    # Test case 7: Custom fill value
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    shift = 1
    fill_value = -1
    expected_output = torch.tensor([[[-1, -1, -1], [1, 2, 3], [4, 5, 6]]])
    assert torch.equal(roll_without_wrap(tensor, shift, fill_value), expected_output)


##################################################
# NuLinear

####################
# Test Shapes

# Initialize parameters
in_features = 10
out_features = 5
batch_size = 3
init_extra_dim = 2
fwd_extra_dim = 4

# Create dummy inputs
input = torch.randn(batch_size, in_features)
init_extra_weight = torch.randn(init_extra_dim, in_features)
fwd_extra_weight = torch.randn(batch_size, fwd_extra_dim, in_features)

# Create NuLinear instances and perform assertions
# Test without any extra weight
model = NuLinear(in_features, out_features, bias=True)
assert model(input).shape == (batch_size, out_features), "Output shape mismatch without extra weight"

# Test with initial extra weight
model_with_init_extra = NuLinear(in_features, out_features, init_extra_weight=init_extra_weight, bias=True)
assert model_with_init_extra(input).shape == (batch_size, out_features + init_extra_dim), "Output shape mismatch with init extra weight"

# Test with forward extra weight
model_with_fwd_extra = NuLinear(in_features, out_features, fwd_extra_dim=fwd_extra_dim, bias=True)
output_with_fwd_extra = model_with_fwd_extra(input, extra_weight=fwd_extra_weight)
assert output_with_fwd_extra.shape == (batch_size, out_features + fwd_extra_dim), "Output shape mismatch with forward extra weight"

# Test with both init and forward extra weight
model_with_both_extra = NuLinear(in_features, out_features, init_extra_weight=init_extra_weight, fwd_extra_dim=fwd_extra_dim, bias=True)
output_with_both_extra = model_with_both_extra(input, extra_weight=fwd_extra_weight)
assert output_with_both_extra.shape == (batch_size, out_features + init_extra_dim + fwd_extra_dim), "Output shape mismatch with both types of extra weight"

####################
# Test Values

# Initialize a simple scenario for predictable outcomes
in_features = 2
out_features = 3
batch_size = 5
init_extra_dim = 7
fwd_extra_dim = 3

# Create simple, distinct inputs for each batch item
input = torch.tensor([[1.0, 2.0],
                      [2.0, 1.0],
                      [10.0, 0.0],
                      [100.0, 0.0],
                      [1000.0, 0.0],
                      ])

##########
# Basic Linear

w = torch.arange(3. * 2).reshape(3, 2)
b = torch.arange(3.)

model = NuLinear(in_features, out_features, bias=True, normalize_input=False, normalize_weight=False)
model.weight.data[:] = w
model.bias.data[:] = b

trg_model = nn.Linear(in_features, out_features, bias=True)
trg_model.weight.data[:] = w
trg_model.bias.data[:] = b

# Calculate expected output manually for basic scenario
target = trg_model(input)
output = model(input)
assert torch.allclose(output, target, atol=1e-6), f"failed.\n\nexpected_output={expected_output}\n\noutput={output}"

##########
# Static weights

w = torch.arange(3. * 2).reshape(3, 2)
b = torch.arange(3.)

iw = torch.arange(init_extra_dim * 2).reshape(init_extra_dim, 2)
ib = torch.zeros(init_extra_dim)

model = NuLinear(in_features, out_features, bias=True, init_extra_weight=iw, normalize_input=False, normalize_weight=False)
model.weight.data[:] = w
model.bias.data[:] = b

trg_model = nn.Linear(in_features, out_features + init_extra_dim, bias=True)
trg_model.weight.data[:3] = w
trg_model.weight.data[3:] = iw
trg_model.bias.data[:3] = b
trg_model.bias.data[3:] = ib

# Calculate expected output manually for basic scenario
target = trg_model(input)
output = model(input)
assert torch.allclose(output, target, atol=1e-6), f"failed.\n\nexpected_output={expected_output}\n\noutput={output}"


##########
# Forward extra weights scenario

w = torch.arange(3. * 2).reshape(3, 2)
b = torch.arange(3.)

fwd_extra_weight = torch.randn(batch_size, fwd_extra_dim, 2)

model = NuLinear(in_features, out_features, bias=True, fwd_extra_dim=fwd_extra_dim, normalize_input=False, normalize_weight=False)
model.weight.data[:] = w
model.bias.data[:] = b

# Manually compute expected output per item in the batch
target = torch.zeros(batch_size, out_features + fwd_extra_dim)
for i in range(batch_size):
    # Adjust the target model's weights for each batch item to simulate fwd extra weight addition
    temp_weight = torch.cat([w, fwd_extra_weight[i]], dim=0)  # Combine basic weights with fwd extra weights for this batch
    temp_bias = torch.cat([b, torch.zeros(fwd_extra_dim)])
    temp_model = nn.Linear(in_features, out_features + fwd_extra_dim, bias=True)
    temp_model.weight.data[:] = temp_weight
    temp_model.bias.data[:] = temp_bias
    target[i] = temp_model(input[i].unsqueeze(0))

# Perform the operation with fwd extra weights
output = model(input, extra_weight=fwd_extra_weight)

# Verify the output against the manually computed target
assert torch.allclose(output, target, atol=1e-6), f"Forward extra weight test failed.\n\nexpected_output={target}\n\noutput={output}"


##########
# Combined Static and Dynamic weights

iw = torch.arange(init_extra_dim * 2).reshape(init_extra_dim, 2)  # Initial extra weights

# Dynamic forward extra weights setup
fwd_extra_weight = torch.randn(batch_size, fwd_extra_dim, 2)

# Initialize NuLinear with both static and dynamic weights
model = NuLinear(in_features, out_features, bias=True,
                 init_extra_weight=iw,
                 fwd_extra_dim=fwd_extra_dim,
                 normalize_input=False, normalize_weight=False)
model.weight.data[:] = w
model.bias.data[:] = b

# The target model setup to reflect both static and dynamic effects
# Here we simulate the combined behavior as PyTorch's nn.Linear does not directly support this feature
target = torch.zeros(batch_size, out_features + init_extra_dim + fwd_extra_dim)
for i in range(batch_size):
    # Combine basic weights with static and dynamic extra weights for this batch
    temp_weight = torch.cat([w, iw, fwd_extra_weight[i]], dim=0)
    temp_bias = torch.cat([b, torch.zeros(init_extra_dim + fwd_extra_dim)])  # Adjust bias to accommodate extra weights
    temp_model = nn.Linear(in_features, out_features + init_extra_dim + fwd_extra_dim, bias=True)
    temp_model.weight.data[:] = temp_weight
    temp_model.bias.data[:] = temp_bias
    target[i] = temp_model(input[i].unsqueeze(0))

# Perform the operation with both static and dynamic weights
output = model(input, extra_weight=fwd_extra_weight)

# Verify the output against the manually computed target
assert torch.allclose(output, target, atol=1e-6), "Combined static and dynamic weight test failed."


##########
# Cosine Similarity Equivalence

# Initialize parameters for the test
in_features = 256
out_features = 8 # n vecs to compare input against
batch_size = 4

# Create normalized inputs
inputs = torch.randn(batch_size, in_features)

model = NuLinear(in_features, out_features, normalize_input=True, normalize_weight=True, bias=False)
model.weight.data = torch.randn(out_features, in_features)
output = model(inputs)

# Manually calculate cosine similarity
cos_sim = torch.zeros(batch_size, out_features)
for i in range(batch_size):
    for j in range(out_features):
        cos_sim[i, j] = F.cosine_similarity(inputs[i].unsqueeze(0), model.weight.data[j].unsqueeze(0), dim=1)

assert torch.allclose(output, cos_sim, atol=1e-6), "Output does not match manual cosine similarity calculation."


##################################################
# Choice

##########
# Test setup

vec_size = 128
n_vecs = 2
n_choices = 2
redundancy = 1
batch_size = 3

##########
# Test 1

choice = Choice(vec_size, n_vecs, n_choices, redundancy, method='max')
input_vectors = torch.randn(batch_size, vec_size)
out = choice([input_vectors] * n_vecs)
assert out.shape == (batch_size, n_choices), "Output shape mismatch for 'max' aggregation."


##########
# Test 2

redundancy = 4
choice = Choice(vec_size, n_vecs, n_choices, redundancy, method='softmax')

for i in range(1, 10):
    for _ in range(4):
        input_vectors = torch.randn(batch_size, vec_size)
        out = choice([input_vectors] * n_vecs)
        assert torch.all(out >= 0) and torch.all(out <= 1), "Output values should be in the range [0, 1] for 'softmax' aggregation."
        assert torch.allclose(out.sum(dim=1), torch.ones(batch_size)), f'Output should sum to 1. out={out}'


##########
# Test 3

fwd_extra_weight_dim = 4 # must be a multiple of redundancy
choice = Choice(vec_size, n_vecs, n_choices, redundancy, fwd_extra_weight_dim=fwd_extra_weight_dim, method='softmax')
input_vectors = torch.randn(batch_size, vec_size)
extra_weights = torch.randn(batch_size, fwd_extra_weight_dim, vec_size * n_vecs)
out = choice([input_vectors] * n_vecs, extra_weights=extra_weights)

# Assert that the output is within the expected range [0, 1]
assert torch.all(out >= 0) and torch.all(out <= 1), "Output values should be in the range [0, 1] with dynamic forward extra weights."
assert torch.allclose(out.sum(dim=1), torch.ones(batch_size)), f'Output should sum to 1. out={out}'
