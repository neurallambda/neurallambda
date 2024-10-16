'''

Torch helpers

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from neurallambda.util import bold, red, green

import torch
import torch.nn.functional as F
import math

##################################################
# Better Torch

def cosine_similarity(tensor1, tensor2, dim=1, eps=1e-8):
    ''' cosine_similarity without implicit broadcasting, and better error messages. '''
    if tensor1.shape != tensor2.shape:
        error_message = (
            f"Tensor shapes must be identical. tensor1: {green(tensor1.shape)}. tensor2: {green(tensor2.shape)}.\n" \
            "Suggestions:\n" \
            f"- Use {bold('torch.broadcast_tensors')}: bx, by = torch.broadcast_tensors(x, y)\n" \
            f"- Use {bold('tensor.view(...)')} to reshape one of the tensors. For example, if you need to match the tensor of shape " \
            f"{tensor1.shape}, you can reshape tensor2 with tensor2.view{tensor1.shape} (adjust dimensions as needed).\n" \
            f"- Use {bold('tensor.expand(...)')} to match shapes without changing the data. For example, tensor2.expand_as(tensor1) " \
            "if tensor1 has the desired shape.\n" \
        )
        raise ValueError(error_message)

    return F.cosine_similarity(tensor1, tensor2, dim=dim, eps=eps)


def roll_without_wrap(tensor, shift, fill_value=0):
    if shift == 0:
        return tensor
    elif shift > 0:
        # pad = (left, right, top, bottom, front, back)
        pad = (0, 0, shift, 0, 0, 0)
        padded_tensor = F.pad(tensor, pad, mode='constant', value=fill_value)
        return padded_tensor[:, :tensor.size(1), :]
    else:
        pad = (0, 0, 0, -shift, 0, 0)
        padded_tensor = F.pad(tensor, pad, mode='constant', value=fill_value)
        return padded_tensor[:, -tensor.size(1):, :]


##################################################
# Modules

class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, weight=None, bias=False, reverse=False):
        ''' If bias=False, this is the same as Cosine Similarity. '''
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reverse = reverse

        if weight is None:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.weight = weight
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        normalized_input = F.normalize(input, p=2, dim=1)
        if self.reverse:
            normalized_weight = F.normalize(self.weight, p=2, dim=1)
            return F.linear(normalized_input, normalized_weight.t()) # - self.bias
        else:
            normalized_weight = F.normalize(self.weight, p=2, dim=1)
            return F.linear(normalized_input, normalized_weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


##################################################
# Structural

class Stack(nn.Module):
    def __init__(self, dim=1):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.stack(inputs, dim=self.dim)


class Cat(nn.Module):
    def __init__(self, dim=-1):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        # inputs is a tuple of tensors
        # Each input tensor shape=[batch, features...]
        return torch.cat(inputs, dim=self.dim)


class Parallel(nn.Module):
    def __init__(self, module_tuple):
        super(Parallel, self).__init__()
        self.ms = nn.ModuleList(module_tuple)

    def forward(self, inputs):
        # inputs is a tuple of tensors, parallelizing operations across the tuple
        # Each input tensor shape=[batch, features...]
        outputs = tuple(module(input) for module, input in zip(self.ms, inputs))
        return outputs  # Output is a tuple of tensors, shapes depend on respective modules


class Split(nn.Module):
    def __init__(self, split_sizes, dim=-1):
        super(Split, self).__init__()
        self.split_sizes = split_sizes  # Tuple of sizes to split the tensor into
        self.dim = dim

    def forward(self, input):
        # input shape=[batch, combined features...]
        # torch.split returns a tuple of tensors split according to self.split_sizes
        return torch.split(input, self.split_sizes, dim=self.dim)  # Output shapes depend on self.split_sizes


class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()
    def forward(self, x):
        return x


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(self.dim)


class Fn(nn.Module):
    def __init__(self, f, nargs=1, parameters=[]):
        super(Fn, self).__init__()
        self.f = f
        self.nargs = nargs
        if isinstance(parameters, nn.ParameterList):
            self.parameters = parameters
        else:
            self.parameters = nn.ParameterList(parameters)
    def forward(self, input):
        if self.nargs == 1:
            return self.f(input)
        if self.nargs == 2:
            x, y = input
            return self.f(x, y)
        if self.nargs == 3:
            x, y, z = input
            return self.f(x, y, z)
        if self.nargs == 4:
            a, b, c, d = input
            return self.f(a, b, c, d)
        if self.nargs == 5:
            a, b, c, d, e = input
            return self.f(a, b, c, d, e)
        if self.nargs == 6:
            a, b, c, d, e, f = input
            return self.f(a, b, c, d, e, f)
        if self.nargs == 7:
            a, b, c, d, e, f, g = input
            return self.f(a, b, c, d, e, f, g)
        if self.nargs == 8:
            a, b, c, d, e, f, g, h = input
            return self.f(a, b, c, d, e, f, g, h)


class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0, hard=False, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard
        self.dim = dim

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=self.dim)


class Diagnose(nn.Module):
    def __init__(self, should_raise=True):
        super(Diagnose, self).__init__()
        self.should_raise = should_raise
    def forward(self, input):
        print(f'Input      :', input)
        print(f'Input Shape: {input.shape}')
        if self.should_raise:
            raise RuntimeError('Done diangosing')


##################################################
# NuLinear
#
#   PROVENANCE: experiment/t04_addition_sandbox_04_maps_02.py
#

class NuLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=False,
                 normalize_input=True,
                 normalize_weight=True,
                 init_extra_weight=None,
                 fwd_extra_dim=0):
        super(NuLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.init_extra_weight = init_extra_weight
        self.fwd_extra_dim = fwd_extra_dim
        self.normalize_input = normalize_input
        self.normalize_weight = normalize_weight

        # Bias
        if bias:
            # Bias shape: [out_features]
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Weight
        if in_features > 0 and out_features > 0:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()
        else:
            self.weight = None

        if init_extra_weight is not None:
            assert init_extra_weight.dim() == 2 and init_extra_weight.size(1) == in_features, f"init_extra_weight must have shape [init_extra_dim, in_features={in_features}], but has shape={init_extra_weight.shape}"
            # Shape: [init_extra_dim, in_features]
            self.init_extra_weight = init_extra_weight
            # Adjust total output features to include init_extra_weight
            self.total_out_features = out_features + init_extra_weight.size(0) + fwd_extra_dim
        else:
            self.init_extra_weight = None
            self.total_out_features = out_features + fwd_extra_dim

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, extra_weight=None):
        # Input shape: [batch_size, in_features]
        if self.normalize_input:
            input = F.normalize(input, p=2, dim=1)

        weight = self.weight
        if self.init_extra_weight is not None:
            # Concatenated weight shape: [out_features + init_extra_dim, in_features]
            if weight is not None:
                weight = torch.cat([self.weight, self.init_extra_weight], dim=0)
            else:
                weight = self.init_extra_weight

        if extra_weight is not None:
            assert extra_weight.shape[1] == self.fwd_extra_dim and extra_weight.shape[2] == self.in_features, f"extra_weight must have shape [batch={input.size(0)}, fwd_extra_dim={self.fwd_extra_dim}, in_features={self.in_features}], but has shape={extra_weight.shape}"
            # Repeat and concatenate for shape: [batch, out_features + init_extra_dim + fwd_extra_dim, in_features]
            weight = torch.cat([weight.unsqueeze(0).repeat(extra_weight.size(0), 1, 1), extra_weight], dim=1)

        if self.normalize_weight:
            # Normalize across the appropriate dimension
            weight = F.normalize(weight, p=2, dim=-1)

        if extra_weight is not None:
            # Corrected output calculation for batched inputs
            output = torch.bmm(weight, input.unsqueeze(2)).squeeze(2)
        else:
            output = input.matmul(weight.t())

        if self.bias is not None:
            # Ensure bias is correctly expanded and added to output
            # Adjust bias shape based on actual output features
            bias = self.bias if self.init_extra_weight is None else torch.cat([self.bias, torch.zeros(self.init_extra_weight.size(0), device=self.bias.device)], 0)
            if extra_weight is not None:
                bias = torch.cat([bias, torch.zeros(self.fwd_extra_dim, device=bias.device)], 0)  # Extend bias for fwd_extra_dim
            output += bias.unsqueeze(0)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, normalize_input={}, normalize_weight={}, init_extra_weight={}, fwd_extra_dim={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.normalize_input, self.normalize_weight, self.init_extra_weight.shape if self.init_extra_weight is not None else None, self.fwd_extra_dim
        )


##################################################
# Choice
#
#   PROVENANCE: experiment/t04_addition_sandbox_04_maps_02.py
#

class Choice(nn.Module):
    ''' n-vectors -> R^m '''
    def __init__(self,
                 vec_size,
                 n_vecs,
                 n_choices,
                 redundancy,
                 has_guard=False,
                 method='softmax',
                 init_extra_weight=None,
                 fwd_extra_weight_dim=0, # must be a multiple of `redundancy`
                 ):
        super(Choice, self).__init__()

        self.vec_size = vec_size
        self.n_vecs = n_vecs
        self.n_choices = n_choices
        self.redundancy = redundancy
        self.has_guard = has_guard
        self.method = method
        self.init_extra_weight = init_extra_weight
        self.fwd_extra_weight_dim = fwd_extra_weight_dim

        assert method in {'max', 'softmax', 'gumbel_softmax', 'sum', 'mean'}

        self.ff = NuLinear(
            vec_size * n_vecs,
            redundancy * n_choices,
            bias=False,
            normalize_input=True,
            normalize_weight=True,
            init_extra_weight = init_extra_weight,
            fwd_extra_dim=fwd_extra_weight_dim,
        )

        if has_guard:
            # Note: weights for guard are init'd randomly
            self.guard = NuLinear(
                vec_size * n_vecs,
                redundancy * n_choices,
                bias=False, # TODO: <<<? seems like bias could turn off entire inputs, and that'd be good
                normalize_input=True,
                normalize_weight=True,
                fwd_extra_dim=fwd_extra_weight_dim,
            )
            self.guard_scale = nn.Parameter(torch.tensor([redundancy * 1.0]))

        self.scale = nn.Parameter(torch.tensor([redundancy * 0.1])) # TODO: this is a total guess

    def forward(self, inp, extra_weights=None, eps = 1e-6):
        for i in inp:
            assert i.ndim == 2
        batch_size = inp[0].size(0)
        sinp = torch.hstack(inp)

        outs = self.ff(sinp, extra_weights)

        hg = self.has_guard
        if hg:
            g = self.guard(sinp)

        sz = self.n_choices + self.fwd_extra_weight_dim // self.redundancy

        if self.method == 'max':
            outs = torch.max(outs.view(batch_size, sz, self.redundancy), dim=2).values
            if hg: g = torch.max(g.view(batch_size, sz, self.redundancy), dim=2).values

        elif self.method in {'softmax', 'gumbel_softmax'}:
            # softmax over the whole redundant vec, then sum each redundant chunk
            # clip because of singularities in tan and log(p/(1-p))
            outs = (outs).clip(eps, 1-eps)  # note: clips neg similarities
            outs = torch.log((outs) / (1 - outs))  # maps [0,1] -> [-inf, inf]
            # outs = torch.tan((outs - 0.5) * pi)  # maps [0,1] -> [-inf, inf]

            # outs = (outs).clip(-1+eps, 1-eps)
            # outs = torch.tan(outs * pi / 2)  # maps [-1,1] -> [-inf, inf]

            if self.method == 'softmax':
                outs = torch.sum(outs.softmax(dim=1).view(batch_size, sz, self.redundancy), dim=2)
            elif self.method == 'gumbel_softmax':
                outs = torch.sum(F.gumbel_softmax(outs, dim=1).view(batch_size, sz, self.redundancy), dim=2)

            if hg:
                g = (g).clip(eps, 1-eps)
                g = torch.log((g) / (1 - g))
                g = torch.sum(g.softmax(dim=1).view(batch_size, sz, self.redundancy), dim=2)

        elif self.method == 'sum':
            outs = torch.sum(outs.view(batch_size, sz, self.redundancy), dim=2)
            if hg: g = torch.sum(g.view(batch_size, sz, self.redundancy), dim=2)

        elif self.method == 'mean':
            outs = torch.mean(outs.view(batch_size, sz, self.redundancy), dim=2)
            if hg: g = torch.mean(g.view(batch_size, sz, self.redundancy), dim=2)

        if hg:
            outs = outs * g

        if self.method in {'sum', 'mean'}:
            outs = torch.sigmoid(outs * self.scale)

        return outs



##################################################
#

def get_last_attended_token(token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the last attended token for each sequence in the batch.

    Args:
    token_ids (torch.Tensor): Tensor of token IDs with shape [batch_size, seq_len]
    attention_mask (torch.Tensor): Boolean tensor with shape [batch_size, seq_len]

    Returns:
    torch.Tensor: Tensor of last attended tokens with shape [batch_size]
                  Contains -1 for sequences with no attended tokens or empty sequences
    """
    batch_size, seq_len = attention_mask.shape

    # Handle empty sequences
    if seq_len == 0:
        return torch.full((batch_size,), -1, dtype=torch.long, device=token_ids.device)

    # Find the position of the last attended token
    last_attended_pos = (seq_len - 1) - torch.fliplr(attention_mask).argmax(dim=1)

    # Create a range tensor for batch indexing
    batch_range = torch.arange(batch_size, device=token_ids.device)

    # Get the last attended token, use -1 to represent None
    last_tokens = torch.where(
        attention_mask.any(dim=1),
        token_ids[batch_range, last_attended_pos],
        torch.tensor(-1, device=token_ids.device)
    )

    return last_tokens


def test_get_last_attended_token():
    # Test 1: Basic case
    token_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([3, 9])), f"Test 1 failed. Expected [3, 9], got {result}"

    # Test 2: No attended tokens in one sequence
    token_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([5, -1])), f"Test 2 failed. Expected [5, -1], got {result}"

    # Test 3: Arbitrary attention patterns
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 0, 1], [0, 1, 0]])
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([3, 5])), f"Test 3 failed. Expected [3, 5], got {result}"

    # Test 4: Empty sequences
    token_ids = torch.tensor([[], []], dtype=torch.long)
    attention_mask = torch.tensor([[], []], dtype=torch.long)
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([-1, -1])), f"Test 4 failed. Expected [-1, -1], got {result}"

    # Test 5: Mixed cases
    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    attention_mask = torch.tensor([[0, 0, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([4, 7, -1])), f"Test 5 failed. Expected [4, 7, -1], got {result}"

    # Test 6: Single element sequences
    token_ids = torch.tensor([[1], [2]])
    attention_mask = torch.tensor([[1], [0]])
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([1, -1])), f"Test 6 failed. Expected [1, -1], got {result}"

    # Test 7: Large sequence length
    token_ids = torch.arange(1, 1001).unsqueeze(0).repeat(2, 1)
    attention_mask = torch.ones(2, 1000, dtype=torch.long)
    attention_mask[0, 500:] = 0
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([500, 1000])), f"Test 7 failed. Expected [500, 1000], got {result}"

    # Test 8: Random attention patterns
    torch.manual_seed(42)
    token_ids = torch.randint(1, 1000, (5, 20))
    attention_mask = torch.randint(0, 2, (5, 20))
    result = get_last_attended_token(token_ids, attention_mask)
    expected = torch.tensor([
        token_ids[0, attention_mask[0].nonzero().max()] if attention_mask[0].any() else -1,
        token_ids[1, attention_mask[1].nonzero().max()] if attention_mask[1].any() else -1,
        token_ids[2, attention_mask[2].nonzero().max()] if attention_mask[2].any() else -1,
        token_ids[3, attention_mask[3].nonzero().max()] if attention_mask[3].any() else -1,
        token_ids[4, attention_mask[4].nonzero().max()] if attention_mask[4].any() else -1,
    ])
    assert torch.all(result == expected), f"Test 8 failed. Expected {expected}, got {result}"

    # Test 9: Edge case - all tokens attended
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.ones_like(token_ids, dtype=torch.long)
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([3, 6])), f"Test 9 failed. Expected [3, 6], got {result}"

    # Test 10: Edge case - no tokens attended
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.zeros_like(token_ids, dtype=torch.long)
    result = get_last_attended_token(token_ids, attention_mask)
    assert torch.all(result == torch.tensor([-1, -1])), f"Test 10 failed. Expected [-1, -1], got {result}"

    print("All tests for get_last_attended_token passed!")

# test_get_last_attended_token()
