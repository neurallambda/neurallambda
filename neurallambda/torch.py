'''

Torch helpers

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, f, nargs=1):
        super(Fn, self).__init__()
        self.f = f
        self.nargs = nargs
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


class Diagnose(nn.Module):
    def __init__(self, should_raise=True):
        super(Diagnose, self).__init__()
        self.should_raise = should_raise
    def forward(self, input):
        print(f'Input      :', input)
        print(f'Input Shape: {input.shape}')
        if self.should_raise:
            raise RuntimeError('Done diangosing')
