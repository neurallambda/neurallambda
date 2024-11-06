'''

Exploring the design space of flattening params, and inflating them back into a model, or function that works like the model

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fx as fx
import operator
from functools import wraps

from typing import Tuple, Dict, Any

TEST = True

##################################################
# Control Inversion


###########
# "Batchified" versions of the `torch.nn.functional` api
#
# NOTE: these currently loop over the batch dim, and still use the `F` api. Ideally, they'd be implemented more efficiently.


# *****
# SEE t13_metalearning_hypernet_autoencoder_functional_03 for NON-LOOP VERSIONS
# *****

def batch_embedding(input, weight, padding_idx=None):
    batch_size = input.shape[0]
    outputs = []
    for i in range(batch_size):
        output = F.embedding(input[i], weight[i], padding_idx)
        outputs.append(output)
    return torch.stack(outputs)

# def batch_linear(input, weight, bias=None):
#     batch_size = input.shape[0]
#     outputs = []
#     for i in range(batch_size):
#         output = F.linear(input[i], weight[i], bias[i] if bias is not None else None)
#         outputs.append(output)
#     return torch.stack(outputs)

def batch_linear(input, weight, bias=None):
    batch_size = input.shape[0]
    outputs = []
    for i in range(batch_size):
        output = F.linear(input[i].unsqueeze(0), weight[i], bias[i] if bias is not None else None)
        outputs.append(output.squeeze(0))
    return torch.stack(outputs)

def batch_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    batch_size = input.shape[0]
    outputs = []
    for i in range(batch_size):
        output = F.conv1d(input[i].unsqueeze(0), weight[i], bias[i] if bias is not None else None,
                          stride=stride, padding=padding, dilation=dilation, groups=groups)
        outputs.append(output.squeeze(0))
    return torch.stack(outputs)

def batch_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    batch_size = input.shape[0]
    outputs = []
    for i in range(batch_size):
        output = F.conv2d(input[i].unsqueeze(0), weight[i], bias[i] if bias is not None else None,
                          stride=stride, padding=padding, dilation=dilation, groups=groups)
        outputs.append(output.squeeze(0))
    return torch.stack(outputs)


def batch_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    batch_size = input.shape[0]
    outputs = []
    for i in range(batch_size):
        output = F.conv3d(input[i].unsqueeze(0), weight[i], bias[i] if bias is not None else None,
                          stride=stride, padding=padding, dilation=dilation, groups=groups)
        outputs.append(output.squeeze(0))
    return torch.stack(outputs)

def batch_layer_norm(input, normalized_shape, weight, bias, eps=1e-5):
    batch_size = input.shape[0]
    outputs = []
    for i in range(batch_size):
        output = F.layer_norm(input[i], normalized_shape, weight[i], bias[i], eps)
        outputs.append(output)
    return torch.stack(outputs)



def batch_conv_transpose2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    batch_size = input.shape[0]
    outputs = []
    for i in range(batch_size):
        output = F.conv_transpose2d(input[i].unsqueeze(0), weight[i], bias[i] if bias is not None else None,
                          stride=stride, padding=padding, dilation=dilation, groups=groups)
        outputs.append(output.squeeze(0))
    return torch.stack(outputs)


####################
#

def create_placeholder(graph, name):
    return graph.create_node('placeholder', name.replace(".", "_"))

def replace_node_with_function(graph, node, func, args, kwargs):
    with graph.inserting_after(node):
        new_node = graph.create_node('call_function', func, args=args, kwargs=kwargs)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

def get_module_params(target_module, param_nodes, prefix=''):
    params = {}
    for name, param in target_module.named_parameters():
        full_name = f"{prefix}.{name}" if prefix else name
        params[name] = param_nodes[full_name]
    return params

def transform_linear(graph, node, target, param_nodes, use_batchify):
    params = get_module_params(target, param_nodes, node.target)
    func = batch_linear if use_batchify else F.linear
    replace_node_with_function(graph, node, func, (node.args[0], params['weight'], params['bias']), {})

def transform_layer_norm(graph, node, target, param_nodes, use_batchify):
    params = get_module_params(target, param_nodes, node.target)
    args = (node.args[0], target.normalized_shape)
    kwargs = {'weight': params['weight'], 'bias': params['bias'], 'eps': target.eps}
    func = batch_layer_norm if use_batchify else F.layer_norm
    replace_node_with_function(graph, node, func, args, kwargs)

def transform_embedding(graph, node, target, param_nodes, use_batchify):
    params = get_module_params(target, param_nodes, node.target)
    args = (node.args[0], params['weight'])
    kwargs = {'padding_idx': target.padding_idx}
    func = batch_embedding if use_batchify else F.embedding
    replace_node_with_function(graph, node, func, args, kwargs)

def transform_conv(graph, node, target, param_nodes, conv_func, batch_conv_func, use_batchify):
    params = get_module_params(target, param_nodes, node.target)
    args = (node.args[0], params['weight'])
    kwargs = {
        'bias': params['bias'],
        'stride': target.stride,
        'padding': target.padding,
        'dilation': target.dilation,
        'groups': target.groups
    }
    func = batch_conv_func if use_batchify else conv_func
    replace_node_with_function(graph, node, func, args, kwargs)


##########

def transform_functional_conv2d(graph, node, param_nodes, use_batchify):
    args = list(node.args)
    kwargs = dict(node.kwargs)

    # Replace weight with placeholder if it's a parameter
    if isinstance(args[1], fx.Node) and args[1].op == 'get_attr':
        args[1] = param_nodes.get(args[1].target, args[1])

    # Replace bias with placeholder if it's a parameter
    if len(args) > 2 and isinstance(args[2], fx.Node) and args[2].op == 'get_attr':
        args[2] = param_nodes.get(args[2].target, args[2])

    func = batch_conv2d if use_batchify else F.conv2d
    replace_node_with_function(graph, node, func, tuple(args), kwargs)

def transform_functional_conv_transpose2d(graph, node, param_nodes, use_batchify):
    args = list(node.args)
    kwargs = dict(node.kwargs)

    # Replace weight with placeholder if it's a parameter
    if isinstance(args[1], fx.Node) and args[1].op == 'get_attr':
        args[1] = param_nodes.get(args[1].target, args[1])

    # Replace bias with placeholder if it's a parameter
    if len(args) > 2 and isinstance(args[2], fx.Node) and args[2].op == 'get_attr':
        args[2] = param_nodes.get(args[2].target, args[2])

    func = batch_conv_transpose2d if use_batchify else F.conv_transpose2d
    replace_node_with_function(graph, node, func, tuple(args), kwargs)


def transform_functional_linear(graph, node, param_nodes, use_batchify):
    args = list(node.args)
    kwargs = dict(node.kwargs)

    # Replace weight with placeholder if it's a parameter
    if isinstance(args[1], fx.Node) and args[1].op == 'get_attr':
        args[1] = param_nodes.get(args[1].target, args[1])

    # Replace bias with placeholder if it's a parameter
    if len(args) > 2 and isinstance(args[2], fx.Node) and args[2].op == 'get_attr':
        args[2] = param_nodes.get(args[2].target, args[2])

    func = batch_linear if use_batchify else F.linear
    replace_node_with_function(graph, node, func, tuple(args), kwargs)

def transform_transpose(graph, node, param_nodes, use_batchify):
    args = list(node.args)
    kwargs = dict(node.kwargs)

    # Replace weight with placeholder if it's a parameter
    if isinstance(args[1], fx.Node) and args[1].op == 'get_attr':
        args[1] = param_nodes.get(args[1].target, args[1])

    func = lambda x, d0, d1: torch.transpose(x, d0, d1)
    replace_node_with_function(graph, node, func, tuple(args), kwargs)



def batch_transpose(x, dim0, dim1):
    return torch.stack([torch.transpose(xi, dim0, dim1) for xi in x])

def transform_transpose(graph, node, param_nodes, use_batchify):
    args = node.args
    kwargs = node.kwargs

    # No need to check for weights or use different versions for batching
    func = batch_transpose if use_batchify else lambda x, d0, d1: torch.transpose(x, d0, d1)
    replace_node_with_function(graph, node, func, args, kwargs)




##########


# targets the `torch.nn` API
def get_layer_transforms(use_batchify):
    return {
        nn.Linear: lambda graph, node, target, param_nodes: transform_linear(graph, node, target, param_nodes, use_batchify),
        nn.LayerNorm: lambda graph, node, target, param_nodes: transform_layer_norm(graph, node, target, param_nodes, use_batchify),
        nn.Embedding: lambda graph, node, target, param_nodes: transform_embedding(graph, node, target, param_nodes, use_batchify),
        nn.Conv1d: lambda graph, node, target, param_nodes: transform_conv(graph, node, target, param_nodes, F.conv1d, batch_conv1d, use_batchify),
        nn.Conv2d: lambda graph, node, target, param_nodes: transform_conv(graph, node, target, param_nodes, F.conv2d, batch_conv2d, use_batchify),
        nn.Conv3d: lambda graph, node, target, param_nodes: transform_conv(graph, node, target, param_nodes, F.conv3d, batch_conv3d, use_batchify),
    }

# targets the `torch.nn.functional` API
def get_functional_transforms(use_batchify):
    return {
        F.conv2d: lambda graph, node, param_nodes: transform_functional_conv2d(graph, node, param_nodes, use_batchify),
        F.conv_transpose2d: lambda graph, node, param_nodes: transform_functional_conv_transpose2d(graph, node, param_nodes, use_batchify),
        F.linear: lambda graph, node, param_nodes: transform_functional_linear(graph, node, param_nodes, use_batchify),
        torch.transpose: lambda graph, node, param_nodes: transform_transpose(graph, node, param_nodes, use_batchify),
    }

def get_attr_or_module(module, target):
    atoms = target.split('.')
    attr = module
    for atom in atoms:
        if not hasattr(attr, atom):
            raise AttributeError(f"Attribute {atom} not found")
        attr = getattr(attr, atom)
    return attr

def transform_control_inversion(module: nn.Module, use_batchify: bool = False) -> nn.Module:
    '''Transform a module so that it no longer tracks its own parameter state, but
    instead these must be passed in as kwargs.

    Args:
        module (nn.Module): The module to transform
        use_batchify (bool): If True, use batched versions of operations to allow
                             different parameter values for each item in the batch

    Currently handles:
    - Parameter, ParameterList, ParameterDict
    - Module, ModuleList, ModuleDict
    - Conv1d, Conv2d, Conv3d
    - Linear
    - Embedding
    - LayerNorm

    Missing support for:
    - RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell: These do not have helper functions in `torch.nn.functional`, so would have to be implemented by hand
    - BatchNorm1d, BatchNorm2d, BatchNorm3d
    - InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
    - TransformerEncoderLayer, TransformerDecoderLayer
    - MultiheadAttention
    - LPPool1d, LPPool2d (when trainable=True)
    - AdaptiveLogSoftmaxWithLoss
    '''
    traced = fx.symbolic_trace(module)
    graph = traced.graph

    layer_transforms = get_layer_transforms(use_batchify)
    functional_transforms = get_functional_transforms(use_batchify)

    # Find the parameter-containing nodes and store their information
    param_layers = []
    for node in graph.nodes:
        if node.op == 'call_module':
            target_module = get_attr_or_module(traced, node.target)
            if isinstance(target_module, tuple(layer_transforms.keys())):
                param_layers.append((node, target_module, type(target_module)))
        elif node.op == 'get_attr':
            target_attr = get_attr_or_module(traced, node.target)
            if isinstance(target_attr, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                param_layers.append((node, target_attr, type(target_attr)))

    # Create new input nodes for parameters at the beginning of the graph
    param_nodes = {}
    input_placeholder = next(node for node in reversed(graph.nodes) if node.op == 'placeholder')  # last arg to input after
    with graph.inserting_after(input_placeholder):
        for name, param in traced.named_parameters():
            param_nodes[name] = create_placeholder(graph, name)

    # Modify the layer calls to use external parameters
    for node, target, layer_type in param_layers:
        if layer_type in layer_transforms:
            layer_transforms[layer_type](graph, node, target, param_nodes)
        elif layer_type in (nn.Parameter, nn.ParameterList, nn.ParameterDict):
            if layer_type == nn.Parameter:
                node.replace_all_uses_with(param_nodes[node.target])
                graph.erase_node(node)
            else:
                for user in list(node.users):  # Create a copy of users to avoid modification during iteration
                    if user.op == 'call_function' and user.target in (operator.getitem, dict.__getitem__):
                        idx_or_key = user.args[1]
                        new_node = param_nodes[f'{node.target}.{idx_or_key}']
                        user.replace_all_uses_with(new_node)
                        graph.erase_node(user)
                graph.erase_node(node)

    # TODO: the underlying transforms try to be batchify aware, but they shouldn't, they should only be relevant in a batchified world
    if use_batchify:
        # Handle functional calls
        for node in graph.nodes:
            if node.op == 'call_function' and node.target in functional_transforms:
                functional_transforms[node.target](graph, node, param_nodes)

    graph.lint()
    return fx.GraphModule(traced, graph)


##################################################
#

def build_params_dict(module: nn.Module, prefix: str = '') -> dict:
    ''' Build a dictionary of kwargs compatible with transform_control_inversion by flattening out and namespacing all parameters within the module. '''
    params = {}
    for name, param in module.named_parameters():
        full_name = f"{prefix}_{name}" if prefix else name
        # params[full_name.replace('.', '_')] = nn.Parameter(param.data.clone())  # should it be copy of Parameter?
        params[full_name.replace('.', '_')] = param

    for name, child in module.named_children():
        child_prefix = f"{prefix}_{name}" if prefix else name
        params.update(build_params_dict(child, prefix=child_prefix))

    return params


##################################################
# Sandbox

##########
# Training Linear

if TEST:
    torch.manual_seed(152)
    print()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Create and transform the MLP
    original_mlp = Model()
    transformed_mlp = transform_control_inversion(original_mlp)

    params = build_params_dict(original_mlp)
    num_epochs = 10

    # Generate sample data
    x = torch.randn(1, 10)
    y = torch.randn(1, 1)

    # Train the transformed model
    optimizer = optim.AdamW(params.values(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        # Forward pass
        y_pred = transformed_mlp(x, **params)

        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


##########
# Stress Test

if TEST:
    torch.manual_seed(152)
    print()

    class ConvModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(32, 64, 3, stride=1, padding=1)
            self.conv2d = nn.Conv2d(3, 32, 3, stride=1, padding=1)
            self.conv3d = nn.Conv3d(3, 32, 3, stride=1, padding=1)

        def forward(self, x1d, x2d, x3d):
            y1d = self.conv1d(x1d)
            y2d = self.conv2d(x2d)
            y3d = self.conv3d(x3d)
            return y1d, y2d, y3d

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 32)
            self.conv_module = ConvModule()
            self.linear = nn.Linear(32, 64)
            self.param_list = nn.ParameterList([nn.Parameter(torch.randn(64)) for _ in range(3)])
            self.param_dict = nn.ParameterDict({
                'a': nn.Parameter(torch.randn(64)),
                'b': nn.Parameter(torch.randn(64))
            })

            self.module_list = nn.ModuleList([
                nn.Linear(64, 64),
                nn.Linear(64, 64),
                nn.Linear(64, 64),
            ])

            self.module_dict = nn.ModuleDict({
                'a': nn.Linear(64, 64),
                'b': nn.Linear(64, 64),
                'c': nn.Linear(64, 64),
            })


        def forward(self, x, x1d, x2d, x3d):
            x = self.embed(x)  # (batch_size, seq_len, 32)
            x = self.linear(x)  # (batch_size, seq_len, 64)
            y1d, y2d, y3d = self.conv_module(x1d, x2d, x3d)

            # Sum over all dimensions except batch
            x = x.sum(dim=(1, 2))  # (batch_size,)
            y1d = y1d.sum(dim=(1, 2))  # (batch_size,)
            y2d = y2d.sum(dim=(1, 2, 3))  # (batch_size,)
            y3d = y3d.sum(dim=(1, 2, 3, 4))  # (batch_size,)

            x = x + y1d + y2d + y3d  # (batch_size,)
            x = x.unsqueeze(1)  # (batch_size, 1)

            x = x + self.param_list[0] + self.param_list[1] + self.param_list[2]
            x = x + self.param_dict['a'] + self.param_dict['b']

            for m in self.module_list:
                x = m(x)

            for k, m in self.module_dict.items():
                x = m(x)

            return x

    # Original module
    original_module = MyModule()

    # Transformed module
    transformed_module = transform_control_inversion(MyModule())

    # Input tensors
    batch_size = 4
    seq_len = 10
    x = torch.randint(0, 1000, (batch_size, seq_len))  # (batch_size, seq_len)
    x1d = torch.randn(batch_size, 32, 20)  # (batch_size, C_in, L)
    x2d = torch.randn(batch_size, 3, 28, 28)  # (batch_size, C_in, H, W)
    x3d = torch.randn(batch_size, 3, 16, 16, 16)  # (batch_size, C_in, D, H, W)

    # Get parameters from the original module
    # params = {name.replace('.', '_'): param for name, param in original_module.named_parameters()}
    params = build_params_dict(original_module)

    # Compute outputs
    with torch.no_grad():
        original_output = original_module(x, x1d, x2d, x3d)
        transformed_output = transformed_module(x, x1d, x2d, x3d, **params)

    # Check if outputs are equal
    assert torch.allclose(original_output, transformed_output, atol=1e-6), "Outputs are not equal!"
    print("Outputs are equal. Control inversion successful!")

    # Print the code of the transformed module
    print("\nTransformed module code:")
    print(transformed_module.code)


##################################################
# Vectorize / Unvectorize

def vectorize(flattened_params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Size]]:
    ''' Take the flattened dictionary output of `build_params_dict` and flatten underlying tensors into a single vector. '''
    vectors = []
    shapes = {}
    for name, param in flattened_params.items():
        if isinstance(param, (nn.Parameter, torch.Tensor)):
            shapes[name] = param.shape
            thing = param + 0
            vectors.append(thing.flatten())
        else:
            raise TypeError(f"Unexpected type in flattened_params: {type(param)}")

    return torch.cat(vectors), shapes


# def unvectorize(vector: torch.Tensor, shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
#     ''' Inflate a large vector into weights that match `shapes` using torch.split. '''
#     unflattened = {}
#     split_sizes = [torch.prod(torch.tensor(shape)).item() for shape in shapes.values()]
#     chunks = torch.split(vector, split_sizes)
#     for (name, shape), chunk in zip(shapes.items(), chunks):
#         unflattened[name] = chunk.view(shape)
#     return unflattened


def numel(params_dict: dict) -> int:
    total_elements = 0
    for param in params_dict.values():
        if isinstance(param, (nn.Parameter, torch.Tensor)):
            total_elements += param.numel()
    return total_elements

# def unvectorize(vector: torch.Tensor, shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
#     ''' Inflate a large vector into weights that match `shapes`. '''
#     unflattened = {}
#     start = 0
#     for name, shape in shapes.items():
#         num_elements = torch.prod(torch.tensor(shape)).item()
#         end = start + num_elements
#         unflattened[name] = vector[start:end].view(shape)
#         start = end

#     if start != vector.numel():
#         raise ValueError(f"Vector size ({vector.numel()}) does not match the total number of elements in shapes ({start})")

#     return unflattened

def unvectorize(vector: torch.Tensor, shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
    ''' Inflate a large vector into weights that match `shapes` using torch.tensor_split. '''
    unflattened = {}
    split_sizes = [torch.prod(torch.tensor(shape)).item() for shape in shapes.values()]

    # Calculate the indices for splitting
    indices = torch.cumsum(torch.tensor(split_sizes), dim=0)[:-1]

    # Use torch.tensor_split to split the vector
    chunks = torch.tensor_split(vector, indices)

    for (name, shape), chunk in zip(shapes.items(), chunks):
        unflattened[name] = chunk.view(shape)

    return unflattened

def unvectorize(vector: torch.Tensor, shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
    ''' Inflate a single vector or a batch of vectors into weights that match `shapes`. '''
    if vector.dim() not in [1, 2]:
        raise ValueError(f"Input vector must be 1D or 2D, but got {vector.dim()}D")

    unflattened = {}
    split_sizes = [torch.prod(torch.tensor(shape)).item() for shape in shapes.values()]
    batch_size = vector.shape[0] if vector.dim() == 2 else 1

    # Split the vector along the last dimension
    chunks = torch.split(vector, split_sizes, dim=-1)

    for (name, shape), chunk in zip(shapes.items(), chunks):
        new_shape = (batch_size,) + shape if vector.dim() == 2 else shape
        unflattened[name] = chunk.view(new_shape)

    return unflattened


def check_equality(dict1: Dict[str, Any], dict2: Dict[str, Any], rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    ''' Check equality between 2 dictionaries of tensors/parameters. '''
    if set(dict1.keys()) != set(dict2.keys()):
        print(f"Keys don't match. dict1 keys: {set(dict1.keys())}, dict2 keys: {set(dict2.keys())}")
        return False

    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, (nn.Parameter, torch.Tensor)) and isinstance(val2, (nn.Parameter, torch.Tensor)):
            if isinstance(val1, nn.Parameter):
                val1 = val1.data
            if isinstance(val2, nn.Parameter):
                val2 = val2.data

            if val1.shape != val2.shape:
                print(f"Shape mismatch for key '{key}': {val1.shape} vs {val2.shape}")
                return False

            if not torch.allclose(val1, val2, rtol=rtol, atol=atol):
                print(f"Values don't match for key '{key}'")
                print(f"Max absolute difference: {torch.max(torch.abs(val1 - val2)).item()}")
                return False

        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not check_equality(val1, val2, rtol=rtol, atol=atol):
                return False

        else:
            print(f"Unexpected type mismatch for key '{key}': {type(val1)} vs {type(val2)}")
            return False

    return True


##############################

# def test_vectorize_backprop():
#     # Create a sample dictionary of parameters
#     params = {
#         'layer1.weight': nn.Parameter(torch.randn(3, 4)),
#         'layer1.bias': nn.Parameter(torch.randn(3)),
#         'layer2.weight': nn.Parameter(torch.randn(2, 3)),
#         'layer2.bias': nn.Parameter(torch.randn(2)),
#     }

#     # Vectorize the parameters
#     vector, shapes = vectorize(params)

#     # Create a dummy loss
#     loss = vector.sum()

#     # Try to backpropagate
#     try:
#         loss.backward()
#         print("Vectorize: Backpropagation successful!")
#         # Check if gradients were computed
#         all_have_grad = all(p.grad is not None for p in params.values())
#         assert all_have_grad
#         print(f"All parameters have gradients: {all_have_grad}")
#     except Exception as e:
#         print(f"Vectorize: Backpropagation failed with error: {str(e)}")


# def test_unvectorize_backprop():
#     # Create a sample vector and shapes dictionary
#     vector = nn.Parameter(torch.randn(23))
#     shapes = {
#         'layer1.weight': (3, 4),
#         'layer1.bias': (3,),
#         'layer2.weight': (2, 3),
#         'layer2.bias': (2,),
#     }

#     # Unvectorize the vector
#     params = unvectorize(vector, shapes)

#     # Create a dummy loss
#     loss = sum(p.sum() for p in params.values())

#     # Try to backpropagate
#     try:
#         loss.backward()
#         print("Unvectorize: Backpropagation successful!")
#         # Check if gradient was computed for the original vector
#         assert vector.grad is not None
#         print(f"Original vector has gradient: {vector.grad is not None}")
#     except Exception as e:
#         print(f"Unvectorize: Backpropagation failed with error: {str(e)}")


# def test_vectorize_unvectorize_backprop():
#     # Create a sample dictionary of parameters
#     original_params = {
#         'layer1.weight': nn.Parameter(torch.randn(3, 4, )),
#         'layer1.bias': nn.Parameter(torch.randn(3, )),
#         'layer2.weight': nn.Parameter(torch.randn(2, 3, )),
#         'layer2.bias': nn.Parameter(torch.randn(2, )),
#     }

#     # Vectorize the parameters
#     vector, shapes = vectorize(original_params)

#     # Apply some operation to the vector
#     modified_vector = vector * 2  # Simple scaling operation

#     # Unvectorize the modified vector
#     reconstructed_params = unvectorize(modified_vector, shapes)

#     # Create a dummy loss using the reconstructed parameters
#     loss = torch.stack([p.sum() for p in reconstructed_params.values()]).sum()

#     # Try to backpropagate
#     try:
#         loss.backward()
#         print("Vectorize + Unvectorize: Backpropagation successful!")

#         # Check if gradients were computed for original parameters
#         original_grads = all(p.grad is not None for p in original_params.values())
#         print(f"All original parameters have gradients: {original_grads}")

#         # # This grad has not been retained
#         # vectorized_grad = vector.grad is not None
#         # print(f'Vectorized version has grads: {vectorized_grad}')

#         # # These grads were not retained
#         # # Check if gradients were computed for reconstructed parameters
#         # reconstructed_grads = all(p.grad is not None for p in reconstructed_params.values())
#         # print(f"All reconstructed parameters have gradients: {reconstructed_grads}")

#         # Print some gradient values for verification
#         print("\nSample gradient values:")
#         for name, param in original_params.items():
#             if param.grad is not None:
#                 assert param.grad.abs().sum().item() > 0
#                 print(f"{name} grad: {param.grad.sum().item()}")

#     except Exception as e:
#         print(f"Vectorize + Unvectorize: Backpropagation failed with error: {str(e)}")



def test_vectorize_backprop(batch_size=None):
    # Create a sample dictionary of parameters
    params = {
        'layer1.weight': nn.Parameter(torch.randn(3, 4)),
        'layer1.bias': nn.Parameter(torch.randn(3)),
        'layer2.weight': nn.Parameter(torch.randn(2, 3)),
        'layer2.bias': nn.Parameter(torch.randn(2)),
    }

    # Vectorize the parameters
    vector, shapes = vectorize(params)

    # Create a batch if specified
    if batch_size is not None:
        vector = vector.unsqueeze(0).repeat(batch_size, 1)

    # Create a dummy loss
    loss = vector.sum()

    # Try to backpropagate
    try:
        loss.backward()
        print(f"Vectorize ({'batched' if batch_size else 'single'}): Backpropagation successful!")
        # Check if gradients were computed
        all_have_grad = all(p.grad is not None for p in params.values())
        assert all_have_grad
        print(f"All parameters have gradients: {all_have_grad}")
    except Exception as e:
        print(f"Vectorize ({'batched' if batch_size else 'single'}): Backpropagation failed with error: {str(e)}")

def test_unvectorize_backprop(batch_size=None):
    # Create a sample vector and shapes dictionary
    vector = nn.Parameter(torch.randn(23))
    shapes = {
        'layer1.weight': (3, 4),
        'layer1.bias': (3,),
        'layer2.weight': (2, 3),
        'layer2.bias': (2,),
    }

    # Create a batch if specified
    if batch_size is not None:
        vector = vector.unsqueeze(0).repeat(batch_size, 1)

    # Unvectorize the vector
    params = unvectorize(vector, shapes)

    # Create a dummy loss
    loss = sum(p.sum() for p in params.values())

    # Try to backpropagate
    try:
        loss.backward()
        print(f"Unvectorize ({'batched' if batch_size else 'single'}): Backpropagation successful!")
        # Check if gradient was computed for the original vector
        assert vector.grad is not None
        print(f"Original vector has gradient: {vector.grad is not None}")
    except Exception as e:
        print(f"Unvectorize ({'batched' if batch_size else 'single'}): Backpropagation failed with error: {str(e)}")

def test_vectorize_unvectorize_backprop(batch_size=None):
    # Create a sample dictionary of parameters
    original_params = {
        'layer1.weight': nn.Parameter(torch.randn(3, 4)),
        'layer1.bias': nn.Parameter(torch.randn(3)),
        'layer2.weight': nn.Parameter(torch.randn(2, 3)),
        'layer2.bias': nn.Parameter(torch.randn(2)),
    }

    # Vectorize the parameters
    vector, shapes = vectorize(original_params)

    # Create a batch if specified
    if batch_size is not None:
        vector = vector.unsqueeze(0).repeat(batch_size, 1)

    # Apply some operation to the vector
    modified_vector = vector * 2  # Simple scaling operation

    # Unvectorize the modified vector
    reconstructed_params = unvectorize(modified_vector, shapes)

    # Create a dummy loss using the reconstructed parameters
    loss = sum(p.sum() for p in reconstructed_params.values())

    # Try to backpropagate
    try:
        loss.backward()
        print(f"Vectorize + Unvectorize ({'batched' if batch_size else 'single'}): Backpropagation successful!")

        # Check if gradients were computed for original parameters
        original_grads = all(p.grad is not None for p in original_params.values())
        print(f"All original parameters have gradients: {original_grads}")

        # Print some gradient values for verification
        print("\nSample gradient values:")
        for name, param in original_params.items():
            if param.grad is not None:
                assert param.grad.abs().sum().item() > 0
                print(f"{name} grad: {param.grad.sum().item()}")

    except Exception as e:
        print(f"Vectorize + Unvectorize ({'batched' if batch_size else 'single'}): Backpropagation failed with error: {str(e)}")


##############################
# round trip check

# if TEST:
#     torch.manual_seed(152)
#     print()

#     class Model(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.fc1 = nn.Linear(10, 20)
#             self.fc2 = nn.Linear(20, 1)

#         def forward(self, x):
#             x = torch.relu(self.fc1(x))
#             return self.fc2(x)

#     original_mlp = Model()
#     params = build_params_dict(original_mlp)

#     v, shape_template = vectorize(params)
#     uv = unvectorize(v, shape_template)

#     assert check_equality(params, uv)
#     print('vectorize + unvectorize round trip works')


#     # More Tests
#     print("Testing vectorize function:")
#     test_vectorize_backprop()

#     print("\nTesting unvectorize function:")
#     test_unvectorize_backprop()

#     print("Testing vectorize and unvectorize functions together:")
#     test_vectorize_unvectorize_backprop()



if TEST:
    print("Testing single input:")
    test_vectorize_backprop()
    test_unvectorize_backprop()
    test_vectorize_unvectorize_backprop()

    print("\nTesting batched input (batch_size=32):")
    test_vectorize_backprop(batch_size=32)
    test_unvectorize_backprop(batch_size=32)
    test_vectorize_unvectorize_backprop(batch_size=32)

    # Round trip check
    torch.manual_seed(152)
    print("\nRound trip check:")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    original_mlp = Model()
    params = build_params_dict(original_mlp)

    v, shape_template = vectorize(params)
    uv = unvectorize(v, shape_template)

    assert check_equality(params, uv)
    print('vectorize + unvectorize round trip works for single input')

    # Batched round trip check
    v_batched = v.unsqueeze(0).repeat(32, 1)
    uv_batched = unvectorize(v_batched, shape_template)
    assert all(torch.allclose(uv[k], uv_batched[k][0]) for k in uv.keys())
    print('vectorize + unvectorize round trip works for batched input')



##################################################
# Integration check

if TEST:
    torch.manual_seed(152)
    print()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Create and transform the MLP
    original_mlp = Model()
    transformed_mlp = transform_control_inversion(original_mlp)
    orig_params = build_params_dict(original_mlp)

    num_epochs = 10

    x = torch.randn(1, 10)
    y = torch.randn(1, 1)


    # Train the transformed model
    optimizer = optim.AdamW(orig_params.values(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        # Forward pass
        v, template = vectorize(orig_params)
        v_mod = v * 0.1
        modified_params = unvectorize(v_mod, template)

        y_pred = transformed_mlp(x, **modified_params)

        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


##################################################
# Test Batchify

if TEST:
    torch.manual_seed(152)

    class BatchTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 32)
            self.conv1d = nn.Conv1d(32, 64, 3, stride=1, padding=1)
            self.linear1 = nn.Linear(64, 32)
            self.linear2 = nn.Linear(32, 1)

        def forward(self, x):
            x = self.embed(x)
            x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
            x = self.conv1d(x)
            x = x.mean(dim=2)  # Global average pooling
            x = torch.relu(self.linear1(x))
            return self.linear2(x)

    # Create and transform the model
    original_model = BatchTestModel()
    transformed_model = transform_control_inversion(original_model, use_batchify=True)

    # Create batched input
    batch_size = 4
    seq_len = 10
    x = torch.arange(batch_size * seq_len).view(batch_size, seq_len) % 1000

    # Get original parameters and create batched versions
    orig_params = build_params_dict(original_model)
    batched_params = {}
    for name, param in orig_params.items():
        batched_params[name] = nn.Parameter(param.repeat(batch_size, *[1 for _ in range(param.dim())]))

    # Check outputs before training
    with torch.no_grad():
        original_output = original_model(x)
        transformed_output = transformed_model(x, **batched_params)
        assert torch.allclose(original_output, transformed_output, atol=1e-6), "Outputs are not equal before training!"
        print("Original and transformed model outputs match before training.")

    # Generate target values
    y = torch.randn(batch_size, 1)

    # Optimizer for batched parameters
    optimizer = optim.Adam(batched_params.values(), lr=0.01)
    criterion = nn.MSELoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        # Forward pass with batched parameters
        y_pred = transformed_model(x, **batched_params)

        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Verify that batched parameters have different values across the batch dimension
    for name, param in batched_params.items():
        unique_params = torch.unique(param, dim=0)
        assert unique_params.shape[0] == batch_size, f"Parameter {name} does not have unique values for each batch item"

    print("Batchify integration test passed successfully!")

    # Compare outputs of original and transformed models after training
    with torch.no_grad():
        original_output = original_model(x)
        transformed_output = transformed_model(x, **batched_params)
        print(f"Max difference between original and transformed outputs after training: {(original_output - transformed_output).abs().max().item()}")



##################################################
# Test `torch.nn.functional` replacements

if TEST:
    torch.manual_seed(152)

    class BatchTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_w = nn.Parameter(torch.randn(32 * 32, 32 * 32))
            # convolution via functional api
            self.conv_w = nn.Parameter(torch.randn(5, 16, 3, 3))  # 16 channels in, 5 channels out
            self.conv_b = nn.Parameter(torch.randn(5))

        def forward(self, x):
            B, chan = x.shape[0], x.shape[1]
            x = torch.flatten(x, start_dim=-2, end_dim=-1)
            x = F.linear(x, self.lin_w)
            x = x.reshape(B, chan, 32, 32)
            x = F.conv2d(x, self.conv_w, self.conv_b, stride=1, padding=1)  # Direct use of F.conv2d
            return x

    # Create and transform the model
    original_model = BatchTestModel()
    transformed_model = transform_control_inversion(original_model, use_batchify=True)

    # Create batched input
    batch_size = 7
    x = torch.randn(batch_size, 16, 32, 32)
    y = torch.randn(batch_size, 5, 32, 32)

    # Get original parameters and create batched versions
    orig_params = build_params_dict(original_model)
    batched_params = {}
    for name, param in orig_params.items():
        batched_params[name] = nn.Parameter(param.repeat(batch_size, *[1 for _ in range(param.dim())]))

    # Check outputs before training
    with torch.no_grad():
        original_output = original_model(x)
        transformed_output = transformed_model(x, **batched_params)
        assert torch.allclose(original_output, transformed_output, atol=1e-3), "Outputs are not equal before training!"
        print("Original and transformed model outputs match before training.")

    # Generate target values


    # Optimizer for batched parameters
    optimizer = optim.Adam(batched_params.values(), lr=0.01)
    criterion = nn.MSELoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        # Forward pass with batched parameters
        y_pred = transformed_model(x, **batched_params)

        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Verify that batched parameters have different values across the batch dimension
    for name, param in batched_params.items():
        unique_params = torch.unique(param, dim=0)
        assert unique_params.shape[0] == batch_size, f"Parameter {name} does not have unique values for each batch item"

    print("Batchify integration test passed successfully!")

    # Compare outputs of original and transformed models after training
    with torch.no_grad():
        original_output = original_model(x)
        transformed_output = transformed_model(x, **batched_params)
        print(f"Max difference between original and transformed outputs after training: {(original_output - transformed_output).abs().max().item()}")
