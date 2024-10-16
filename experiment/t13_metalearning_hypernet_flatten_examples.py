'''

Small examples demonstrating the torch.fx API as I learn how to use it

'''


'''

Bubble up control of a Linear model's `weight` parameter to the function caller

# Example:

batch_size = 3
x = torch.randn(batch_size, 32)

# normal model usage
model = MyModule()
y = model(x)

# transformed model, to give control of the Linear.weight to the function caller
w = torch.randn(16, 32)
transformed_model = ...
y = transformed_model(x, w)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
torch.manual_seed(152)
print()


##################################################
# Print info

def print_module_info(module: nn.Module):
    print(f"Module: {module.__class__.__name__}")
    print(f"Module structure:\n{module}\n")

    traced = fx.symbolic_trace(module)

    print("Traced Graph:")
    print(traced.graph)
    print("\nGraph Code:")
    print(traced.code)

    print("\nTabular Graph Representation:")
    traced.graph.print_tabular()

    print("\nDetailed Node Information:")
    for node in traced.graph.nodes:
        print(f"\nNode: {node.name}")
        print(f"  Op: {node.op}")
        print(f"  Target: {node.target}")
        print(f"  Args: {node.args}")
        print(f"  Kwargs: {node.kwargs}")
        print(f"  Type: {node.type}")
        if node.op == 'call_module':
            submodule = traced.get_submodule(node.target)
            print(f"  Submodule: {submodule}")
            if isinstance(submodule, nn.Linear):
                print(f"    Input features: {submodule.in_features}")
                print(f"    Output features: {submodule.out_features}")
                print(f"    Bias: {submodule.bias is not None}")

    print("\nModule Parameters:")
    for name, param in module.named_parameters():
        print(f"  {name}: shape {param.shape}")


##################################################
# Exploring the api

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
print()

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
        )

    def forward(self, x):
        return self.ffnn(x)

# Create an instance of the original module
original_module = MyModule()

traced = fx.symbolic_trace(original_module)

print(f'{traced.graph=}')
# <torch.fx.graph.Graph object at 0x7ad2f02ad2d0>

print()
print(f'traced.code:\n', traced.code.strip())
# def forward(self, x):
#     ffnn_0 = getattr(self.ffnn, "0")(x);  x = None
#     ffnn_1 = getattr(self.ffnn, "1")(ffnn_0);  ffnn_0 = None
#     ffnn_2 = getattr(self.ffnn, "2")(ffnn_1);  ffnn_1 = None
#     return ffnn_2

print()
traced.graph.print_tabular()
# opcode       name    target    args       kwargs
# -----------  ------  --------  ---------  --------
# placeholder  x       x         ()         {}
# call_module  ffnn_0  ffnn.0    (x,)       {}
# call_module  ffnn_1  ffnn.1    (ffnn_0,)  {}
# call_module  ffnn_2  ffnn.2    (ffnn_1,)  {}
# output       output  output    (ffnn_2,)  {}


##################################################
# torch.fx.replace_pattern

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
print()

class M(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        val1 = torch.neg(w1)
        m1 = torch.cat([val1, w2]).sum()
        val2 = torch.neg(w1)
        m2 = torch.cat([val2, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)

traced = fx.symbolic_trace(M())

def pattern(a1, a2):
    val1 = torch.neg(a1)
    return torch.cat([val1, a2]).sum()

def replacement(w1, w2):
    return torch.stack([w1, w2])


matches = fx.replace_pattern(traced, pattern, replacement)  # mutates!

print(traced.code)



##################################################
# ShapeProp example

from torch.fx.node import Node

from typing import Dict

class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph)

original = MyModule()
traced = fx.symbolic_trace(original)
new = ShapeProp(traced)
x = torch.randn(32)
y = new.propagate(x)

for node in traced.graph.nodes:
    print(node, node.shape, node.dtype)
    # print(node.name, node.meta['tensor_meta'].dtype,
    #     node.meta['tensor_meta'].shape)


##################################################
# Control Inversion

'''

Bubble up control of a Linear model's `weight` parameter to the function caller

# Example:

batch_size = 3
x = torch.randn(batch_size, 32)

# normal model usage
model = MyModule()
y = model(x)

# transformed model, to give control of the Linear.weight to the function caller
w = torch.randn(16, 32)
transformed_model = ...
y = transformed_model(x, w)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
torch.manual_seed(152)
print()

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.linear(x)


def transform_linear_weight(module: nn.Module) -> nn.Module:
    # Symbolically trace the module
    traced = fx.symbolic_trace(module)

    # Get the graph
    graph = traced.graph

    # Find the linear layer node
    linear_node = None
    for node in graph.nodes:
        if node.op == 'call_module' and isinstance(getattr(traced, node.target), nn.Linear):
            linear_node = node
            break

    if linear_node is None:
        raise ValueError("No Linear layer found in the module")

    # Create a new input node for the weight
    weight_node = graph.create_node('placeholder', 'weight')

    # Modify the linear layer call to use the external weight
    with graph.inserting_after(linear_node):
        new_args = linear_node.args + (weight_node,)
        new_linear_node = graph.create_node('call_function', F.linear, args=new_args)
        linear_node.replace_all_uses_with(new_linear_node)
        graph.erase_node(linear_node)

    # Create the new GraphModule
    transformed_module = fx.GraphModule(traced, graph)

    return transformed_module

# Usage
batch_size = 3
x = torch.randn(batch_size, 32)

# basic model
model = MyModule()
y = model(x)

# transformed model
transformed_model = transform_linear_weight(model)

# clone the weight initialization since we're going to check that outputs are equivalent
w = model.linear.weight.clone()

# Verify that the transformed model works correctly
y_transformed = transformed_model(x, w)
assert y.shape == y_transformed.shape, "Output shapes are different"
assert torch.allclose(y, y_transformed), "Transformation changed the output"
print("Transformation successful and outputs match!")


##################################################
# Control Inversion, Multiple Linears

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
        )

    def forward(self, x):
        return self.ffnn(x)


def transform_control_inversion(module: nn.Module) -> nn.Module:
    # Symbolically trace the module
    traced = fx.symbolic_trace(module)
    graph = traced.graph

    # Find the linear layer nodes
    linear_nodes = []
    for node in graph.nodes:
        if node.op == 'call_module':
            target_module = traced
            for attr in node.target.split('.'):
                target_module = getattr(target_module, attr)
            if isinstance(target_module, nn.Linear):
                linear_nodes.append(node)

    if len(linear_nodes) != 2:
        raise ValueError(f"Expected 2 Linear layers, found {len(linear_nodes)}")

    # Create new input nodes for weights and biases
    weight1_node = graph.create_node('placeholder', 'weight1')
    bias1_node = graph.create_node('placeholder', 'bias1')
    weight2_node = graph.create_node('placeholder', 'weight2')
    bias2_node = graph.create_node('placeholder', 'bias2')

    # Modify the linear layer calls to use external weights and biases
    for i, node in enumerate(linear_nodes):
        weight_node = weight1_node if i == 0 else weight2_node
        bias_node = bias1_node if i == 0 else bias2_node

        with graph.inserting_after(node):
            new_args = (node.args[0], weight_node, bias_node)
            new_linear_node = graph.create_node('call_function', F.linear, args=new_args)
            node.replace_all_uses_with(new_linear_node)
            graph.erase_node(node)

    # Create the new GraphModule
    transformed_module = fx.GraphModule(traced, graph)
    return transformed_module


# Original module
original_module = MyModule()

# Transformed module
transformed_module = transform_control_inversion(MyModule())

# Input tensor
x = torch.randn(4, 32)

# Get weights and biases from the original module
weight1 = original_module.ffnn[0].weight
bias1 = original_module.ffnn[0].bias
weight2 = original_module.ffnn[2].weight
bias2 = original_module.ffnn[2].bias

# Compute outputs
with torch.no_grad():
    original_output = original_module(x)
    transformed_output = transformed_module(x, weight1, bias1, weight2, bias2)

# Check if outputs are equal
assert torch.allclose(original_output, transformed_output, atol=1e-6), "Outputs are not equal!"
print("Outputs are equal. Control inversion successful!")

# Print the code of the transformed module
print("\nTransformed module code:")
print(transformed_module.code)



##################################################
# Control Inversion, Nested Linears

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
print()

class SomeOtherModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linx = nn.Linear(17, 71)
    def forward(self, x):
        return self.linx(x)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.Sequential(
                nn.Linear(64, 17)
            )
        )
        self.some_other_module = SomeOtherModule()
        self.linear = nn.Linear(71, 43)

    def forward(self, x):
        return self.linear(
            self.some_other_module(
                self.ffnn(x)
            )
        )

def transform_control_inversion(module: nn.Module) -> nn.Module:
    # Symbolically trace the module
    traced = fx.symbolic_trace(module)
    graph = traced.graph

    # Find the linear layer nodes and store their names
    linear_nodes = []
    for node in graph.nodes:
        if node.op == 'call_module':
            target_module = traced
            for attr in node.target.split('.'):
                target_module = getattr(target_module, attr)
            if isinstance(target_module, nn.Linear):
                linear_nodes.append((node, node.target))

    # Create new input nodes for weights and biases at the beginning of the graph
    param_nodes = {}
    input_placeholder = next(iter(graph.nodes))
    with graph.inserting_after(input_placeholder):  # Insert after the input placeholder
        for _, target in linear_nodes:
            weight_name = f'{target.replace(".", "_")}_weight'
            bias_name = f'{target.replace(".", "_")}_bias'
            param_nodes[weight_name] = graph.create_node('placeholder', weight_name)
            param_nodes[bias_name] = graph.create_node('placeholder', bias_name)

    # Modify the linear layer calls to use external weights and biases
    for node, target in linear_nodes:
        weight_node = param_nodes[f'{target.replace(".", "_")}_weight']
        bias_node = param_nodes[f'{target.replace(".", "_")}_bias']

        with graph.inserting_after(node):
            new_args = (node.args[0], weight_node, bias_node)
            new_linear_node = graph.create_node('call_function', F.linear, args=new_args)
            node.replace_all_uses_with(new_linear_node)
            graph.erase_node(node)

    # Create the new GraphModule
    graph.lint()
    transformed_module = fx.GraphModule(traced, graph)
    return transformed_module


# Original module
original_module = MyModule()

# Transformed module
transformed_module = transform_control_inversion(MyModule())

# Input tensor
x = torch.randn(4, 32)

# Get weights and biases from the original module
params = {}
for name, module in original_module.named_modules():
    if isinstance(module, nn.Linear):
        params[f'{name.replace(".", "_")}_weight'] = module.weight
        params[f'{name.replace(".", "_")}_bias'] = module.bias

# Compute outputs
with torch.no_grad():
    original_output = original_module(x)
    transformed_output = transformed_module(x, **params)

# Check if outputs are equal
assert torch.allclose(original_output, transformed_output, atol=1e-6), "Outputs are not equal!"
print("Outputs are equal. Control inversion successful!")

# Print the code of the transformed module
print("\nTransformed module code:")
print(transformed_module.code)


##################################################
# More complex: LayerNorm, Parameter, ParameterList, ParameterDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import operator
print()

class SomeOtherModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linx = nn.Linear(17, 71)
        self.norm = nn.LayerNorm(71)
        self.custom_param = nn.Parameter(torch.randn(71))
    def forward(self, x):
        return self.norm(self.linx(x)) + self.custom_param

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 32)
        self.ffnn = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.LayerNorm(64),
            nn.Sequential(
                nn.Linear(64, 17)
            )
        )
        self.some_other_module = SomeOtherModule()
        self.linear = nn.Linear(71, 43)
        self.param_list = nn.ParameterList([nn.Parameter(torch.randn(43)) for _ in range(3)])
        self.param_dict = nn.ParameterDict({
            'a': nn.Parameter(torch.randn(43)),
            'b': nn.Parameter(torch.randn(43))
        })

    def forward(self, x):
        x = self.embed(x)
        x = self.linear(
            self.some_other_module(
                self.ffnn(x)
            )
        )
        x = x + self.param_list[0] + self.param_list[1] + self.param_list[2]
        x = x + self.param_dict['a'] + self.param_dict['b']
        return x



def transform_control_inversion(module: nn.Module) -> nn.Module:
    # Symbolically trace the module
    traced = fx.symbolic_trace(module)
    graph = traced.graph

    # Find the parameter-containing layer nodes and store their names
    param_layers = []
    for node in graph.nodes:
        if node.op == 'call_module':
            target_module = traced
            for attr in node.target.split('.'):
                target_module = getattr(target_module, attr)
            if isinstance(target_module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                param_layers.append((node, node.target, type(target_module)))
        elif node.op == 'get_attr':
            target_attr = traced
            for attr in node.target.split('.'):
                target_attr = getattr(target_attr, attr)
            if isinstance(target_attr, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                param_layers.append((node, node.target, type(target_attr)))

    # Create new input nodes for parameters at the beginning of the graph
    param_nodes = {}
    input_placeholder = next(iter(graph.nodes))
    with graph.inserting_after(input_placeholder):
        for node, target, layer_type in param_layers:
            if layer_type in (nn.Linear, nn.LayerNorm):
                param_nodes[f'{target.replace(".", "_")}_weight'] = graph.create_node('placeholder', f'{target.replace(".", "_")}_weight')
                param_nodes[f'{target.replace(".", "_")}_bias'] = graph.create_node('placeholder', f'{target.replace(".", "_")}_bias')
            elif layer_type == nn.Embedding:
                param_nodes[f'{target.replace(".", "_")}_weight'] = graph.create_node('placeholder', f'{target.replace(".", "_")}_weight')
            elif layer_type == nn.Parameter:
                param_nodes[target.replace(".", "_")] = graph.create_node('placeholder', target.replace(".", "_"))
            elif layer_type in (nn.ParameterList, nn.ParameterDict):
                target_attr = traced
                for attr in target.split('.'):
                    target_attr = getattr(target_attr, attr)
                for idx, (key, _) in enumerate(target_attr.named_parameters()):
                    param_name = f'{target.replace(".", "_")}_{key if isinstance(target_attr, nn.ParameterDict) else idx}'
                    param_nodes[param_name] = graph.create_node('placeholder', param_name)

    # Modify the layer calls to use external parameters
    for node, target, layer_type in param_layers:
        if layer_type == nn.Linear:
            weight_node = param_nodes[f'{target.replace(".", "_")}_weight']
            bias_node = param_nodes[f'{target.replace(".", "_")}_bias']
            with graph.inserting_after(node):
                new_args = (node.args[0], weight_node, bias_node)
                new_node = graph.create_node('call_function', F.linear, args=new_args)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
        elif layer_type == nn.LayerNorm:
            weight_node = param_nodes[f'{target.replace(".", "_")}_weight']
            bias_node = param_nodes[f'{target.replace(".", "_")}_bias']
            with graph.inserting_after(node):
                original_module = traced
                for attr in target.split('.'):
                    original_module = getattr(original_module, attr)
                new_args = (node.args[0], original_module.normalized_shape)
                new_kwargs = {
                    'weight': weight_node,
                    'bias': bias_node,
                    'eps': original_module.eps,
                }
                new_node = graph.create_node('call_function', F.layer_norm, args=new_args, kwargs=new_kwargs)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
        elif layer_type == nn.Embedding:
            weight_node = param_nodes[f'{target.replace(".", "_")}_weight']
            with graph.inserting_after(node):
                original_module = traced
                for attr in target.split('.'):
                    original_module = getattr(original_module, attr)
                new_args = (node.args[0], weight_node)
                new_kwargs = {'padding_idx': original_module.padding_idx}
                new_node = graph.create_node('call_function', F.embedding, args=new_args, kwargs=new_kwargs)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
        elif layer_type in (nn.Parameter, nn.ParameterList, nn.ParameterDict):
            if layer_type == nn.Parameter:
                new_node = param_nodes[target.replace(".", "_")]
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
            else:
                # For ParameterList and ParameterDict, we need to replace the getitem calls
                for user in node.users:
                    if user.op == 'call_function' and user.target in (operator.getitem, dict.__getitem__):
                        idx_or_key = user.args[1]
                        if isinstance(idx_or_key, str):
                            new_node = param_nodes[f'{target.replace(".", "_")}_{idx_or_key}']
                        else:
                            new_node = param_nodes[f'{target.replace(".", "_")}_{idx_or_key}']
                        user.replace_all_uses_with(new_node)
                        graph.erase_node(user)
                graph.erase_node(node)

    # Create the new GraphModule
    graph.lint()
    transformed_module = fx.GraphModule(traced, graph)
    return transformed_module


# Original module
original_module = MyModule()

# Transformed module
transformed_module = transform_control_inversion(MyModule())

# Input tensor
x = torch.randint(0, 1000, (4, 10))  # 4 sequences of 10 tokens each

# Get parameters from the original module
params = {}
for name, param in original_module.named_parameters():
    params[name.replace(".", "_")] = param

# Compute outputs
with torch.no_grad():
    original_output = original_module(x)
    transformed_output = transformed_module(x, **params)

# Check if outputs are equal
assert torch.allclose(original_output, transformed_output, atol=1e-6), "Outputs are not equal!"
print("Outputs are equal. Control inversion successful!")

# Print the code of the transformed module
print("\nTransformed module code:")
print(transformed_module.code)




##################################################
# Add Convolutions

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


def transform_control_inversion(module: nn.Module) -> nn.Module:
    # Symbolically trace the module
    traced = fx.symbolic_trace(module)
    graph = traced.graph

    # Find the parameter-containing layer nodes and store their names
    param_layers = []
    for node in graph.nodes:
        if node.op == 'call_module':
            target_module = traced
            for attr in node.target.split('.'):
                target_module = getattr(target_module, attr)
            if isinstance(target_module, (nn.Linear, nn.LayerNorm, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                param_layers.append((node, node.target, type(target_module)))
        elif node.op == 'get_attr':
            target_attr = traced
            for attr in node.target.split('.'):
                target_attr = getattr(target_attr, attr)
            if isinstance(target_attr, (nn.Parameter, nn.ParameterList, nn.ParameterDict)):
                param_layers.append((node, node.target, type(target_attr)))

    # Create new input nodes for parameters at the beginning of the graph
    param_nodes = {}

    # Find final input placeholder, to insert new placeholders after
    input_placeholders = [node for node in graph.nodes if node.op == 'placeholder']
    input_placeholder = input_placeholders[-1]

    with graph.inserting_after(input_placeholder):
        for node, target, layer_type in param_layers:
            if layer_type in (nn.Linear, nn.LayerNorm, nn.Conv1d, nn.Conv2d, nn.Conv3d):
                param_nodes[f'{target}.weight'] = graph.create_node('placeholder', f'{target.replace(".", "_")}_weight')
                param_nodes[f'{target}.bias'] = graph.create_node('placeholder', f'{target.replace(".", "_")}_bias')
            elif layer_type == nn.Embedding:
                param_nodes[f'{target}.weight'] = graph.create_node('placeholder', f'{target.replace(".", "_")}_weight')
            elif layer_type == nn.Parameter:
                param_nodes[target] = graph.create_node('placeholder', target.replace(".", "_"))
            elif layer_type in (nn.ParameterList, nn.ParameterDict):
                target_attr = traced
                for attr in target.split('.'):
                    target_attr = getattr(target_attr, attr)
                for idx, (key, _) in enumerate(target_attr.named_parameters()):
                    param_name = f'{target}.{key if isinstance(target_attr, nn.ParameterDict) else idx}'
                    param_nodes[param_name] = graph.create_node('placeholder', f'{target.replace(".", "_")}_{key if isinstance(target_attr, nn.ParameterDict) else idx}')

    # Modify the layer calls to use external parameters
    for node, target, layer_type in param_layers:
        if layer_type == nn.Linear:
            weight_node = param_nodes[f'{target}.weight']
            bias_node = param_nodes[f'{target}.bias']
            with graph.inserting_after(node):
                new_args = (node.args[0], weight_node, bias_node)
                new_node = graph.create_node('call_function', F.linear, args=new_args)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
        elif layer_type == nn.LayerNorm:
            weight_node = param_nodes[f'{target}.weight']
            bias_node = param_nodes[f'{target}.bias']
            with graph.inserting_after(node):
                original_module = traced
                for attr in target.split('.'):
                    original_module = getattr(original_module, attr)
                new_args = (node.args[0], original_module.normalized_shape)
                new_kwargs = {
                    'weight': weight_node,
                    'bias': bias_node,
                    'eps': original_module.eps,
                }
                new_node = graph.create_node('call_function', F.layer_norm, args=new_args, kwargs=new_kwargs)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
        elif layer_type == nn.Embedding:
            weight_node = param_nodes[f'{target}.weight']
            with graph.inserting_after(node):
                original_module = traced
                for attr in target.split('.'):
                    original_module = getattr(original_module, attr)
                new_args = (node.args[0], weight_node)
                new_kwargs = {'padding_idx': original_module.padding_idx}
                new_node = graph.create_node('call_function', F.embedding, args=new_args, kwargs=new_kwargs)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
        elif layer_type in (nn.Conv1d, nn.Conv2d, nn.Conv3d):
            weight_node = param_nodes[f'{target}.weight']
            bias_node = param_nodes[f'{target}.bias']
            with graph.inserting_after(node):
                original_module = traced
                for attr in target.split('.'):
                    original_module = getattr(original_module, attr)
                new_args = (node.args[0], weight_node)
                new_kwargs = {
                    'bias': bias_node,
                    'stride': original_module.stride,
                    'padding': original_module.padding,
                    'dilation': original_module.dilation,
                    'groups': original_module.groups
                }
                if layer_type == nn.Conv1d:
                    conv_func = F.conv1d
                elif layer_type == nn.Conv2d:
                    conv_func = F.conv2d
                else:  # nn.Conv3d
                    conv_func = F.conv3d
                new_node = graph.create_node('call_function', conv_func, args=new_args, kwargs=new_kwargs)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
        elif layer_type in (nn.Parameter, nn.ParameterList, nn.ParameterDict):
            if layer_type == nn.Parameter:
                new_node = param_nodes[target]
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
            else:
                # For ParameterList and ParameterDict, we need to replace the getitem calls
                for user in node.users:
                    if user.op == 'call_function' and user.target in (operator.getitem, dict.__getitem__):
                        idx_or_key = user.args[1]
                        if isinstance(idx_or_key, str):
                            new_node = param_nodes[f'{target}.{idx_or_key}']
                        else:
                            new_node = param_nodes[f'{target}.{idx_or_key}']
                        user.replace_all_uses_with(new_node)
                        graph.erase_node(user)
                graph.erase_node(node)

    # Create the new GraphModule
    graph.lint()
    transformed_module = fx.GraphModule(traced, graph)
    return transformed_module

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
params = {name.replace('.', '_'): param for name, param in original_module.named_parameters()}

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
