'''

can a simple MLP output weight updates that help it in-context?

steps:
1. pretrain autoencoder  (t13_metalearning_hypernet_autoencoder_functional)
2. metalearner (t13_metalearning_hypernet_metalearner)
3. hypernetwork (this module ties everything together)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
from copy import deepcopy
import math
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Union

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from neurallambda.lab.common import print_model_info
import t13_metalearning_hypernet_data as data
import t13_metalearning_hypernet_autoencoder_functional as AE

torch.manual_seed(152)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_params(params: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> Tuple[List[Tuple[str, ...]], List[torch.nn.Parameter]]:
    keys = []
    parameters = []
    for k, v in params.items():
        if isinstance(v, (dict, nn.ParameterDict)):
            sub_keys, sub_params = get_params(v, prefix + (k,))
            keys.extend(sub_keys)
            parameters.extend(sub_params)
        elif isinstance(v, (list, nn.ParameterList)):
            for i, item in enumerate(v):
                if isinstance(item, torch.nn.Parameter):
                    keys.append(prefix + (k, i))
                    parameters.append(item)
                elif isinstance(item, (dict, nn.ParameterDict, list, nn.ParameterList)):
                    sub_keys, sub_params = get_params({i: item}, prefix + (k,))
                    keys.extend(sub_keys)
                    parameters.extend(sub_params)
        elif isinstance(v, torch.nn.Parameter):
            keys.append(prefix + (k,))
            parameters.append(v)
    return keys, parameters

def set_params(params: Dict[str, Any], keys: List[Tuple[str, ...]], new_params: List[torch.nn.Parameter]) -> None:
    for key, new_param in zip(keys, new_params):
        current = params
        for k in key[:-1]:
            current = current[k]
        current[key[-1]] = new_param

class Thing(nn.Module):
    def __init__(self, in_channels, img_size, bottleneck_size, kernel_size, pool_size, layer_config, actfn='swish', padding_size=1):
        super().__init__()
        self.ae_params = AE.ae_init(
            in_channels=in_channels,
            img_size=img_size,
            bottleneck_size=bottleneck_size,
            kernel_size=kernel_size,
            pool_size=pool_size,
            layer_config=layer_config,
            device=device
        )
        self.lr_inner = 1e-2

    def forward(self, x):
        # Mask the images
        mask_ratio = 0.2
        if self.training:
            mask = torch.rand(x.shape, device=x.device) < mask_ratio
            mx = x.clone()
            with torch.no_grad():
                mx[mask] = 1  # 1=white
        else:
            mx = x

        with torch.enable_grad():
            params_copy = self.ae_params.copy()
            keys, parameters = get_params(params_copy)
            output = AE.ae_forward(params_copy, mx)
            loss = F.mse_loss(output, x)
            grads = torch.autograd.grad(loss, parameters, create_graph=True)
            updated_params = [p - g * self.lr_inner for p, g in zip(parameters, grads)]
            set_params(params_copy, keys, updated_params)

        final_output = AE.ae_forward(params_copy, mx)
        return final_output


def run_epoch(model, dataloader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    criterion = nn.MSELoss()

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):
            imgs, alphabets, labels = batch
            imgs = imgs.to(device).float()
            imgs = imgs.unsqueeze(1)  # add channel dim=1
            B = imgs.shape[0]
            reconstructed = model(imgs)
            loss = criterion(reconstructed, imgs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples
    return avg_loss


##########
# Go

batch_size = 64
num_epochs = 20
lr = 2e-3
wd = 0.0
mask_ratio = 0.2  # 0.1 = mask 10%

in_channels = 1
img_dim = 32
bottleneck_size = 32
kernel_size = 3
pool_size = 2

layer_config = [
    (4, True),
    (4, True),
    (4, True),
    (4, True),
    (4, True),
    (4, False),
]

model = Thing(
    in_channels=in_channels,
    img_size=img_dim,
    bottleneck_size=bottleneck_size,
    kernel_size=kernel_size,
    pool_size=pool_size,
    layer_config=layer_config
)

model.to(device)

print_model_info(model)

# Load Omniglot data
train_alphabets = ["Latin", "Greek", "Cyrillic"]
test_alphabets = ["Mongolian"]
train_dl, val_dl = data.omniglot_dataloader(train_alphabets, test_alphabets, image_size=img_dim, batch_size=batch_size)

# Train the model
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

for epoch in range(num_epochs):
    train_loss = run_epoch(model, train_dl, optimizer, device, train=True)
    val_loss = run_epoch(model, val_dl, optimizer, device, train=False)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

# Visualize reconstructions
print("Visualizing reconstructions on validation set...")
AE.visualize_reconstructions(model.ae_params, val_dl, device)




##################################################


# START_BLOCK_1

import torch
import torch.nn as nn

# Create the Object as a top-level variable
Object = nn.ParameterDict({
    'layer1': nn.ParameterDict({
        'weight': nn.Parameter(torch.randn(3, 3)),
        'bias': nn.Parameter(torch.randn(3))
    }),
    'layer2': nn.ParameterList([
        nn.Parameter(torch.randn(2, 2)),
        nn.ParameterList([
            nn.Parameter(torch.randn(1)),
            nn.Parameter(torch.randn(1))
        ]),
        nn.ParameterDict({
            'nested': nn.Parameter(torch.randn(2, 2))
        })
    ])
})


##########
# GET
keys, parameters = get_params(Object)

print("Keys:")
for key in keys:
    print(key)

print("\nNumber of parameters:", len(parameters))

##########
# SET

# Create new parameters with the same shapes as the original ones
new_params = [torch.nn.Parameter(torch.randn_like(param)) for param in parameters]

# Set the new parameters
set_params(Object, keys, new_params)

# Verify that the parameters have been updated
for key, new_param in zip(keys, new_params):
    current = Object
    for k in key[:-1]:
        current = current[k]
    assert torch.all(current[key[-1]] == new_param), f"Parameter at {key} was not updated correctly"

print("All parameters were successfully updated!")


##########
# NELEM

from typing import Union, Dict, List

def count_nelem(obj: Union[nn.ParameterDict, nn.ParameterList, nn.Parameter, Dict, List]) -> int:
    total_elements = 0

    if isinstance(obj, (nn.ParameterDict, dict)):
        for value in obj.values():
            total_elements += count_nelem(value)
    elif isinstance(obj, (nn.ParameterList, list)):
        for item in obj:
            total_elements += count_nelem(item)
    elif isinstance(obj, nn.Parameter):
        total_elements += obj.numel()
    else:
        raise TypeError(f"Unexpected type: {type(obj)}")

    return total_elements

# Example usage:
nelem = count_nelem(Object)
print(f"Total number of elements in Object: {nelem}")


##########
# FLATTEN/UNFLATTEN

def flatten_object(obj: Union[nn.ParameterDict, Dict[str, Any]]) -> Tuple[torch.Tensor, List[Tuple[Tuple[str, ...], torch.Size]]]:
    # Get the parameters in a deterministic order
    keys, parameters = get_params(obj)

    # Flatten each parameter and add to a list
    flattened_params = []
    param_info = []

    for key, param in zip(keys, parameters):
        flattened_params.append(param.flatten())
        param_info.append((key, param.size()))

    # Concatenate all flattened parameters into a single vector
    flattened_vector = torch.cat(flattened_params)

    return flattened_vector, param_info

def unflatten_object(flattened_vector: torch.Tensor, param_info: List[Tuple[Tuple[str, ...], torch.Size]], obj_template: Union[nn.ParameterDict, Dict[str, Any]]) -> Union[nn.ParameterDict, Dict[str, Any]]:
    new_obj = type(obj_template)()
    index = 0

    for (key, size) in param_info:
        num_elements = torch.prod(torch.tensor(size)).item()
        param_values = flattened_vector[index:index + num_elements].view(size)
        new_param = nn.Parameter(param_values)

        current = new_obj
        for i, k in enumerate(key[:-1]):
            if isinstance(current, (dict, nn.ParameterDict)):
                if k not in current:
                    if isinstance(obj_template[k], nn.ParameterList):
                        current[k] = nn.ParameterList()
                    elif isinstance(obj_template[k], nn.ParameterDict):
                        current[k] = nn.ParameterDict()
                    else:
                        current[k] = {}
                current = current[k]
            elif isinstance(current, (list, nn.ParameterList)):
                while len(current) <= k:
                    if i == len(key) - 2 and isinstance(key[-1], str):
                        current.append(nn.ParameterDict())
                    else:
                        current.append(nn.ParameterList())
                current = current[k]

        if isinstance(current, (list, nn.ParameterList)):
            current.append(new_param)
        else:
            current[key[-1]] = new_param

        index += num_elements

    return new_obj


Object_ = {
    'layer1': {
        'weight': torch.randn(3, 3, requires_grad=True),
        'bias': torch.randn(3, requires_grad=True)
    },
    'layer2': [
        torch.randn(2, 2, requires_grad=True),
        [
            torch.randn(1, requires_grad=True),
            torch.randn(1, requires_grad=True)
        ],
        {
            'nested': torch.randn(2, 2, requires_grad=True)
        }
    ]
}


def flatten_object_(obj: Union[Dict[str, Any], List[Any]]) -> Tuple[torch.Tensor, List[Tuple[Tuple[Union[str, int], ...], torch.Size]]]:
    def get_params_(obj: Union[Dict[str, Any], List[Any]], prefix: Tuple[Union[str, int], ...] = ()) -> Tuple[List[Tuple[Union[str, int], ...]], List[torch.Tensor]]:
        ''' like `get_params` above but a non ParameterContainer version '''
        keys = []
        tensors = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert not isinstance(v, (nn.Parameter, nn.ParameterDict, nn.ParameterList)), f"Found nn.Parameter or ParameterContainer at {prefix + (k,)}"
                if isinstance(v, (dict, list)):
                    sub_keys, sub_tensors = get_params_(v, prefix + (k,))
                    keys.extend(sub_keys)
                    tensors.extend(sub_tensors)
                elif isinstance(v, torch.Tensor):
                    keys.append(prefix + (k,))
                    tensors.append(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                assert not isinstance(v, (nn.Parameter, nn.ParameterDict, nn.ParameterList)), f"Found nn.Parameter or ParameterContainer at {prefix + (i,)}"
                if isinstance(v, (dict, list)):
                    sub_keys, sub_tensors = get_params_(v, prefix + (i,))
                    keys.extend(sub_keys)
                    tensors.extend(sub_tensors)
                elif isinstance(v, torch.Tensor):
                    keys.append(prefix + (i,))
                    tensors.append(v)
        return keys, tensors

    keys, tensors = get_params_(obj)

    flattened_tensors = []
    param_info = []

    for key, tensor in zip(keys, tensors):
        flattened_tensors.append(tensor.flatten())
        param_info.append((key, tensor.size()))

    flattened_vector = torch.cat(flattened_tensors)

    return flattened_vector, param_info

def unflatten_object_(flattened_vector: torch.Tensor, param_info: List[Tuple[Tuple[Union[str, int], ...], torch.Size]], obj_template: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    assert not isinstance(obj_template, (nn.Parameter, nn.ParameterDict, nn.ParameterList)), "obj_template should not be nn.Parameter or ParameterContainer"

    new_obj = type(obj_template)()
    index = 0

    for (key, size) in param_info:
        num_elements = torch.prod(torch.tensor(size)).item()
        tensor_values = flattened_vector[index:index + num_elements].view(size)

        current = new_obj
        for i, k in enumerate(key[:-1]):
            if isinstance(current, dict):
                if k not in current:
                    if isinstance(obj_template[k], list):
                        current[k] = []
                    elif isinstance(obj_template[k], dict):
                        current[k] = {}
                    else:
                        current[k] = {}
                current = current[k]
            elif isinstance(current, list):
                while len(current) <= k:
                    current.append({} if i == len(key) - 2 and isinstance(key[-1], str) else [])
                current = current[k]

        if isinstance(current, list):
            current.append(tensor_values)
        else:
            current[key[-1]] = tensor_values

        index += num_elements

    return new_obj

# Example usage:
flattened_vector, param_info = flatten_object(Object)
print(f"Flattened vector shape: {flattened_vector.shape}")
print(f"Number of elements: {flattened_vector.numel()}")

# Unflatten the vector back into an object
flattened_vector = torch.cat([flattened_vector, torch.arange(12)], dim=0)  # add cruft data on for fun
reconstructed_obj = unflatten_object(flattened_vector, param_info, Object)

# Verify that the reconstructed object has the same structure and values
def verify_objects(obj1, obj2):
    if isinstance(obj1, (nn.ParameterDict, dict)):
        assert set(obj1.keys()) == set(obj2.keys()), "Keys don't match"
        for k in obj1.keys():
            verify_objects(obj1[k], obj2[k])
    elif isinstance(obj1, (nn.ParameterList, list)):
        assert len(obj1) == len(obj2), "Lengths don't match"
        for i in range(len(obj1)):
            verify_objects(obj1[i], obj2[i])
    elif isinstance(obj1, nn.Parameter):
        assert torch.all(obj1 == obj2), "Parameter values don't match"
    else:
        raise TypeError(f"Unexpected type: {type(obj1)}")

verify_objects(Object, reconstructed_obj)
print("Verification passed: Original and reconstructed objects match!")


##########
# CHECK GRADS

# Step 1: Ensure all parameters require gradients
for param in Object.parameters():
    param.requires_grad = True

# Step 2: Flatten the object
flattened_vector, param_info = flatten_object(Object)
reconstructed_obj = unflatten_object_(flattened_vector, param_info, Object_)
flattened_vector, param_info = flatten_object_(reconstructed_obj)

# Step 3: Compute random thing as "loss"
loss = flattened_vector.prod()

# Step 4: Backpropagate
loss.backward()

# Step 5: Check that all parameters have non-zero gradients
all_grads_nonzero = True
for key, param in Object.named_parameters():
    if param.grad is None or torch.all(param.grad == 0):
        print(f"Parameter {key} has no gradient or all-zero gradient")
        all_grads_nonzero = False

if all_grads_nonzero:
    print("All parameters have non-zero gradients. Backpropagation successful!")
else:
    print("Some parameters have no gradients or all-zero gradients. Backpropagation failed.")

# Optional: Print out the gradients for inspection
for key, param in Object.named_parameters():
    print(f"Gradient for {key}:")
    print(param.grad)
    print()

# Reset gradients for future operations
Object.zero_grad()

# END_BLOCK_1
