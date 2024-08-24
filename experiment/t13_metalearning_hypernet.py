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
from typing import Dict, Any, Union, List, Tuple, Optional

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

from neurallambda.lab.common import print_model_info
import t13_metalearning_hypernet_data as data


import importlib

try:
    importlib.reload(AE)
    importlib.reload(Flatten)
    print('RELOADING MODULE')
except NameError:
    import t13_metalearning_hypernet_autoencoder as AE
    import t13_metalearning_hypernet_flatten as Flatten

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################################
# Metalearning Checks

def assert_no_global_mutation(model, forward_pass_fn, x):
    """
    Asserts that the forward pass (which includes the inner loop)
    doesn't mutate the global state of the model.

    :param model: The PyTorch model (Thing class in this case)
    :param forward_pass_fn: A function that performs the forward pass
    :param x: Input tensor for the forward pass
    """
    # Store initial state of ae_params
    initial_state = {name: param.clone().detach() for name, param in model.ae_params.items()}

    # Perform the forward pass
    forward_pass_fn(x)

    for name, param in model.ae_params.items():
        assert torch.allclose(param, initial_state[name], atol=1e-6), f"Parameter {name} was mutated during the inner loop"

    print("Assertion passed: No global mutation occurred during the inner loop.")

def assert_outer_loop_backprop(model, forward_pass_fn, loss_fn, x):
    """
    Asserts that the outer loop can backpropagate to the original parameters.

    :param model: The PyTorch model (Thing class in this case)
    :param forward_pass_fn: A function that performs the forward pass
    :param loss_fn: A function that computes the loss
    :param x: Input tensor for the forward pass
    """
    # Enable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Perform forward pass and compute loss
    output = forward_pass_fn(x)
    loss = loss_fn(output, x)

    # Compute gradients
    loss.backward()

    # Check if gradients exist for all parameters in ae_params
    for name, param in model.ae_params.items():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert torch.any(param.grad != 0), f"Zero gradient for parameter {name}"

    print("Assertion passed: Outer loop can backpropagate to all original parameters.")

def metalearning_checks(model, x):
    def forward_pass(input_x):
        return model(input_x)

    def loss_fn(output, target):
        return F.mse_loss(output, target)

    assert_no_global_mutation(model, forward_pass, x)
    assert_outer_loop_backprop(model, forward_pass, loss_fn, x)


##################################################

class Thing(nn.Module):
    def __init__(self, in_channels, img_size, bottleneck_size, kernel_size, pool_size, layer_config, actfn='swish', padding_size=1):
        super().__init__()
        self.ae = AE.Autoencoder(
            in_channels=in_channels,
            img_size=img_size,
            bottleneck_size=bottleneck_size,
            kernel_size=kernel_size,
            pool_size=pool_size,
            layer_config=layer_config,
        )

        self.ae_params = nn.ParameterDict(Flatten.build_params_dict(self.ae))  # ParamDict so outerloop sees the params

        # Create control-inverted version of `ae`. It is also "batchified",
        # meaning that the params passed in have been copied on a per row
        # basis.
        self.ae = Flatten.transform_control_inversion(self.ae, use_batchify=True)  # overwrite original module

        self.lr_inner = 1e-2

    def forward(self, x):
        B = x.shape[0]

        # Mask the images
        mask_ratio = 0.2
        if self.training:
            mask = torch.rand(x.shape, device=x.device) < mask_ratio
            mx = x.clone()
            with torch.no_grad():
                mx[mask] = 1  # 1=white
        else:
            mx = x

        # Metalearning
        with torch.enable_grad():
            params_copy = self.ae_params
            params_copy = {k: v.unsqueeze(0).repeat(B, *[1] * v.ndim) for k, v in params_copy.items()}
            # params_copy = {k: v + 0 for k, v in params_copy.items()}

            keys, parameters = zip(*params_copy.items())
            output = self.ae(mx, **params_copy)
            loss = F.mse_loss(output, x)
            grads = torch.autograd.grad(loss, parameters, create_graph=True)
            updated_params = [p - g * self.lr_inner for p, g in zip(parameters, grads)]

            for k, up in zip(keys, updated_params):
                params_copy[k] = up

        # Predict from metalearning-improved params
        final_output = self.ae(mx, **params_copy)
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

batch_size = 71
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

# metalearning assertions
if False:  # turn off bc the global step does mutate params
    raise Exception('these checks dont yet support batchify')
    x = torch.randn(64, 1, img_dim, img_dim, device=device)
    metalearning_checks(model, x)

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
AE.visualize_reconstructions(model.ae, model.ae_params, val_dl)




##############################

# START_BLOCK_1

# END_BLOCK_1
