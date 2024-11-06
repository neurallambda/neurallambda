import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
from copy import deepcopy
import math
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

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

if False:
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
