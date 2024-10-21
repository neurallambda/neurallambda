'''

DEPRECATED(?): the torch.fx work can derive this, in `t13_metalearning_hypernet_flatten`

A functional (not `nn.Module`) version of t13_metalearning_hypernet_autoencoder.py

NOTE: this file is DOWNSTREAM of the original nn.Module version, so, for sake of syncing changes, upgrades should probably happen there first


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

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from neurallambda.lab.common import print_model_info
import t13_metalearning_hypernet_data as data

torch.manual_seed(152)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ae_init(in_channels, img_size, bottleneck_size, kernel_size, pool_size, layer_config, actfn='swish', padding_size=1, device='cpu'):
    assert actfn in {'relu', 'swish'}

    params = nn.ParameterDict({
        'conv_weights': nn.ParameterList(),
        'conv_bias_encoder': nn.ParameterList(),
        'conv_bias_decoder': nn.ParameterList(),
    })

    # Store non-tensor values
    params['metadata'] = {
        'in_channels': in_channels,
        'img_size': img_size,
        'bottleneck_size': bottleneck_size,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'layer_config': layer_config,
        'actfn': actfn,
        'padding_size': padding_size,
        'use_pool': []
    }

    current_channels = in_channels
    current_size = img_size + 2 * padding_size  # Account for initial padding
    for out_channels, use_pool in layer_config:
        # He initialization for conv weights
        conv_weight = nn.Parameter(torch.empty(out_channels, current_channels, kernel_size, kernel_size, device=device))
        torch.nn.init.kaiming_normal_(conv_weight, mode='fan_out', nonlinearity='relu')
        params['conv_weights'].append(conv_weight)

        # Initialize biases to zero
        params['conv_bias_encoder'].append(nn.Parameter(torch.zeros(out_channels, device=device)))
        params['conv_bias_decoder'].append(nn.Parameter(torch.zeros(current_channels, device=device)))

        params['metadata']['use_pool'].append(use_pool)
        current_channels = out_channels
        if use_pool:
            current_size = current_size // 2

    # Linear projection of latents
    linear_in_features = current_channels * current_size * current_size
    params['linear_weights'] = nn.Parameter(torch.empty(bottleneck_size, linear_in_features, device=device))
    torch.nn.init.xavier_uniform_(params['linear_weights'])
    params['linear_bias_encoder'] = nn.Parameter(torch.zeros(bottleneck_size, device=device))
    params['linear_bias_decoder'] = nn.Parameter(torch.zeros(linear_in_features, device=device))

    return params

def encode(params, x, use_skip_connections=True):
    encoded = x
    if params['metadata']['padding_size'] > 0:
        encoded = F.pad(encoded, (params['metadata']['padding_size'],) * 4, mode='reflect')
    pool_indices = []
    skip_connections = []
    shapes = [encoded.shape]  # Store initial shape

    for conv_weight, conv_bias, use_pool in zip(params['conv_weights'], params['conv_bias_encoder'], params['metadata']['use_pool']):
        encoded = F.conv2d(encoded, conv_weight, conv_bias, padding=1)
        if params['metadata']['actfn'] == 'relu':
            encoded = F.relu(encoded)
        elif params['metadata']['actfn'] == 'swish':
            encoded = F.silu(encoded)
        if use_skip_connections:
            skip_connections.append(encoded)
        if use_pool:
            encoded, indices = F.max_pool2d(encoded, params['metadata']['pool_size'], return_indices=True)
            pool_indices.append(indices)
        else:
            pool_indices.append(None)
        shapes.append(encoded.shape)  # Store shape after each layer

    last_conv_shape = encoded.shape[1:]
    flattened = encoded.view(encoded.size(0), -1)
    encoded = F.linear(flattened, params['linear_weights'], params['linear_bias_encoder'])
    if params['metadata']['actfn'] == 'relu':
        encoded = F.relu(encoded)
    elif params['metadata']['actfn'] == 'swish':
        encoded = F.silu(encoded)

    return encoded, pool_indices, last_conv_shape, skip_connections, shapes

def decode(params, encoded, pool_indices, last_conv_shape, skip_connections=None, shapes=None):
    decoded = F.linear(encoded, params['linear_weights'].t(), params['linear_bias_decoder'])
    decoded = decoded.view(-1, *last_conv_shape)

    for i, (conv_weight, conv_bias, use_pool, indices) in enumerate(zip(
        reversed(params['conv_weights']),
        reversed(params['conv_bias_decoder']),
        reversed(params['metadata']['use_pool']),
        reversed(pool_indices)
    )):
        if use_pool:
            decoded = F.max_unpool2d(decoded, indices, params['metadata']['pool_size'], output_size=shapes[-(i + 2)][2:])
        if skip_connections and i < len(skip_connections):
            decoded = decoded + skip_connections[-(i + 1)]
        decoded = F.conv_transpose2d(decoded, conv_weight, conv_bias, padding=1)
        if i < len(params['conv_weights']) - 1:
            if params['metadata']['actfn'] == 'relu':
                decoded = F.relu(decoded)
            elif params['metadata']['actfn'] == 'swish':
                decoded = F.silu(decoded)

    # Remove padding
    if params['metadata']['padding_size'] > 0:
        decoded = decoded[:, :, params['metadata']['padding_size']:-params['metadata']['padding_size'], params['metadata']['padding_size']:-params['metadata']['padding_size']]
    return decoded

def ae_forward(params, x):
    encoded, pool_indices, last_conv_shape, skip_connections, shapes = encode(params, x, use_skip_connections=True)
    decoded = decode(params, encoded, pool_indices, last_conv_shape, skip_connections, shapes)

    # Note: imgs are normalized to [-1, 1]
    decoded = decoded.tanh()
    return decoded


##########


def run_epoch(params, dataloader, optimizer, device, mask_ratio, train=True):
    params.train() if train else params.eval()

    total_loss = 0
    total_samples = 0

    criterion = nn.MSELoss()

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):
            imgs, alphabets, labels = batch
            imgs = imgs.to(device).float()
            imgs = imgs.unsqueeze(1)  # add channel dim=1
            B = imgs.shape[0]

            # Mask the images
            if train:
                mask = torch.rand(imgs.shape, device=device) < mask_ratio
                masked_imgs = imgs.clone()
                masked_imgs[mask] = 1  # 1=white
            else:
                masked_imgs = imgs

            reconstructed = ae_forward(params, masked_imgs)
            loss = criterion(reconstructed, imgs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples
    return avg_loss

def visualize_reconstructions(params, dataloader, device, num_images=10):
    with torch.no_grad():
        # Get a batch of images
        images, _, _ = next(iter(dataloader))
        images = images.to(device).float().unsqueeze(1)  # add channel dim

        # Get reconstructions
        reconstructions = ae_forward(params, images)

        # Plot original and reconstructed images
        fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
        for i in range(num_images):
            # Original image
            axes[0, i].imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

            # Reconstructed image
            axes[1, i].imshow(reconstructions[i].squeeze().cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')

        plt.tight_layout()
        plt.show()


##########
# USAGE

if False:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
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

    # Initialize autoencoder parameters
    params = ae_init(
        in_channels=in_channels,
        img_size=img_dim,
        bottleneck_size=bottleneck_size,
        kernel_size=kernel_size,
        pool_size=pool_size,
        layer_config=layer_config,
        device=device
    )

    print_model_info(params)

    # Load Omniglot data
    train_alphabets = ["Latin", "Greek", "Cyrillic"]
    test_alphabets = ["Mongolian"]
    train_dl, val_dl = data.omniglot_dataloader(train_alphabets, test_alphabets, image_size=img_dim, batch_size=batch_size)

    # Initialize optimizer
    optimizer = optim.AdamW(params.parameters(), lr=lr, weight_decay=wd)

    # Train the model
    for epoch in range(num_epochs):
        train_loss = run_epoch(params, train_dl, optimizer, device, mask_ratio, train=True)
        val_loss = run_epoch(params, val_dl, optimizer, device, mask_ratio, train=False)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

    # Visualize reconstructions
    print("Visualizing reconstructions on validation set...")
    visualize_reconstructions(params, val_dl, device)
