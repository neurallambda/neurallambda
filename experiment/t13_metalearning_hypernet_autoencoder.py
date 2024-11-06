'''

can a simple MLP output weight updates that help it in-context?

PROVENANCE:
- t12_metalearning.py


Reconstruct unlearned MNIST/Symbols


ABLATIONS
- CNN vs MLP Mixer
- AE vs AE-Hypernet vs AE-Metalearning vs AE-Hypernet-Metalearning
- pooling: avg vs max

IMPROVING AUTOENCODER
- Use MLP Mixer?? just in middle after convs?
- batch/layer norm?
- [ ] BatchNorm is supposed to help a lot. It may also be responsible for decoder-only working without skip connections. Use batchnorm *before* activations (apparently its debated). BatchNorm centers you wrt activation functions.
- [X] use_skip_connections helped a lot.
- [X] pad input image, then rm padding after decoding. (https://ai.stackexchange.com/a/34929) helped a lot
- interpolate + conv instead of deconvolution. But then can you still tie params? Maybe interpolate to quadruple, then use the tied weights to halve it?
- https://distill.pub/2016/deconv-checkerboard/
  - "sub pixel convolution"
  - resize with nearest neighbor (maybe bilinear but that struggled), then conv
  - fwd convolution = bwd deconvolution, so, artifacts happen less obviously there
  - max pooling linked to high freq artifacts

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

class Autoencoder(nn.Module):
    def __init__(self, in_channels, img_size, bottleneck_size, kernel_size, pool_size, layer_config, actfn='swish', padding_size=1, use_skip_connections=True):
        super().__init__()
        assert actfn in {'relu', 'swish'}
        self.in_channels = in_channels
        self.img_size = img_size
        self.bottleneck_size = bottleneck_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.layer_config = layer_config
        self.actfn = actfn
        self.padding_size = padding_size
        self.use_skip_connections = use_skip_connections

        # Initialize layer parameters
        self.conv_weights = nn.ParameterList()
        self.conv_bias_encoder = nn.ParameterList()
        self.conv_bias_decoder = nn.ParameterList()
        self.use_pool = []

        current_channels = in_channels
        current_size = img_size + 2 * padding_size  # Account for initial padding
        for out_channels, use_pool in layer_config:
            # He initialization for conv weights
            conv_weight = nn.Parameter(torch.Tensor(out_channels, current_channels, kernel_size, kernel_size))
            nn.init.kaiming_normal_(conv_weight, mode='fan_out', nonlinearity='relu')
            self.conv_weights.append(conv_weight)

            # Initialize biases to zero
            self.conv_bias_encoder.append(nn.Parameter(torch.zeros(out_channels)))
            self.conv_bias_decoder.append(nn.Parameter(torch.zeros(current_channels)))

            self.use_pool.append(use_pool)
            current_channels = out_channels
            if use_pool:
                current_size = current_size // 2

        # Linear projection of latents
        linear_in_features = current_channels * current_size * current_size
        self.linear_weights = nn.Parameter(torch.Tensor(bottleneck_size, linear_in_features))
        nn.init.xavier_uniform_(self.linear_weights)
        self.linear_bias_encoder = nn.Parameter(torch.zeros(bottleneck_size))
        self.linear_bias_decoder = nn.Parameter(torch.zeros(linear_in_features))

    def encode(self, x):
        encoded = x
        if self.padding_size > 0:
            encoded = F.pad(encoded, (self.padding_size,) * 4, mode='reflect')
        pool_indices = []
        skip_connections = []
        shapes = [encoded.shape]  # Store initial shape

        for conv_weight, conv_bias, use_pool in zip(self.conv_weights, self.conv_bias_encoder, self.use_pool):
            encoded = F.conv2d(encoded, conv_weight, conv_bias, padding=1)
            if self.actfn == 'relu':
                encoded = F.relu(encoded)
            elif self.actfn == 'swish':
                encoded = F.silu(encoded)
            if self.use_skip_connections:
                skip_connections.append(encoded)
            if use_pool:
                encoded, indices = F.max_pool2d(encoded, self.pool_size, return_indices=True)
                pool_indices.append(indices)
            else:
                pool_indices.append(None)
            shapes.append(encoded.shape)  # Store shape after each layer

        last_conv_shape = encoded.shape[1:]
        flattened = encoded.view(encoded.size(0), -1)
        encoded = F.linear(flattened, self.linear_weights, self.linear_bias_encoder)
        if self.actfn == 'relu':
            encoded = F.relu(encoded)
        elif self.actfn == 'swish':
            encoded = F.silu(encoded)

        return encoded, pool_indices, last_conv_shape, skip_connections, shapes

    def decode(self, encoded, pool_indices, last_conv_shape, skip_connections=None, shapes=None):
        # decoded = F.linear(encoded, self.linear_weights.t(), self.linear_bias_decoder)
        decoded = F.linear(encoded, torch.transpose(self.linear_weights, -2, -1), self.linear_bias_decoder)

        # note: `*last_conv_shape` doesn't work with fx.Proxy, so, unpack by hand
        decoded = decoded.view(-1, last_conv_shape[0], last_conv_shape[1], last_conv_shape[2])

        for i, (conv_weight, conv_bias, use_pool, indices) in enumerate(zip(
            reversed(self.conv_weights),
            reversed(self.conv_bias_decoder),
            reversed(self.use_pool),
            reversed(pool_indices)
        )):
            if use_pool:
                decoded = F.max_unpool2d(decoded, indices, self.pool_size, output_size=shapes[-(i + 2)][2:])
            if skip_connections and i < len(skip_connections):
                decoded = decoded + skip_connections[-(i + 1)]
            decoded = F.conv_transpose2d(decoded, conv_weight, conv_bias, padding=1)
            if i < len(self.conv_weights) - 1:
                if self.actfn == 'relu':
                    decoded = F.relu(decoded)
                elif self.actfn == 'swish':
                    decoded = F.silu(decoded)

        # Remove padding
        if self.padding_size > 0:
            decoded = decoded[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        return decoded

    def forward(self, x):
        encoded, pool_indices, last_conv_shape, skip_connections, shapes = self.encode(x)
        decoded = self.decode(encoded, pool_indices, last_conv_shape, skip_connections, shapes)

        # Note: imgs are normalized to [-1, 1]
        decoded = decoded.tanh()
        return decoded


def run_epoch(model, dataloader, optimizer, device, mask_ratio, train=True):
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

            # Mask the images
            if train:
                mask = torch.rand(imgs.shape, device=device) < mask_ratio
                masked_imgs = imgs.clone()
                masked_imgs[mask] = 1  # 1=white
            else:
                masked_imgs = imgs

            reconstructed = model(masked_imgs)
            loss = criterion(reconstructed, imgs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples
    return avg_loss


def visualize_reconstructions(vmodel, vmodel_params, dataloader, num_images=10):
    device = list(vmodel_params.parameters())[0].device
    vmodel.eval()
    with torch.no_grad():
        # Get a batch of images
        images, _, _ = next(iter(dataloader))
        images = images.to(device).float().unsqueeze(1)  # add channel dim

        # Get reconstructions
        reconstructions = vmodel(images, **vmodel_params)

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
# Go

if False:
    torch.manual_seed(152)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = Autoencoder(
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
        train_loss = run_epoch(model, train_dl, optimizer, device, mask_ratio, train=True)
        val_loss = run_epoch(model, val_dl, optimizer, device, mask_ratio, train=False)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

    # Visualize reconstructions
    print("Visualizing reconstructions on validation set...")
    visualize_reconstructions(model, val_dl)
