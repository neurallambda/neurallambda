'''

Like previous version, but flattening params, and separate args

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


def batch_linear(input, weight, bias=None):
    output = torch.bmm(input.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
    if bias is not None:
        output += bias
    return output


##################################################
# For-loop version, can sometimes be faster?

def batch_conv2d_loop(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    batch_size, _, _, _ = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape[1:]

    # Perform grouped convolution for each item in the batch
    output = torch.stack([
        F.conv2d(input[i:i+1], weight[i], bias[i], stride, padding, dilation, groups).squeeze(0)
        for i in range(batch_size)
    ])
    return output

def batch_conv_transpose2d_loop(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    batch_size, _, _, _ = input.shape
    _, in_channels, kernel_height, kernel_width = weight.shape[1:]

    # Perform grouped transposed convolution for each item in the batch
    output = torch.stack([
        F.conv_transpose2d(input[i:i+1], weight[i], bias[i], stride, padding, output_padding, groups, dilation).squeeze(0)
        for i in range(batch_size)
    ])
    return output


##################################################
# Non-for-loop version

def batch_conv2d(x, weights, biases=None, stride=1, padding=0, dilation=1, groups=1):
    ''' F.conv2d, but where using weights/biases *per batch item* (ie not shared from module) '''
    b, n_in, h, w = x.size()
    b, n_out, n_in, kh, kw = weights.size()

    # Reshape input to (1, batch_size * in_channels, height, width)
    x_reshaped = x.view(1, -1, h, w)

    # Reshape weights to (batch_size * out_channels, in_channels, kernel_height, kernel_width)
    # weights_reshaped = weights.view(-1, n_in, kh, kw)
    weights_reshaped = weights.reshape(b * n_out, n_in, kh, kw)

    # Reshape biases to (batch_size * out_channels)
    # biases_reshaped = biases.view(-1)
    biases_reshaped = biases.reshape(b * n_out)

    # Perform group convolution
    # breakpoint()
    output = F.conv2d(x_reshaped, weights_reshaped, bias=biases_reshaped, stride=stride, padding=padding, groups=b)

    # Reshape output to (batch_size, out_channels, height, width)
    return output.view(b, n_out, h, w)

def batch_conv_transpose2d(x, weights, biases=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    ''' F.conv_transpose2d, but where using weights/biases *per batch item* (ie not shared from module) '''
    b, n_in, h, w = x.size()
    b, n_out, n_in, kh, kw = weights.size()

    # Reshape input to (1, batch_size * n_out, height, width)
    x_reshaped = x.view(1, -1, h, w)

    # Reshape weights to (batch_size * n_out, n_in, kernel_height, kernel_width)
    # weights_reshaped = weights.view(-1, n_in, kh, kw)
    weights_reshaped = weights.reshape(b * n_out, n_in, kh, kw)

    # Reshape biases to (batch_size * n_in)
    # biases_reshaped = biases.view(-1)
    biases_reshaped = biases.reshape(b * n_in)

    # Perform group transposed convolution
    output = F.conv_transpose2d(x_reshaped, weights_reshaped, bias=biases_reshaped,
                                stride=stride, padding=padding, output_padding=0, groups=b)

    # Reshape output to (batch_size, n_in, height, width)
    return output.view(b, n_in, h, w)



##################################################

def ae_init(in_channels, img_size, bottleneck_size, kernel_size, pool_size, padding_size=1, device='cpu'):
    params = nn.ParameterDict({})

    # Hardcoded output channel sizes for 3 layers
    layer_config = [4, 4, 4]

    # Store non-tensor values
    metadata = {
        'in_channels': in_channels,
        'img_size': img_size,
        'bottleneck_size': bottleneck_size,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'padding_size': padding_size,
    }

    current_size = img_size

    # Layer 0
    params['conv_weights_0'] = nn.Parameter(torch.empty(layer_config[0], in_channels, kernel_size, kernel_size, device=device))
    torch.nn.init.kaiming_normal_(params['conv_weights_0'], mode='fan_out', nonlinearity='relu')
    params['conv_bias_encoder_0'] = nn.Parameter(torch.zeros(layer_config[0], device=device))
    params['conv_bias_decoder_0'] = nn.Parameter(torch.zeros(in_channels, device=device))
    current_size = current_size // pool_size

    # Layer 1
    params['conv_weights_1'] = nn.Parameter(torch.empty(layer_config[1], layer_config[0], kernel_size, kernel_size, device=device))
    torch.nn.init.kaiming_normal_(params['conv_weights_1'], mode='fan_out', nonlinearity='relu')
    params['conv_bias_encoder_1'] = nn.Parameter(torch.zeros(layer_config[1], device=device))
    params['conv_bias_decoder_1'] = nn.Parameter(torch.zeros(layer_config[0], device=device))
    current_size = current_size // pool_size

    # Layer 2
    params['conv_weights_2'] = nn.Parameter(torch.empty(layer_config[2], layer_config[1], kernel_size, kernel_size, device=device))
    torch.nn.init.kaiming_normal_(params['conv_weights_2'], mode='fan_out', nonlinearity='relu')
    params['conv_bias_encoder_2'] = nn.Parameter(torch.zeros(layer_config[2], device=device))
    params['conv_bias_decoder_2'] = nn.Parameter(torch.zeros(layer_config[1], device=device))
    current_size = current_size // pool_size

    # Calculate the size of the flattened features
    linear_in_features = layer_config[-1] * current_size * current_size

    # Linear projection of latents
    params['linear_weights'] = nn.Parameter(torch.empty(bottleneck_size, linear_in_features, device=device))
    torch.nn.init.xavier_uniform_(params['linear_weights'])
    params['linear_bias_encoder'] = nn.Parameter(torch.zeros(bottleneck_size, device=device))
    params['linear_bias_decoder'] = nn.Parameter(torch.zeros(linear_in_features, device=device))

    return metadata, params

def encode(x, metadata, conv_weights_0, conv_weights_1, conv_weights_2,
           conv_bias_encoder_0, conv_bias_encoder_1, conv_bias_encoder_2,
           linear_weights, linear_bias_encoder,
           batch_mode=False):

    linear = batch_linear if batch_mode else F.linear
    conv2d = batch_conv2d if batch_mode else F.conv2d

    encoded = x

    pad = metadata['padding_size']

    pool_indices = []

    # Layer 0

    # encoded.shape  =   torch.Size([64, 1, 32, 32])
    # conv_weights_0.shape  =   torch.Size([64, 6, 1, 3, 3])
    # conv_bias_encoder_0.shape  =   torch.Size([64, 6])
    # pad  =   1

    encoded = conv2d(encoded, conv_weights_0, conv_bias_encoder_0, padding=pad)
    encoded = F.silu(encoded)
    encoded, indices = F.max_pool2d(encoded, metadata['pool_size'], return_indices=True)
    pool_indices.append(indices)

    # Layer 1
    encoded = conv2d(encoded, conv_weights_1, conv_bias_encoder_1, padding=pad)
    encoded = F.silu(encoded)
    encoded, indices = F.max_pool2d(encoded, metadata['pool_size'], return_indices=True)
    pool_indices.append(indices)

    # Layer 2
    encoded = conv2d(encoded, conv_weights_2, conv_bias_encoder_2, padding=pad)
    encoded = F.silu(encoded)
    encoded, indices = F.max_pool2d(encoded, metadata['pool_size'], return_indices=True)
    pool_indices.append(indices)

    last_conv_shape = encoded.shape[1:]
    flattened = encoded.view(encoded.size(0), -1)
    encoded = linear(flattened, linear_weights, linear_bias_encoder)
    encoded = F.silu(encoded)

    return encoded, pool_indices, last_conv_shape

def decode(encoded, metadata, conv_weights_0, conv_weights_1, conv_weights_2,
           conv_bias_decoder_0, conv_bias_decoder_1, conv_bias_decoder_2,
           linear_weights, linear_bias_decoder, pool_indices, last_conv_shape,
           batch_mode=False):

    linear = batch_linear if batch_mode else F.linear
    conv_transpose2d = batch_conv_transpose2d if batch_mode else F.conv_transpose2d


    decoded = linear(encoded, torch.transpose(linear_weights, -2, -1), linear_bias_decoder)  # transpose needs to work regardless of batch_mode
    decoded = decoded.view(-1, *last_conv_shape)

    pad = metadata['padding_size']

    # Layer 2
    decoded = F.max_unpool2d(decoded, pool_indices[2], metadata['pool_size'])
    decoded = conv_transpose2d(decoded, conv_weights_2, conv_bias_decoder_2, padding=pad)
    decoded = F.silu(decoded)

    # Layer 1
    decoded = F.max_unpool2d(decoded, pool_indices[1], metadata['pool_size'])
    decoded = conv_transpose2d(decoded, conv_weights_1, conv_bias_decoder_1, padding=pad)
    decoded = F.silu(decoded)

    # Layer 0
    decoded = F.max_unpool2d(decoded, pool_indices[0], metadata['pool_size'])
    decoded = conv_transpose2d(decoded, conv_weights_0, conv_bias_decoder_0, padding=pad)

    return decoded

def ae_forward(x, metadata, conv_weights_0, conv_weights_1, conv_weights_2,
               conv_bias_encoder_0, conv_bias_encoder_1, conv_bias_encoder_2,
               conv_bias_decoder_0, conv_bias_decoder_1, conv_bias_decoder_2,
               linear_weights, linear_bias_encoder, linear_bias_decoder,
               batch_mode=False):
    encoded, pool_indices, last_conv_shape = encode(
        x, metadata, conv_weights_0, conv_weights_1, conv_weights_2,
        conv_bias_encoder_0, conv_bias_encoder_1, conv_bias_encoder_2,
        linear_weights, linear_bias_encoder,
        batch_mode
    )
    decoded = decode(
        encoded, metadata, conv_weights_0, conv_weights_1, conv_weights_2,
        conv_bias_decoder_0, conv_bias_decoder_1, conv_bias_decoder_2,
        linear_weights, linear_bias_decoder, pool_indices, last_conv_shape,
        batch_mode
    )

    # Note: imgs are normalized to [-1, 1]
    decoded = decoded.tanh()
    return decoded


##########


def run_epoch(metadata, params, dataloader, optimizer, device, mask_ratio, train=True):
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

            # inflate params so each row of batch has separate version
            batch_params = {k: v.unsqueeze(0).repeat(B, *[1] * v.dim()) for k, v in params.items()}

            reconstructed = ae_forward(masked_imgs, metadata, batch_mode=True, **batch_params)
            loss = criterion(reconstructed, imgs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples
    return avg_loss

def visualize_reconstructions(metadata, params, dataloader, num_images=10):
    device = list(params.parameters())[0].device

    with torch.no_grad():
        # Get a batch of images
        images, _, _ = next(iter(dataloader))
        images = images.to(device).float().unsqueeze(1)  # add channel dim

        B = images.shape[0]
        params = {k: v.unsqueeze(0).repeat(B, *[1] * v.dim()) for k, v in params.items()}

        # Get reconstructions
        reconstructions = ae_forward(images, metadata, batch_mode=True, **params)

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
    # cudnn gives a warning, nondeterminism, and fails non-gracefully, but alleges speed improvements if used
    torch.backends.cudnn.enabled = False

    torch.manual_seed(152)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

    # Initialize autoencoder parameters
    metadata, params = ae_init(
        in_channels=in_channels,
        img_size=img_dim,
        bottleneck_size=bottleneck_size,
        kernel_size=kernel_size,
        pool_size=pool_size,
        device=device,
        padding_size=1,
    )

    print_model_info(params)

    # Load Omniglot data
    train_alphabets = ["Latin"] # "Greek", "Cyrillic"]
    test_alphabets = ["Mongolian"]
    train_dl, val_dl = data.omniglot_dataloader(train_alphabets, test_alphabets, image_size=img_dim, batch_size=batch_size)

    # Initialize optimizer
    optimizer = optim.AdamW(params.parameters(), lr=lr, weight_decay=wd)

    # Train the model
    for epoch in range(num_epochs):
        train_loss = run_epoch(metadata, params, train_dl, optimizer, device, mask_ratio, train=True)
        val_loss = run_epoch(metadata, params, val_dl, optimizer, device, mask_ratio, train=False)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

    # Visualize reconstructions
    print("Visualizing reconstructions on validation set...")
    visualize_reconstructions(metadata, params, val_dl)




##################################################
# Getting conv2d/conv_transpose2d batchified n working
if False:

    ##########
    ## CONV2D

    # START_BLOCK_2


    # Set up the problem parameters
    img_size = 32
    batch_size = 8
    n_in = 1  # channels in
    n_out = 6  # channels out
    h = img_size  # img height
    w = img_size  # img width
    kh = 3  # kernel height
    kw = 3  # kernel width

    # Create a random input tensor
    # Shape: (batch_size, in_channels, height, width)
    x = torch.randn(batch_size, n_in, h, w)

    # Create a random weight tensor (kernel) for each item in the batch
    # Shape: (batch_size, out_channels, in_channels, kernel_height, kernel_width)
    weights = torch.randn(batch_size, n_out, n_in, kh, kw)

    # Create a random bias tensor for each item in the batch
    # Shape: (batch_size, out_channels)
    biases = torch.randn(batch_size, n_out)

    # Group convolution method
    def group_conv(x, weights, biases):
        # Reshape input to (1, batch_size * in_channels, height, width)
        x_reshaped = x.view(1, -1, h, w)

        # Reshape weights to (batch_size * out_channels, in_channels, kernel_height, kernel_width)
        weights_reshaped = weights.view(-1, n_in, kh, kw)

        # Reshape biases to (batch_size * out_channels)
        biases_reshaped = biases.view(-1)

        # Perform group convolution
        output = F.conv2d(x_reshaped, weights_reshaped, bias=biases_reshaped, stride=1, padding='same', groups=batch_size)

        # Reshape output to (batch_size, out_channels, height, width)
        return output.view(batch_size, n_out, h, w)

    # For loop method
    def for_loop_conv(x, weights, biases):
        output = []
        for i in range(batch_size):
            # Perform convolution for each batch item
            out = F.conv2d(x[i:i + 1], weights[i], bias=biases[i], stride=1, padding='same')
            output.append(out)
        return torch.cat(output, dim=0)

    # Run both methods
    group_output = group_conv(x, weights, biases)
    loop_output = for_loop_conv(x, weights, biases)

    # Compare results
    are_close = torch.allclose(group_output, loop_output, rtol=1e-5, atol=1e-5)
    print("Outputs are close:", are_close)

    # Print shapes
    print("Input shape:", x.shape)
    print("Weights shape:", weights.shape)
    print("Biases shape:", biases.shape)
    print("Group output shape:", group_output.shape)
    print("Loop output shape:", loop_output.shape)

    # Optionally, print the maximum absolute difference
    if not are_close:
        max_diff = (group_output - loop_output).abs().max().item()
        print("Maximum absolute difference:", max_diff)

    # END_BLOCK_2




    ##########
    ## CONV_TRANSPOSE2d


    # START_BLOCK_3

    # Set up the problem parameters
    img_size = 32
    batch_size = 17
    n_in = 6  # channels in (note: this is now output channels for conv_transpose2d)
    n_out = 2  # channels out (note: this is now input channels for conv_transpose2d)
    h = img_size  # img height
    w = img_size  # img width
    kh = 3  # kernel height
    kw = 3  # kernel width

    # Create a random input tensor
    # Shape: (batch_size, n_out, height, width)
    x = torch.randn(batch_size, n_out, h, w)

    # Create a random weight tensor (kernel) for each item in the batch
    # Shape: (batch_size, n_out, n_in, kernel_height, kernel_width)
    weights = torch.randn(batch_size, n_out, n_in, kh, kw)

    # Create a random bias tensor for each item in the batch
    # Shape: (batch_size, n_in)
    biases = torch.randn(batch_size, n_in)

    # Group transposed convolution method
    def group_conv_transpose2d(x, weights, biases):
        # Reshape input to (1, batch_size * n_out, height, width)
        x_reshaped = x.view(1, -1, h, w)

        # Reshape weights to (batch_size * n_out, n_in, kernel_height, kernel_width)
        weights_reshaped = weights.view(-1, n_in, kh, kw)

        # Reshape biases to (batch_size * n_in)
        biases_reshaped = biases.view(-1)

        # Perform group transposed convolution
        output = F.conv_transpose2d(x_reshaped, weights_reshaped, bias=biases_reshaped,
                                    stride=1, padding=1, output_padding=0, groups=batch_size)

        # Reshape output to (batch_size, n_in, height, width)
        return output.view(batch_size, n_in, h, w)

    # For loop method
    def for_loop_conv_transpose2d(x, weights, biases):
        output = []
        for i in range(batch_size):
            # Perform transposed convolution for each batch item
            out = F.conv_transpose2d(x[i:i+1], weights[i], bias=biases[i],
                                     stride=1, padding=1, output_padding=0)
            output.append(out)
        return torch.cat(output, dim=0)

    # Run both methods
    group_output = group_conv_transpose2d(x, weights, biases)
    loop_output = for_loop_conv_transpose2d(x, weights, biases)

    # Compare results
    are_close = torch.allclose(group_output, loop_output, rtol=1e-5, atol=1e-5)
    print("Outputs are close:", are_close)

    # Print shapes
    print("Input shape:", x.shape)
    print("Weights shape:", weights.shape)
    print("Biases shape:", biases.shape)
    print("Group output shape:", group_output.shape)
    print("Loop output shape:", loop_output.shape)

    # Optionally, print the maximum absolute difference
    if not are_close:
        max_diff = (group_output - loop_output).abs().max().item()
        print("Maximum absolute difference:", max_diff)

    # END_BLOCK_3
