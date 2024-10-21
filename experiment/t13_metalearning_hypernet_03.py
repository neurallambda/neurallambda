'''

can a simple MLP output weight updates that help it in-context?

RESULTS:

- the previous version (t13_metalearning_hypernet.py) did NOT have bugs
- with skip layers, it was learning too well for all tasks, so metalearning couldn't demonstrate its lift
- then, with skip layers off, this was too deep for metalearning's lr=1e-2 to really demonstrate online learning
- I will go forward with this hard-coded AE, instead of the torch.fx derived AE, for the time being
- next step is to use a hyper net to generate the AE's params

PROVENANCE:
- t13_metalearning_hypernet.py  :  that file was struggling to prove a lift from metalearning, but i suspect bugs, so i'm simplifying here.

STEPS:
1. pretrain autoencoder  (t13_metalearning_hypernet_autoencoder_functional)
2. metalearner (t13_metalearning_hypernet_metalearner)
3. hypernetwork (this module ties everything together)


Q:
- metalearn hypernet too? or just its outputs?

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
    # import t13_metalearning_hypernet_autoencoder as AE
    import t13_metalearning_hypernet_autoencoder_functional_03 as AE
    import t13_metalearning_hypernet_flatten as Flatten


# cudnn gives a warning, nondeterminism, and fails non-gracefully, but alleges speed improvements if used
torch.backends.cudnn.enabled = False

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
        assert torch.allclose(param, initial_state[name], atol=1e-12), f"Parameter {name} was mutated during the inner loop"

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
        assert torch.any(param.grad.abs() < 1e-4), f"Zero gradient for parameter {name}"

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
    def __init__(self, in_channels, img_size, bottleneck_size, kernel_size, pool_size, padding_size=1):
        super().__init__()
        self.ae_metadata, self.ae_params = AE.ae_init(
            in_channels=in_channels,
            img_size=img_size,
            bottleneck_size=bottleneck_size,
            kernel_size=kernel_size,
            pool_size=pool_size,
            padding_size=padding_size,
            device=device,
        )
        self.lr_inner = 1e-1 * 5

        n_ae_params = sum(p.numel() for p in self.ae_params.parameters())
        print(f'N AE PARAMS: {n_ae_params}')

        ##########
        #
        hdim = 64
        self.hypernet = nn.Sequential(
            nn.Linear(img_size ** 2 * in_channels, hdim),
            nn.SiLU(),
            nn.Linear(hdim, n_ae_params),
        )
        _, self.ae_shape_template = Flatten.vectorize(self.ae_params)

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
        BATCH_MODE = False
        if BATCH_MODE:
            with torch.enable_grad():

                flat_w = self.hypernet(torch.flatten(mx, start_dim=1, end_dim=-1))
                params_copy = Flatten.unvectorize(flat_w, self.ae_shape_template)

                new_params = params_copy
                for _ in range(1):
                    # params_copy = self.ae_params
                    # params_copy = {k: v.unsqueeze(0).repeat(B, *[1] * v.ndim) for k, v in params_copy.items()}

                    keys, parameters = zip(*new_params.items())
                    output = AE.ae_forward(mx, self.ae_metadata, batch_mode=BATCH_MODE, **new_params)

                    loss = F.mse_loss(output, x)

                    grads = torch.autograd.grad(loss, parameters, create_graph=True)

                    new_params = {}
                    for k, p, g in zip(keys, parameters, grads):
                        new_params[k] = p - g * self.lr_inner


        else:
            new_params = self.ae_params

        # Predict from metalearning-improved params
        final_output = AE.ae_forward(mx, self.ae_metadata, batch_mode=BATCH_MODE, **new_params)
        return final_output


    # # AN ATTEMPT TO ITERATE OVER BATCHES DIRECTLY INSTEAD OF USE_BATCHIFY
    # def forward(self, x):
    #     B = x.shape[0]
    #     outputs = []

    #     for i in range(B):
    #         xi = x[i].unsqueeze(0)  # Add batch dimension of 1

    #         # Mask the images
    #         mask_ratio = 0.2
    #         if self.training:
    #             mask = torch.rand(xi.shape, device=xi.device) < mask_ratio
    #             mxi = xi.clone()
    #             with torch.no_grad():
    #                 mxi[mask] = 1  # 1=white
    #         else:
    #             mxi = xi

    #         if True:
    #             # Metalearning
    #             with torch.enable_grad():
    #                 params_copy = self.ae_params
    #                 # breakpoint()
    #                 params_copy = {k: v + 0 for k, v in params_copy.items()}

    #                 keys, parameters = zip(*params_copy.items())
    #                 output = AE.ae_forward(mxi, self.ae_metadata, **params_copy)
    #                 loss = F.mse_loss(output, xi)

    #                 grads = torch.autograd.grad(loss, parameters, create_graph=True)

    #                 new_params = {}
    #                 for k, p, g in zip(keys, parameters, grads):
    #                     assert g.abs().sum() > 0, f'key={k} has low grad'
    #                     new_params[k] = p - g * self.lr_inner


    #                 # round 2
    #                 params_copy = new_params
    #                 # breakpoint()
    #                 params_copy = {k: v + 0 for k, v in params_copy.items()}

    #                 keys, parameters = zip(*params_copy.items())
    #                 output = AE.ae_forward(mxi, self.ae_metadata, **params_copy)
    #                 loss = F.mse_loss(output, xi)

    #                 grads = torch.autograd.grad(loss, parameters, create_graph=True)

    #                 new_params = {}
    #                 for k, p, g in zip(keys, parameters, grads):
    #                     assert g.abs().sum() > 0, f'key={k} has low grad'
    #                     new_params[k] = p - g * self.lr_inner

    #                 # round 3
    #                 params_copy = new_params
    #                 # breakpoint()
    #                 params_copy = {k: v + 0 for k, v in params_copy.items()}

    #                 keys, parameters = zip(*params_copy.items())
    #                 output = AE.ae_forward(mxi, self.ae_metadata, **params_copy)
    #                 loss = F.mse_loss(output, xi)

    #                 grads = torch.autograd.grad(loss, parameters, create_graph=True)

    #                 new_params = {}
    #                 for k, p, g in zip(keys, parameters, grads):
    #                     assert g.abs().sum() > 0, f'key={k} has low grad'
    #                     new_params[k] = p - g * self.lr_inner


    #         else:
    #             new_params = self.ae_params

    #         # Predict from metalearning-improved params
    #         final_output = AE.ae_forward(mxi, self.ae_metadata, **new_params)
    #         outputs.append(final_output)

    #     # Stack all outputs
    #     return torch.cat(outputs, dim=0)


def visualize_reconstructions(model, dataloader, num_images=10):
    device = list(model.parameters())[0].device
    with torch.no_grad():
        # Get a batch of images
        images, _, _ = next(iter(dataloader))
        images = images.to(device).float().unsqueeze(1)  # add channel dim

        # Get reconstructions
        reconstructions = model(images)

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
            assert reconstructed.shape == imgs.shape
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
lr = 2e-3 * 1
wd = 0.0
mask_ratio = 0.2  # 0.1 = mask 10%

in_channels = 1
img_dim = 32
bottleneck_size = 32
kernel_size = 3
pool_size = 2

model = Thing(
    in_channels=in_channels,
    img_size=img_dim,
    bottleneck_size=bottleneck_size,
    kernel_size=kernel_size,
    pool_size=pool_size,
)

model.to(device)

# metalearning assertions
if False:  # turn off bc the global step does mutate params
    # raise Exception('these checks dont yet support batchify')
    x = torch.randn(64, 1, img_dim, img_dim, device=device)
    metalearning_checks(model, x)

print_model_info(model)

# Load Omniglot data
train_alphabets = ["Latin"]  # , "Greek", "Cyrillic"]
test_alphabets = ["Mongolian"]
train_dl, val_dl = data.omniglot_dataloader(train_alphabets, test_alphabets, image_size=img_dim, batch_size=batch_size)

# # EXPERIMENT: shring training to 1 batch
# print('SHRINKING TRAINING')
# from torch.utils.data import Subset
# batch_size = train_dl.batch_size
# shrink_size = 1
# single_batch_dataset = Subset(train_dl.dataset, range(shrink_size))
# train_dl = DataLoader(
#     single_batch_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=train_dl.num_workers,
#     collate_fn=train_dl.collate_fn,
#     pin_memory=train_dl.pin_memory
# )

# Train the model
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

for epoch in range(num_epochs):
    train_loss = run_epoch(model, train_dl, optimizer, device, train=True)
    val_loss = run_epoch(model, val_dl, optimizer, device, train=False)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

# Visualize reconstructions
print("Visualizing reconstructions on validation set...")
visualize_reconstructions(model, val_dl)
