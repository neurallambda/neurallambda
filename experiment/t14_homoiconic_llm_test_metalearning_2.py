'''.

Continue the t13 series, and do N-way k-shot learning task, where a transformer
backbone outputs LOR updates to itself. Trained with "train-time-data", so
hopefully, at inference we won't even need metalearning!

The N-way k-shot learning task, for testing few-shot learners:
    1. N classes: The task tests classification between N different classes that the model has not seen during training.
    2. k examples per class: For each of the N classes, the model is given only k labeled examples (usually k is a small number like 1 or 5).
    3. Support set: The N*k labeled examples make up the "support set" that the model can use to learn about the new classes.
    4. Query set: The model is then asked to classify new unlabeled examples (the "query set") from the N classes.
    5. Meta-learning: Models are typically trained on many different N-way k-shot tasks so they can learn to adapt quickly to new tasks at test time.
    6. Evaluation: Performance is measured by classification accuracy on the query set.

Metalearning with pretraining on Omniglot, Testing on Mini-Imagenet Test


PROVENANCE:
- t13_metalearning_hypernet_03

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import random
from typing import List, Dict, Any, Tuple

from t13_metalearning_hypernet_data import omniglot_n_way_k_shot, ALPHABET_DICT

from tqdm import tqdm
import warnings

from datetime import datetime
current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = SummaryWriter(f'log/t12_metalearning_3_optimizers_2/{current_time}')
LOG = True

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'

# Notes:
#
# F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
#   input: [B, in_channels, iH, iW]
#   weight: [out_channels, in_channels/groups, kH, kW]
#   bias: [out_channels]


##################################################
# Vectorize / Unvectorize Params
#
#   PROVENANCE: t13_metalearning_hypernet_flatten

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


def numel(params_dict: dict) -> int:
    total_elements = 0
    for param in params_dict.values():
        if isinstance(param, (nn.Parameter, torch.Tensor)):
            total_elements += param.numel()
    return total_elements


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


##################################################
# Batchified Helpers
#
#   In a metalearning context, each batch item will maintain its own in-context
#   weights. These are helper functions from the torch.nn.functional api but
#   upgraded to work with batchified weights.
#
# PROVENANCE: t13_metalearning_hypernet_autoencoder_functional

def batch_linear(input, weight, bias=None):
    ''' F.linear, but using weights/biases *per batch item* (ie not shared from module) '''
    output = torch.bmm(input.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
    if bias is not None:
        output += bias
    return output


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



####################
# Convolutional Encoder

def encoder_init(in_channels, img_size, bottleneck_size, kernel_size, pool_size, padding_size=1, device='cpu'):
    ''' Initialize parameters for the encoder. We won't use this in this module since the transformer will output the weights for us. '''
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
    current_size = current_size // pool_size

    # Layer 1
    params['conv_weights_1'] = nn.Parameter(torch.empty(layer_config[1], layer_config[0], kernel_size, kernel_size, device=device))
    torch.nn.init.kaiming_normal_(params['conv_weights_1'], mode='fan_out', nonlinearity='relu')
    params['conv_bias_encoder_1'] = nn.Parameter(torch.zeros(layer_config[1], device=device))
    current_size = current_size // pool_size

    # Layer 2
    params['conv_weights_2'] = nn.Parameter(torch.empty(layer_config[2], layer_config[1], kernel_size, kernel_size, device=device))
    torch.nn.init.kaiming_normal_(params['conv_weights_2'], mode='fan_out', nonlinearity='relu')
    params['conv_bias_encoder_2'] = nn.Parameter(torch.zeros(layer_config[2], device=device))
    current_size = current_size // pool_size

    # Calculate the size of the flattened features
    linear_in_features = layer_config[-1] * current_size * current_size

    # Linear projection of latents
    params['linear_weights'] = nn.Parameter(torch.empty(bottleneck_size, linear_in_features, device=device))
    torch.nn.init.xavier_uniform_(params['linear_weights'])
    params['linear_bias_encoder'] = nn.Parameter(torch.zeros(bottleneck_size, device=device))

    return metadata, params

def encode(x,
           metadata,
           conv_weights_0, conv_weights_1, conv_weights_2,
           conv_bias_encoder_0, conv_bias_encoder_1, conv_bias_encoder_2,
           linear_weights, linear_bias_encoder,
           ):
    ''' 3 layer convolutional encoder '''

    pad = metadata['padding_size']

    pool_indices = []

    # Ex:
    #   encoded.shape  =   torch.Size([64, 1, 32, 32])
    #   conv_weights_0.shape  =   torch.Size([64, 6, 1, 3, 3])
    #   conv_bias_encoder_0.shape  =   torch.Size([64, 6])
    #   pad  =   1

    # Layer 0
    encoded = x

    encoded = batch_conv2d(encoded, conv_weights_0, conv_bias_encoder_0, padding=pad)
    encoded = F.silu(encoded)
    encoded, indices = F.max_pool2d(encoded, metadata['pool_size'], return_indices=True)
    pool_indices.append(indices)

    # Layer 1
    encoded = batch_conv2d(encoded, conv_weights_1, conv_bias_encoder_1, padding=pad)
    encoded = F.silu(encoded)
    encoded, indices = F.max_pool2d(encoded, metadata['pool_size'], return_indices=True)
    pool_indices.append(indices)

    # Layer 2
    encoded = batch_conv2d(encoded, conv_weights_2, conv_bias_encoder_2, padding=pad)
    encoded = F.silu(encoded)
    encoded, indices = F.max_pool2d(encoded, metadata['pool_size'], return_indices=True)
    pool_indices.append(indices)

    last_conv_shape = encoded.shape[1:]
    flattened = encoded.view(encoded.size(0), -1)
    encoded = batch_linear(flattened, linear_weights, linear_bias_encoder)
    encoded = F.silu(encoded)

    return encoded, pool_indices, last_conv_shape


##################################################
# Functional Optimizers
#
#   Differentiable variants of: SGD, SGD-Momentum, RMSProp, Adam


##########
# SGD

def sgd(params, grads, lr=0.01):
    return {k: p - grads[k] * lr for k, p in params.items()}


##########
# SGD Momentum

def sgd_momentum_init(params):
    return {k: torch.zeros_like(p) for k, p in params.items()}

def sgd_momentum(params, grads, velocity, lr=0.01, momentum=0.9):
    updated_params = {}
    updated_velocity = {}
    for k in params.keys():
        v_new = momentum * velocity[k] + lr * grads[k]
        updated_params[k] = params[k] - v_new
        updated_velocity[k] = v_new
    return updated_params, updated_velocity


##########
# RMSProp

def rmsprop_init(params):
    return {k: torch.zeros_like(p) for k, p in params.items()}  # square averages

def rmsprop(params, grads, square_avg, lr=0.01, alpha=0.99, eps=1e-8):
    updated_params = {}
    updated_square_avg = {}
    for k in params.keys():
        avg_new = alpha * square_avg[k] + (1 - alpha) * grads[k].pow(2)
        updated_params[k] = params[k] - lr * grads[k] / (avg_new.sqrt() + eps)
        updated_square_avg[k] = avg_new
    return updated_params, updated_square_avg


##########
# ADAM

def adam_init(params):
    return (
        {k: torch.zeros_like(p) for k, p in params.items()},  # m
        {k: torch.zeros_like(p) for k, p in params.items()},  # v
        0  # t
    )

def adam(params, grads, m, v, t, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
    updated_params = {}
    updated_m = {}
    updated_v = {}
    t += 1
    for k in params.keys():
        m_new = betas[0] * m[k] + (1 - betas[0]) * grads[k]
        v_new = betas[1] * v[k] + (1 - betas[1]) * grads[k].pow(2)
        m_hat = m_new / (1 - betas[0]**t)
        v_hat = v_new / (1 - betas[1]**t)
        updated_params[k] = params[k] - lr * m_hat / (v_hat.sqrt() + eps)
        updated_m[k] = m_new
        updated_v[k] = v_new
    return updated_params, updated_m, updated_v, t



##################################################
# Model

class Model(torch.nn.Module):
    ''' A transformer model that outputs the weights of a convolutional encoder for solving the N-way k-shot task '''
    def __init__(self):
        super(Model, self).__init__()

        # Transformer backbone.
        #
        # This processes the support images/labels and generates weights for
        # classifying the query_images

        # encoder-style transformer
        self.transformer = TODO

        # output embeddings get projected to the right shape to be
        # unvectorized, and used in the convolutional encoder
        self.weight_head = nn.Linear(...)

        self.encoder_metadata = TODO
        self.encoder_shapes = TODO

    def forward(self, support_imgs, support_labels, query_imgs):
        n_support = len(support_imgs)

        # Stack support images and labels
        TODO

        # Metalearn over support_imgs and support_labels
        with torch.enable_grad():
            # each support image generates its own set of encoder weights
            weights = self.transformer(support_imgs)

            loss = 0
            for w_ix in range(n_support):
                w = self.weight_head(weights[:, w_ix])
                w = unvectorize(w, self.encoder_shapes)
                support_logits = encode(support_imgs, self.encoder_metadata, **w)
                loss = loss + F.cross_entropy(support_logits, support_labels)  # TODO: these need to be reshaped appropriately
            grads = torch.autograd.grad(loss, parameters, create_graph=True)

        pass


##################################################
# Training

def run_epoch(model, dataloader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):

            # batch (Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]):
            #     A batch containing a single task, where:
            #     - The first element is the support set: a list of N*k tuples, each containing
            #       (batched_image, batched_label) for support examples.
            #     - The second element is the query set: a list of N*q tuples, each containing
            #       (batched_image, batched_label) for query examples.

            # (Pdb) type(batch)
            # <class 'list'>
            # (Pdb) len(batch)
            # 2
            # (Pdb) type(batch[0])
            # <class 'list'>
            # (Pdb) len(batch[0])
            # 5
            # (Pdb) type(batch[0][0])
            # <class 'list'>
            # (Pdb) len(batch[0][0])
            # 2
            # (Pdb) batch[0][0][0].shape
            # torch.Size([32, 28, 28])
            # (Pdb) batch[0][0][1].shape
            # torch.Size([32])


            # supports: N*k tuples
            # queries: N queries (or N*q if multiple queries)
            supports, queries = batch
            support_imgs = [x[0].to(device).unsqueeze(1) for x in supports]  # N*k tensors, shape=[B, 1, IMG_SIZE, IMG_SIZE]
            support_labels = [x[1].to(device) for x in supports]  # N*k tensors, shape=[B]
            query_imgs = [x[0].to(device).unsqueeze(1) for x in queries]  # N*k tensors, shape=[B, 1, IMG_SIZE, IMG_SIZE]
            query_labels = [x[1].to(device) for x in queries]  # N*k tensors, shape=[B]
            B = query_labels[0].shape[0]  # batch size

            #####
            # Go

            loss = 0
            for img, target_label in zip(support_imgs + query_imgs, support_labels + query_labels):
                output_labels = model(img)
                loss = loss + F.cross_entropy(output_labels, target_label)

            loss = loss / (len(support_imgs) + len(query_imgs))

            if train:
                optimizer.zero_grad()
                loss.backward()

                # Grad clip
                MAX_GRAD_NORM = 1.0
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples

    # Log weight histogram
    if LOG and train:
        try:
            # for name, param in itertools.chain(model.named_parameters(), lor_models.named_parameters()):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'weights/{name}', param.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'grads/{name}', param.grad.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
        except Exception as e:
            warnings.warn(f'Failed to write to tensorboard: {e}')

    return avg_loss


##################################################
# Go

global_epoch = 0

# Parameters
train_alphabets = ["Latin", "Greek"]
test_alphabets = ["Mongolian"]

image_size = 28
num_filters = 64
n_way = 5  # for 5-way classification
k_shot = 1  # for 1-shot learning
q_query = 1  # query examples per class
num_tasks = 100  # number of tasks per epoch

num_epochs = 100
batch_size = 32
lr = 1e-3
wd = 1e-2

train_dl, test_dl = omniglot_n_way_k_shot(
    train_alphabets,
    test_alphabets,
    n_way,
    k_shot,
    q_query,
    num_tasks,
    image_size,
    batch_size,
)

# Initialize model
model = Model(in_channels=1, num_filters=num_filters, num_classes=n_way, image_size=image_size).to(DEVICE)

parameters = [{
    'params': model.parameters(),
    'lr': lr,
    'wd': wd
}]

optimizer = optim.AdamW(parameters)

####################

train_losses = []
test_losses = []
best_loss = float('inf')

model.train()
for epoch in range(num_epochs):
    global_epoch += 1
    train_loss = run_epoch(model, train_dl, optimizer, DEVICE, train=True)
    train_losses.append(train_loss)
    writer.add_scalars('loss', {'train': train_loss}, global_epoch)

    if epoch % 1 == 0:
        test_loss = run_epoch(model, test_dl, optimizer, DEVICE, train=False)
        # test_loss = 0
        test_losses.append(test_loss)
        writer.add_scalars('loss', {'test': test_loss}, global_epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
