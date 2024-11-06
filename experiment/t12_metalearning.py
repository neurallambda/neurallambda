'''

SGD in context

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum
from torch.utils.data import DataLoader, TensorDataset, random_split
from copy import deepcopy
import math
from neurallambda.lab.common import print_model_info

torch.manual_seed(152)


# ##################################################
# #
# def generate_unit_vectors(num_samples, dim):
#     vecs = torch.randn(num_samples, dim)
#     return vecs / torch.norm(vecs, dim=1, keepdim=True)

# def generate_random_rotation_matrix(dim):
#     random_matrix = torch.randn(dim, dim)
#     q, _ = torch.linalg.qr(random_matrix)
#     return q

# def create_rotation_dataset(num_samples, vec_dim, set_size):
#     '''
#     Generate num_samples of sets of vectors, successively rotate by a matrix
#     '''
#     rotation_matrices = torch.stack([generate_random_rotation_matrix(vec_dim) for _ in range(num_samples)])
#     out = []
#     for _ in range(set_size):
#         v1 = generate_unit_vectors(num_samples, vec_dim)
#         v2 = torch.einsum('bij, bi -> bj', rotation_matrices, v1)
#         v3 = torch.einsum('bij, bi -> bj', rotation_matrices, v2)
#         out.append(v1)
#         out.append(v2)
#         out.append(v3)

#     return TensorDataset(*out)


# # Verify data
# if False:
#     dim = 128
#     num_samples = 1000
#     dataset = create_rotation_dataset(num_samples, dim, set_size=2)

#     # one sample
#     v1, rv1, v2, rv2 = dataset[0]
#     print()
#     print(f"Original vector: {v1.shape} {v1[:5]}")
#     print(f"Rotated vector : {rv1.shape} {rv1[:5]}")
#     print(f"Original vector length: {torch.norm(v1):.6f}")
#     print(f"Rotated vector length: {torch.norm(rv1):.6f}")
#     print(f"Dot product of original and rotated: {torch.dot(v1, rv1):.6f}")
#     print()
#     print(f"Original vector: {v2.shape} {v2[:5]}")
#     print(f"Rotated vector : {rv2.shape} {rv2[:5]}")
#     print(f"Original vector length: {torch.norm(v2):.6f}")
#     print(f"Rotated vector length: {torch.norm(rv2):.6f}")
#     print(f"Dot product of original and rotated: {torch.dot(v2, rv2):.6f}")
#     BRK


##################################################
# Spherical Interpolation Data

def generate_unit_vector(dim):
    vec = torch.randn(dim)
    return F.normalize(vec, dim=0)

def slerp(v1, v2, t):
    """Spherical linear interpolation."""
    theta = torch.acos((v1 * v2).sum())
    sin_theta = torch.sin(theta)
    return (torch.sin((1 - t) * theta) / sin_theta) * v1 + (torch.sin(t * theta) / sin_theta) * v2

def create_arc_dataset(num_samples, vec_dim, set_size):
    """Generate num_samples of sets of vectors along arcs on a hypersphere."""
    start_vectors = torch.stack([generate_unit_vector(vec_dim) for _ in range(num_samples)])
    end_vectors = torch.stack([generate_unit_vector(vec_dim) for _ in range(num_samples)])

    out = []
    for i in range(set_size):
        t = i / (set_size - 1)
        interpolated = torch.stack([slerp(start, end, t) for start, end in zip(start_vectors, end_vectors)])
        out.append(interpolated)
    return torch.stack(out, dim=1)  # [num_samples, set_size, vec_dim]

def verify_dataset(dataset, tolerance=1e-6):
    """Verify that all vectors in each set lie on the same great circle arc."""
    num_samples, set_size, vec_dim = dataset.shape

    for i in range(num_samples):
        sample = dataset[i]

        # Check unit length
        norms = torch.norm(sample, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            print(f"Sample {i}: Not all vectors have unit length")
            return False

        # Check coplanarity using SVD
        _, s, _ = torch.svd(sample.T)
        if not (s[2:] < tolerance).all():
            print(f"Sample {i}: Vectors are not coplanar")
            return False

        # Check angle consistency
        angles = torch.acos(F.cosine_similarity(sample[:-1], sample[1:], dim=1))
        if not torch.allclose(angles, angles[0] * torch.ones_like(angles), atol=tolerance):
            print(f"Sample {i}: Angles between consecutive vectors are not consistent")
            return False

    print("All samples pass the verification tests")
    return True


if False:
    # Example usage
    num_samples = 1000
    vec_dim = 128
    set_size = 10

    dataset = create_arc_dataset(num_samples, vec_dim, set_size)
    print(f"Dataset created with shape: {dataset.shape}")

    # Verify dataset
    verify_dataset(dataset)



##################################################
# Linear dataset


def generate_random_vector(dim):
    return torch.randn(dim)

def lerp(v1, v2, t):
    """Linear interpolation."""
    return (1 - t) * v1 + t * v2

def create_linear_dataset(num_samples, vec_dim, set_size):
    """Generate num_samples of sets of vectors along linear paths."""
    start_vectors = torch.stack([generate_random_vector(vec_dim) for _ in range(num_samples)])
    end_vectors = torch.stack([generate_random_vector(vec_dim) for _ in range(num_samples)])

    out = []
    for i in range(set_size):
        t = i / (set_size - 1)
        interpolated = torch.stack([lerp(start, end, t) for start, end in zip(start_vectors, end_vectors)])
        out.append(interpolated)
    return torch.stack(out, dim=1)  # [num_samples, set_size, vec_dim]

def verify_dataset(dataset, tolerance=1e-6):
    """Verify that all vectors in each set lie on a straight line."""
    num_samples, set_size, vec_dim = dataset.shape

    for i in range(num_samples):
        sample = dataset[i]

        # Check collinearity
        v1 = sample[1] - sample[0]
        for j in range(2, set_size):
            v2 = sample[j] - sample[0]
            if not torch.allclose(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1), torch.tensor([1.0]), atol=tolerance):
                print(f"Sample {i}: Vectors are not collinear")
                return False

        # Check equal spacing
        distances = torch.norm(sample[1:] - sample[:-1], dim=1)
        if not torch.allclose(distances, distances[0] * torch.ones_like(distances), atol=tolerance):
            print(f"Sample {i}: Distances between consecutive vectors are not consistent")
            return False

    print("All samples pass the verification tests")
    return True

if False:
    # Example usage
    num_samples = 1000
    vec_dim = 128
    set_size = 5

    dataset = create_linear_dataset(num_samples, vec_dim, set_size)
    print(f"Dataset created with shape: {dataset.shape}")

    # Verify dataset
    verify_dataset(dataset)
    BRK

##################################################
# Models

class NaieveMLP(nn.Module):
    '''
    (v1, rv1, v2) -> rv2
    '''

    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 64
        self.layers = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, v1, rv1, v2):
        combined = torch.cat([v1, rv1, v2], dim=1)
        return self.layers(combined)


class NaieveEnc(nn.Module):
    '''
    (v1, rv1) -> W
    rv2 = W @ v2
    '''

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = 64
        self.enc = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * input_dim),
        )

    def forward(self, v1, rv1, v2):
        b = v1.shape[0]
        w = self.enc(torch.cat([v1, rv1], dim=1)).reshape(b, self.input_dim, self.input_dim)
        out = torch.einsum('bxy, bx -> by', w, v2)
        return out


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, lr_inner=1e-2, num_inner_steps=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr_inner = lr_inner
        self.num_inner_steps = num_inner_steps

        self.l1 = nn.Parameter(torch.randn(input_dim, hidden_dim) / math.sqrt(input_dim))
        self.l2 = nn.Parameter(torch.randn(hidden_dim, input_dim) / math.sqrt(hidden_dim))

        # self.l1 = nn.Parameter(torch.randn(input_dim, input_dim) / math.sqrt(input_dim))



    def forward(self, inp):
        B, S = inp.shape[0], inp.shape[1]

        with torch.enable_grad():
            # clone parameters for each sample in the batch
            l1 = self.l1.unsqueeze(0).repeat(B, 1, 1)
            l2 = self.l2.unsqueeze(0).repeat(B, 1, 1)

            # inner loop optimization
            for step in range(self.num_inner_steps):
                all_loss = 0
                for i in range(0, S - 1):
                    ii = inp[:, i]
                    oo = inp[:, i + 1]

                    h1 = einsum('bij, bi -> bj', l1, ii).relu()
                    oo_pred = einsum('bij, bi -> bj', l2, h1)


                    # oo_pred = einsum('bij, bi -> bj', l1, ii)


                    # Compute loss
                    # loss = cosine_distance(oo_pred, oo).mean()
                    loss = F.mse_loss(oo_pred, oo)
                    all_loss = all_loss + loss

                # print(f'{step=}, {i=}, {all_loss=}')

                # compute gradients
                grad1, grad2 = torch.autograd.grad(loss, [l1, l2], create_graph=True)
                # grad1, = torch.autograd.grad(all_loss, [l1,], create_graph=True)

                # Update parameters
                l1 = l1 - grad1 * self.lr_inner
                l2 = l2 - grad2 * self.lr_inner

            # Final prediction for the query

            h1 = einsum('bij, bi -> bj', l1, inp[:, -1]).relu()
            pred = einsum('bij, bi -> bj', l2, h1)

            # pred = einsum('bij, bi -> bj', l1, inp[:, -1])

            return pred


def cosine_distance(pred, target):
    return 1 - F.cosine_similarity(pred, target, dim=1)


def run_epoch(model, dataloader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0

    with torch.set_grad_enabled(train):
        for batch in dataloader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            trg = batch[:, -1]
            B = trg.shape[0]

            predictions = model(inp)
            loss = cosine_distance(predictions, trg).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

            # Accuracy, as thresholded cosine similarity
            with torch.no_grad():
                cosine_sim = F.cosine_similarity(predictions, trg, dim=1)
                correct_predictions += torch.sum(cosine_sim > accuracy_threshold).item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy



##########
# Go

# Hyperparameters and setup
dim = 1024
set_size = 5
num_samples = 1000
batch_size = 32
num_epochs = 100
lr = 1e-3
wd = 0.0
accuracy_threshold = 0.8  # cut off to determine accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# full_dataset = create_arc_dataset(num_samples, dim, set_size)
full_dataset = create_linear_dataset(num_samples, dim, set_size)
# full_dataset = create_arc_dataset(num_samples, dim, set_size)
print(f"Dataset created with shape: {full_dataset.shape}")


# Split dataset
train_p = 0.8
train_size = int(train_p * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model, optimizer
# model = NaieveMLP(dim).to(device)
# model = NaieveEnc(dim).to(device)
model = Model(dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

print_model_info(model)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
    val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")




# for epoch in range(num_epochs):
#     model.train()
#     for batch in train_loader:
#         batch = [x.to(device) for x in batch]
#         optimizer.zero_grad()
#         loss = model.training_step(batch, 0)
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     val_losses = []
#     for batch in val_loader:
#         batch = [x.to(device) for x in batch]
#         with torch.no_grad():
#             val_loss = model.validation_step(batch, 0)
#         val_losses.append(val_loss.item())

#     print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {sum(val_losses)/len(val_losses):.8f}")

# ##################################################

# # START_BLOCK_1


# # Set up the initial tensor
# x = torch.tensor([1.0], requires_grad=True)
# y = x * x * x

# # Compute first-order gradient
# y.backward(create_graph=True)
# print("First-order gradient:", x.grad)  # Should be 3x^2 = 3

# # Retain the gradient for x.grad
# x.grad.retain_grad()

# # Compute second-order gradient
# grad_x = torch.autograd.grad(x.grad, x, create_graph=True)[0]
# print("Second-order gradient:", grad_x)  # Should be 6x = 6

# # If we want to compute even higher-order gradients:
# grad_grad_x = torch.autograd.grad(grad_x, x)[0]
# print("Third-order gradient:", grad_grad_x)  # Should be 6


# # END_BLOCK_1




# # START_BLOCK_2
# class InnerModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.inner_layer = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.inner_layer(x)

# class OuterModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.outer_layer = nn.Linear(5, 1)
#         self.inner_model = InnerModel()

#     def forward(self, x):
#         inner_output = self.inner_model(x)
#         return self.outer_layer(inner_output)

# def inner_loop_optimization(model, x, y, num_inner_steps=5):
#     # inner_optimizer = optim.SGD(model.inner_model.parameters(), lr=0.01)
#     inner_optimizer = optim.AdamW(model.inner_model.parameters(), lr=0.01)

#     for _ in range(num_inner_steps):
#         inner_optimizer.zero_grad()
#         output = model(x)
#         loss = nn.MSELoss()(output, y)
#         loss.backward(create_graph=True)  # Important: create_graph=True allows for higher-order gradients
#         inner_optimizer.step()

#     return model(x)

# # Outer loop training
# def train(model, num_epochs=100):
#     outer_optimizer = optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(num_epochs):
#         outer_optimizer.zero_grad()

#         # Generate some dummy data
#         x = torch.randn(32, 10)
#         y = torch.randn(32, 1)

#         # Perform inner loop optimization
#         output = inner_loop_optimization(model, x, y)

#         # Compute outer loop loss
#         outer_loss = nn.MSELoss()(output, y)

#         # Backward pass for outer loop
#         outer_loss.backward()

#         # Update outer loop parameters
#         outer_optimizer.step()

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {outer_loss.item()}")

# # Create and train the model
# model = OuterModel()
# train(model)
# # END_BLOCK_2





# # START_BLOCK_3

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np

# class MetaLearner(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )

#     def forward(self, x):
#         return self.net(x)

#     def clone_weights(self):
#         return [p.clone() for p in self.parameters()]

#     def load_weights(self, weights):
#         for p, w in zip(self.parameters(), weights):
#             p.data.copy_(w.data)

# def generate_task(num_samples=100, num_classes=2):
#     angle = np.random.uniform(0, 2 * np.pi)
#     X = torch.randn(num_samples, 2)
#     rotation_matrix = torch.tensor([
#         [np.cos(angle), -np.sin(angle)],
#         [np.sin(angle), np.cos(angle)]
#     ])
#     X = X @ rotation_matrix
#     y = (X[:, 1] > 0).long()
#     return X, y

# def inner_loop_update(model, X, y, num_inner_steps=5, inner_lr=0.01):
#     inner_optimizer = optim.SGD(model.parameters(), lr=inner_lr)
#     criterion = nn.CrossEntropyLoss()

#     for _ in range(num_inner_steps):
#         inner_optimizer.zero_grad()
#         output = model(X)
#         loss = criterion(output, y)
#         loss.backward()
#         inner_optimizer.step()

#     return loss.item()

# def meta_train(meta_model, num_tasks, num_epochs, meta_lr, num_inner_steps, inner_lr, device):
#     meta_optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr)
#     losses = []

#     for epoch in range(num_epochs):
#         meta_loss = 0
#         for _ in range(num_tasks):
#             X, y = generate_task()
#             X, y = X.to(device), y.to(device)

#             init_weights = meta_model.clone_weights()
#             task_model = MetaLearner(2, 20, 2).to(device)
#             task_model.load_weights(init_weights)

#             task_loss = inner_loop_update(task_model, X, y, num_inner_steps, inner_lr)
#             meta_loss += task_loss

#         meta_loss /= num_tasks
#         meta_optimizer.zero_grad()
#         meta_loss.backward()
#         meta_optimizer.step()

#         losses.append(meta_loss)
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Meta Loss: {meta_loss:.4f}")

#     return losses

# def visualize_results(losses, meta_model, device, num_tasks=5):
#     plt.figure(figsize=(15, 5))

#     plt.subplot(121)
#     plt.plot(losses)
#     plt.title('Meta-Loss over epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')

#     plt.subplot(122)
#     for i in range(num_tasks):
#         X, y = generate_task(num_samples=200)
#         X, y = X.to(device), y.to(device)

#         init_weights = meta_model.clone_weights()
#         task_model = MetaLearner(2, 20, 2).to(device)
#         task_model.load_weights(init_weights)

#         before_loss = inner_loop_update(task_model, X, y, num_inner_steps=0)
#         after_loss = inner_loop_update(task_model, X, y, num_inner_steps=5)

#         X = X.cpu().numpy()
#         y = y.cpu().numpy()

#         plt.scatter(X[y==0, 0], X[y==0, 1], c='r', alpha=0.5)
#         plt.scatter(X[y==1, 0], X[y==1, 1], c='b', alpha=0.5)

#         print(f"Task {i+1}: Before adaptation loss: {before_loss:.4f}, After adaptation loss: {after_loss:.4f}")

#     plt.title('Task Examples')
#     plt.xlabel('X')
#     plt.ylabel('Y')

#     plt.tight_layout()
#     plt.show()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# meta_model = MetaLearner(2, 20, 2).to(device)
# num_tasks = 10
# num_epochs = 1000
# meta_lr = 0.001
# num_inner_steps = 5
# inner_lr = 0.1

# losses = meta_train(meta_model, num_tasks, num_epochs, meta_lr, num_inner_steps, inner_lr, device)

# visualize_results(losses, meta_model, device)

# print("Meta-training completed.")

# # END_BLOCK_3





##################################################


# # START_BLOCK_4

# # Define the model
# class RotationModel(nn.Module):
#     def __init__(self, input_dim):
#         super(RotationModel, self).__init__()
#         self.layers = nn.Sequential(

#             nn.Linear(input_dim, input_dim),

#             # nn.Linear(input_dim, 64),
#             # nn.ReLU(),
#             # # nn.Linear(64, 64),
#             # nn.Linear(64, input_dim),

#         )

#     def forward(self, x):
#         return self.layers(x)

# # Create dataset
# vec_dim = 64
# set_size = 5
# num_samples = 64
# dataset = create_rotation_dataset(num_samples, vec_dim, set_size)

# # Split dataset
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Create data loaders
# batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# # Initialize model, loss function, and optimizer
# model = RotationModel(vec_dim)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     for batch in train_loader:
#         for i in range(0, len(batch), 2):
#             input_vec, target_vec = batch[i], batch[i+1]
#             optimizer.zero_grad()
#             output = model(input_vec)
#             loss = 1 - F.cosine_similarity(output, target_vec, dim=1).mean()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch in val_loader:
#             for i in range(0, len(batch), 2):
#                 input_vec, target_vec = batch[i], batch[i+1]
#                 output = model(input_vec)
#                 val_loss += 1 - F.cosine_similarity(output, target_vec, dim=1).mean().item()

#     print(f"Epoch {epoch+1}/{num_epochs}, "
#           f"Train Loss: {train_loss/len(train_loader):.4f}, "
#           f"Val Loss: {val_loss/len(val_loader):.4f}")

# # Test the model
# model.eval()
# test_input = generate_unit_vectors(1, vec_dim)
# with torch.no_grad():
#     predicted_rotation = model(test_input)

# print("Test Input:", test_input)
# print("Predicted Rotation:", predicted_rotation)

# # END_BLOCK_4

















# START_BLOCK_5

from tqdm import tqdm

# Hyperparameters and setup
dim = 32
set_size = 5
num_samples = 2
batch_size = 32
num_epochs = 100
lr = 1e-0
wd = 0.0
accuracy_threshold = 0.9  # cut off to determine accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = create_arc_dataset(num_samples, dim, set_size)
print(f"Dataset created with shape: {dataset.shape}")


# Split dataset
train_p = 0.8
train_size = int(train_p * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)



class ArcRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ArcRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, accuracy_threshold):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = batch[:, :-1, :].to(device), batch[:, -1, :].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cosine_distance(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            breakpoint()
            train_correct += (F.cosine_similarity(outputs, targets, dim=1) > accuracy_threshold).float().sum().item()
            train_total += targets.shape[0]

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[:, :-1, :].to(device), batch[:, -1, :].to(device)
                outputs = model(inputs)
                loss = cosine_distance(outputs, targets)

                val_loss += loss.item()
                val_correct += (F.cosine_similarity(outputs, targets, dim=1) > accuracy_threshold).float().sum().item()
                val_total += targets.shape[0]

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print()

# Assuming your setup code is already run and datasets are created

# Model parameters
input_size = dim
hidden_size = 64
output_size = dim

# Create model, loss function, and optimizer
model = ArcRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

# Train the model
train_model(model, train_loader, val_loader, optimizer, num_epochs, device, accuracy_threshold)


# END_BLOCK_5
