import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import pdb
import torch
import numpy as np
import random
import string
from datasets import Dataset
torch.set_printoptions(precision=3)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'

##########
# Params

NUM_EPOCHS = 1000
BATCH_SIZE = 10
GRAD_CLIP = 10.0

INIT_SCALE = 1.0
LR = 1e-2
WD = 0

N_DATASET_POS = 100
N_DATASET_NEG = 100
VEC_SIZE = 512
DTYPE = torch.float32
# DTYPE = torch.complex64

##########
# Data

# We'll try and find these same vectors being learned within the network.
predicate_vec = torch.randn(VEC_SIZE, dtype=DTYPE)
true_vec = torch.randn(VEC_SIZE, dtype=DTYPE)
false_vec = torch.randn(VEC_SIZE, dtype=DTYPE)

dataset = (
    # positives
    [(predicate_vec, true_vec) for _ in range(N_DATASET_POS)] +
    # negatives
    [(torch.randn(VEC_SIZE, dtype=DTYPE), false_vec) for _ in range(N_DATASET_NEG)]
)

dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

##########
#

def conjugate(x):
    if x.dtype in {torch.complex32, torch.complex64, torch.complex128}:
        return x.real - x.imag * 1j
    return x

def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """
    Cosine similarity between two complex numbers in matrix format.
    """
    if x1.dtype not in {torch.complex32, torch.complex64, torch.complex128}:
        return torch.cosine_similarity(x1, x2, dim=dim, eps=eps)

    x1, x2 = torch.broadcast_tensors(x1, x2)
    dot = torch.sum(x1 * conjugate(x2), dim=dim)
    n1 = torch.norm(x1, dim=dim)
    # n1 = torch.maximum(n1, torch.tensor([eps], device=x1.device))
    n2 = torch.norm(x2, dim=dim)
    # n2 = torch.maximum(n2, torch.tensor([eps], device=x1.device))

    return dot / (n1 * n2)



##########
# Test Cos Sim

test_eps = 1e-4
vec_size = 128

x1 = torch.rand((3, vec_size))
x2 = torch.rand((3, vec_size))
cs = cosine_similarity(x1, x2)
expected_cs = torch.cosine_similarity(x1, x2)
assert torch.allclose(cs, expected_cs, rtol=test_eps, atol=test_eps)

x = 1j * torch.rand((3, vec_size))
cs = cosine_similarity(x, x)
assert torch.allclose(cs, torch.ones_like(cs))

x1 = torch.rand((3, vec_size)) + 1j * torch.rand((3, vec_size))
x2 = torch.rand((3, vec_size)) + 1j * torch.rand((3, vec_size))
cs = cosine_similarity(x1, x2)
assert torch.allclose(cs.real, cs.real, rtol=test_eps, atol=test_eps)

x1 = torch.rand((3, vec_size)) + 1j * torch.rand((3, vec_size))
x2 = -x1
cs = cosine_similarity(x1, x2)
assert torch.allclose(cs, -1 * torch.ones_like(cs))


##########
# Model

class Sim(nn.Module):
    def __init__(self, ):
        super(Sim, self).__init__()
        self.predicate = nn.Parameter(torch.randn(VEC_SIZE, dtype=DTYPE) * INIT_SCALE)
        self.true = nn.Parameter(torch.randn(VEC_SIZE, dtype=DTYPE) * INIT_SCALE)
        self.false = nn.Parameter(torch.randn(VEC_SIZE, dtype=DTYPE) * INIT_SCALE)

    def forward(self, input):
        # input  : [batch_size, vec_size]
        # output : [batch_size, vec_size]
        batch_size = input.size(0)
        predicate = self.predicate.unsqueeze(0)
        matched = cosine_similarity(predicate, input, dim=1)
        return (
            einsum('v, b -> bv', self.true, matched) +
            einsum('v, b -> bv', self.false, 1 - matched)
        )

def run_epoch(data_loader, model, optimizer):
    model.train()
    total_loss = 0
    all_predictions = []
    all_true_values = []

    for batch in data_loader:
        input_tensor, target_tensor = batch
        input_tensor = input_tensor.to(DEVICE)
        target_tensor = target_tensor.to(DEVICE)

        model.zero_grad()
        output = model(input_tensor)
        loss = (1 - cosine_similarity(target_tensor, output.unsqueeze(1))).real.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() / target_tensor.size(1)

    return total_loss / len(data_loader)


##########
# Go

model = Sim()
model = model.to(DEVICE)

##########
# Cheat and set to expected value

# with torch.no_grad():
#     model.predicate[:] = predicate_vec
#     model.true[:] = true_vec
#     model.false[:] = false_vec

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

def check(model):
    ''' Check how much internal vectors are aligning to known vectors. '''
    p = cosine_similarity(model.predicate.to(DEVICE), predicate_vec.to(DEVICE), dim=0)
    t = cosine_similarity(model.true.to(DEVICE),      true_vec.to(DEVICE), dim=0)
    f = cosine_similarity(model.false.to(DEVICE),     false_vec.to(DEVICE), dim=0)
    print(f'SIMILARITY OF LEARNED VECS: p={p:>.3f} t={t:>.3f} f={f:>.3f}')

for epoch in range(NUM_EPOCHS):
    loss = run_epoch(dataset_loader, model, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss:>.9f}')

    if epoch % 100 == 0:
        check(model)
