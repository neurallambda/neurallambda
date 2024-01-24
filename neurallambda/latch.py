'''

A Neural Latch.

'''


from torch import einsum, tensor, allclose
import neurallambda.hypercomplex as H
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.util import transform_runs
import neurallambda.debug as D
import neurallambda.hypercomplex as H

class Latch(nn.Module):
    def __init__(self, vec_size, number_system):
        super(Latch, self).__init__()
        self.vec_size = vec_size
        self.number_system = number_system
        self.predicate = nn.Parameter(number_system.randn((vec_size,)) * 1e-2)
        self.true = nn.Parameter(number_system.randn((vec_size,)) * 1e-2)
        self.false = nn.Parameter(number_system.randn((vec_size,)) * 1e-2)

    def forward(self, input):
        # input  : [batch_size, vec_size]
        # output : [batch_size, vec_size]
        N = self.number_system
        predicate = N.to_mat(self.predicate).unsqueeze(0)
        true = N.to_mat(self.true)
        false = N.to_mat(self.false)
        matched = N.cosine_similarity(predicate.real, input.real, dim=1)

        matched = matched.squeeze(-1).squeeze(-1) # squeeze hypercomplex dims
        return (
            torch.einsum('vqr, b -> bvqr', true, matched) +
            torch.einsum('vqr, b -> bvqr', false, 1 - matched)
        )


##################################################
#

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cpu'

NUM_EPOCHS = 1000
BATCH_SIZE = 10
GRAD_CLIP = 10.0

LR = 1e-3
WD = 0

N_DATASET_POS = 100
N_DATASET_NEG = 100
VEC_SIZE = 128

NUMBER_SYSTEM = H.Real
# NUMBER_SYSTEM = H.ComplexTorch
# NUMBER_SYSTEM = H.Complex
# NUMBER_SYSTEM = H.Quaternion
N = NUMBER_SYSTEM # shorthand

##########
# Data

# We'll try and find these same vectors being learned within the network.
predicate_vec = N.randn((VEC_SIZE,))
true_vec = N.randn((VEC_SIZE,))
false_vec = N.randn((VEC_SIZE,))

dataset = (
    # positives
    [(predicate_vec, true_vec) for _ in range(N_DATASET_POS)] +
    # negatives
    [(N.randn((VEC_SIZE,)), false_vec) for _ in range(N_DATASET_POS)]
)

dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)


##########
# Go

model = Latch(VEC_SIZE, NUMBER_SYSTEM)
model = model.to(DEVICE)


##########
# Model

def run_epoch(data_loader, model, optimizer):
    model.train()
    total_loss = 0
    all_predictions = []
    all_true_values = []

    for batch in data_loader:
        input_tensor, target_tensor = batch
        input_tensor  = N.to_mat(input_tensor).to(DEVICE)
        target_tensor = N.to_mat(target_tensor).to(DEVICE)

        model.zero_grad()
        output = model(input_tensor)

        loss = (1 - N.cosine_similarity(target_tensor, output, dim=1)).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() / target_tensor.size(1)

    return total_loss / len(data_loader)



##########
# Cheat and set to expected value

# with torch.no_grad():
#     model.predicate[:] = predicate_vec
#     model.true[:] = true_vec
#     model.false[:] = false_vec

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

def check(model):
    ''' Check how much internal vectors are aligning to known vectors. '''
    p = N.cosine_similarity(N.to_mat(model.predicate).to(DEVICE), N.to_mat(predicate_vec).to(DEVICE), dim=0).item()
    t = N.cosine_similarity(N.to_mat(model.true).to(DEVICE),      N.to_mat(true_vec).to(DEVICE), dim=0).item()
    f = N.cosine_similarity(N.to_mat(model.false).to(DEVICE),     N.to_mat(false_vec).to(DEVICE), dim=0).item()
    print(f'SIMILARITY OF LEARNED VECS: p={p:>.3f} t={t:>.3f} f={f:>.3f}')

for epoch in range(NUM_EPOCHS):
    loss = run_epoch(dataset_loader, model, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss:>.9f}')

    if epoch % 100 == 0:
        check(model)
