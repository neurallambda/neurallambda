'''

Test Time Training allows SGD within SGD

There's an inner loop of optimization/SGD within the Model.

Then the normal optimzer, Adam, makes the outer loop.

This toy model just does this:

y = x * weight

y and x are vectors, weight is a scalar.

The problem learns to map [1,2,3] -> [4,5,6].

The weight will be updated in both the inner and outer loop.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.lr_inner = 0.1

    def forward(self, x, y_true):
        weight_copy = self.weight
        with torch.enable_grad():
            # one step of SGD
            y_pred = x * weight_copy
            loss = F.mse_loss(y_pred, y_true)
            grad = torch.autograd.grad(loss, weight_copy, create_graph=True)[0]
            weight_copy = weight_copy - grad * self.lr_inner
            # use learned weights
            final_output = x * weight_copy
        return final_output

model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# dataset
x      = torch.tensor([1., 2., 3.])
y_true = torch.tensor([2., 4., 6.])


##########
# t=0: Weight initialized to 1.0, would ace problem if it were 2.0

print(f"weight @t=0: {model.weight.item()}")
# = 1.0


##########
# t=1: First round, do inner loop and outer loop SGD

output = model(x, y_true)
outer_loss = F.mse_loss(output, y_true)
outer_loss.backward()
optimizer.step()

print(f"output @t=1: {output.detach()}")
# = [1.93, 3.86, 5.80]  -  pretty close to real solution

print(f"weight @t=1: {model.weight.item()}")
# = 1.0099  -  outer loop can't account for how close the output was


##########
# t=2: Just do inner loop

output = model(x, y_true)

for _ in range(1000):
    output = model(x, y_true)  # inner loop is not mutating original params

print(f"output @t=2: {output.detach()}")
# = [1.94, 3.88, 5.82] even closer than @t=1, bc the outer loop worked @t=1!

print(f"weight @t=2: {model.weight.item()}")
# same as @t=1 bc we didn't run the optimizer
