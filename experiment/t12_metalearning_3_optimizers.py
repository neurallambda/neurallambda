'''

"Functionalize" Optimizers that are end-to-end differentiable for use with metalearning.

1. Test backpropability of optimizers
2. Test performance on XOR (no metalearning)
3. Test performance on XOR (with metalearning inits)


NOTE: you could automate module functionalization, and have differentiable optimizers:
  - "functionalizing" a nn.Module: https://gist.github.com/apaszke/4c8ead6f17a781d589f6655692e7f6f0
  - meta `higher`: https://github.com/facebookresearch/higher/

NOTE: Keys for Metalearning to Work With Pytorch

Typically a pytorch model will optimize layers as static parameters within a model. For metalearning to work, we do not use static parameters, but parameters that are generated online, and for the inner loop of optimization, they are only trained in context. These "online" parameters *can* however be initialized from normal model parameters. In order for them to be trained online, a few things must happen:

    1. You must maintain more control over weights than the pytorch `nn` library usually provides. You accomplish this by only using the functional api, `torch.nn.functional`, and providing all the weights yourself. For instance, instead of using `nn.Linear`, you must handle your weight/bias tensors yourself and use `F.linear(x, weight, bias)`.

    1. You must create an node in pytorch's computation graph, simply by adding 0 to the parameters.

    2. This new variant must be made capable of optimization by using `requires_grad_()`

    3. In the inner loop, we must enable gradient calculation by using `with torch.enable_grad():`

    4. gradients can be calculated during the inner loop using something like: `grads = torch.autograd.grad(loss, flat_params, create_graph=True)`

    5. Use a functional version of an optimizer, from `functional_optimizers.py` to apply the gradients during the inner loop.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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
# Test differentiability/backpropability of optimizer by testing the outer loop
# separate from the inner loop.
#
# The test is given a random input-output pair, and must predict the output from
# the input. The only parameter is a single element bias term. This can be
# metalearned per batch sample, but the model of course only has a single bias
# term that gets replicated per batch sample. This single bias term will be
# initialized poorly, so the outer-loop will be tasked with basically finding
# the data mean, to provide the ideal starting point for metalearning to
# continue from.


# START_BLOCK_1


def simple_net_init(init_value=1.0):
    # bias is initialized poorly, since dataset is drawn from randn. This tests
    # outer-loop's ability to find a good value.
    return {
        'bias': nn.Parameter(torch.tensor([init_value]))
    }

def simple_net(x, params):
    # a separate bias is added per batch item (since these biases are
    # metalearned per batch item)
    return x + params['bias']

def inner_loss(output, target):
    """
    Compute the inner loss (MSE in this case).
    """
    return F.mse_loss(output, target, reduction='mean')

def inner_loop(params, x, y, optimizer_func, num_steps=50, lr=0.1):
    """
    Perform the inner-loop optimization and return the final loss.
    """
    # Initialize optimizer state
    if optimizer_func == sgd:
        opt_state = None
    elif optimizer_func == sgd_momentum:
        opt_state = sgd_momentum_init(params)
    elif optimizer_func == rmsprop:
        opt_state = rmsprop_init(params)
    elif optimizer_func == adam:
        opt_state = adam_init(params)

    # Perform optimization steps

    for _ in range(num_steps):
        with torch.enable_grad():
            output = simple_net(x, params)
            loss = inner_loss(output, y)  # Mean across batch
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=True)
        grads = {k: g for k, g in zip(params.keys(), grads)}

        # Update parameters using the optimizer function
        if optimizer_func == sgd:
            params = optimizer_func(params, grads, lr=lr)
        elif optimizer_func in [sgd_momentum, rmsprop]:
            params, opt_state = optimizer_func(params, grads, opt_state, lr=lr)
        elif optimizer_func == adam:
            params, *opt_state = optimizer_func(params, grads, *opt_state, lr=lr)

    return (
        loss,  # final loss only, not all losses
        params  # final param
    )

def test_optimizer(optimizer_func, initial_params, batch_size, outer_loop_lr, inner_loop_lr, meta_steps):
    torch.manual_seed(0)

    # Create a batched dataset where input equals output
    x = torch.randn((batch_size, 1))
    y = torch.randn((batch_size, 1))

    # Compute meta-loss
    meta_optimizer = torch.optim.Adam([initial_params['bias']], lr=inner_loop_lr)

    print(f"Testing {optimizer_func.__name__}:")

    all_losses = []
    all_biases = []


    for i in range(meta_steps):
        meta_optimizer.zero_grad()
        params = {k: (v + 0).repeat(batch_size, *v.shape).contiguous().requires_grad_() for k, v in initial_params.items()}

        # Forward pass through the entire optimization process
        final_loss, final_param = inner_loop(params, x, y, optimizer_func, lr=outer_loop_lr)

        # Backward pass
        final_loss.backward()

        grad_norm = initial_params['bias'].grad.norm().item()
        inner_biases = [f'{x.item():>.1f}' for x in final_param['bias']][:5]
        print(f"  Step {i+1}: Loss = {final_loss.item():.7f}, Gradient norm = {grad_norm:.7f}, outer_loop_bias={initial_params['bias'].item():>.4f}, some inner_loop_bias={inner_biases}")

        meta_optimizer.step()

        all_losses.append(final_loss.item())
        all_biases.append(initial_params['bias'].detach().item())

    print(f"  {optimizer_func.__name__} completed meta-learning.")
    return all_losses, all_biases

# Run tests
optimizers = [sgd, sgd_momentum, rmsprop, adam]
results = {}

for opt in optimizers:

    initial_params = simple_net_init()  # a single bias term
    losses, biases = test_optimizer(opt, initial_params, batch_size=32, outer_loop_lr=0.1, inner_loop_lr=0.01, meta_steps=50)
    results[opt.__name__] = {'losses': losses, 'biases': biases}
    print()



# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Optimizer Comparison', fontsize=16)

colors = ['b', 'g', 'r', 'c']
optimizers = ['sgd', 'sgd_momentum', 'rmsprop', 'adam']

# Plot losses
for i, opt in enumerate(optimizers):
    losses = results[opt]['losses']
    ax1.plot(range(1, len(losses) + 1), losses, color=colors[i], label=opt)

ax1.set_xlabel('Meta-steps')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs. Meta-steps')
ax1.legend()
ax1.grid(True)

# Plot biases
for i, opt in enumerate(optimizers):
    biases = results[opt]['biases']
    ax2.plot(range(1, len(biases) + 1), biases, color=colors[i], label=opt)

ax2.set_xlabel('Meta-steps')
ax2.set_ylabel('Outer Loop Bias')
ax2.set_title('Outer Loop Bias vs. Meta-steps')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# END_BLOCK_1


##################################################
# Test various lrs and optimizers, plot results (Not metalearning, just, see how they perform)


# START_BLOCK_2

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

def xor_net_init():
    hdim = 4
    return {
        'fc1.weight': torch.randn(hdim, 2, requires_grad=True),
        'fc1.bias': torch.zeros(hdim, requires_grad=True),
        'fc2.weight': torch.randn(1, hdim, requires_grad=True),
        'fc2.bias': torch.zeros(1, requires_grad=True)
    }

def xor_net(x, params):
    x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
    x = torch.sigmoid(x)
    x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
    return torch.sigmoid(x)

def train(optimizer_func, lr=0.01, epochs=1000):
    params = xor_net_init()

    # Initialize optimizer state
    if optimizer_func == sgd:
        opt_state = None
    elif optimizer_func == sgd_momentum:
        opt_state = {k: torch.zeros_like(p) for k, p in params.items()}
    elif optimizer_func == rmsprop:
        opt_state = {k: torch.zeros_like(p) for k, p in params.items()}
    elif optimizer_func == adam:
        opt_state = ({k: torch.zeros_like(p) for k, p in params.items()},
                     {k: torch.zeros_like(p) for k, p in params.items()}, 0)

    losses = []

    for epoch in range(epochs):
        with torch.enable_grad():
            output = xor_net(X, params)
            loss = F.binary_cross_entropy(output, y)
            flat_params = params.values()
            grads = torch.autograd.grad(loss, flat_params, create_graph=True)

        grads = {k: g for k, g in zip(params.keys(), grads)}

        # Update parameters using the optimizer function
        if optimizer_func == sgd:
            params = optimizer_func(params, grads, lr=lr)
        elif optimizer_func == sgd_momentum:
            params, opt_state = optimizer_func(params, grads, opt_state, lr=lr)
        elif optimizer_func == rmsprop:
            params, opt_state = optimizer_func(params, grads, opt_state, lr=lr)
        elif optimizer_func == adam:
            params, *opt_state = optimizer_func(params, grads, *opt_state, lr=lr)

        losses.append(loss.item())

    return losses

optimizers = [sgd, sgd_momentum, rmsprop, adam]
learning_rates = [0.5, 1e-1, 1e-2, 5e-3]

def run_experiments(n_runs, epochs):
    results = {opt.__name__: {lr: [] for lr in learning_rates} for opt in optimizers}

    for run in range(n_runs):
        torch.manual_seed(run)  # Set random seed for reproducibility
        for opt in optimizers:
            for lr in learning_rates:
                losses = train(opt, lr=lr, epochs=epochs)
                results[opt.__name__][lr].append(losses)

        print(f"Completed run {run + 1}/{n_runs}")

    return results

# Run experiments
n_runs = 10
epochs = 200
results = run_experiments(n_runs, epochs)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
fig.suptitle('XOR Problem: Loss vs Epoch for Different Optimizers and Learning Rates', fontsize=16)

colors = plt.cm.rainbow(np.linspace(0, 1, len(optimizers)))

for i, lr in enumerate(learning_rates):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    ax.set_ylim(0, 1.0)

    for (name, lr_results), color in zip(results.items(), colors):
        losses_array = np.array(lr_results[lr])

        mean_loss = np.mean(losses_array, axis=0)
        std_loss = np.std(losses_array, axis=0)
        best_run = losses_array[np.argmin(losses_array[:, -1])]

        epochs = range(1, len(mean_loss) + 1)

        ax.plot(epochs, mean_loss, label=f"{name} (Mean)", color=color, linewidth=2)
        ax.plot(epochs, best_run, label=f"{name} (Best)", color=color, linestyle='--', linewidth=1.5)
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Rate: {lr}')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# Print final mean loss and standard deviation for each optimizer and learning rate
print("\nFinal mean loss and standard deviation for each optimizer and learning rate:")
for opt_name, lr_results in results.items():
    print(f"\n{opt_name}:")
    for lr, losses in lr_results.items():
        final_losses = [run[-1] for run in losses]
        mean_loss = np.mean(final_losses)
        std_loss = np.std(final_losses)
        print(f"  Learning Rate {lr}:")
        print(f"    Mean: {mean_loss:.6f}")
        print(f"    Std:  {std_loss:.6f}")

# END_BLOCK_2


##################################################
# METALEARNING

# START_BLOCK_3

print('__________________________________________________')
print('RUNNING METALEARNING')

def mlp(x, w1, b1, w2, b2):
    y = F.linear(x, w1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, w2, b2)
    y = F.sigmoid(y)
    return y

class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn((16, 2)))
        self.b1 = nn.Parameter(torch.zeros(16))
        self.w2 = nn.Parameter(torch.randn((1, 16)))
        self.b2 = nn.Parameter(torch.zeros(1))
        self.lr_inner = 0.1

    def _params(self):
        def rg(x):
            x = x + 0  # make new node on computation graph
            x.requires_grad = True
            return x
        return {
            'w1': rg(self.w1.data),
            'b1': rg(self.b1.data),
            'w2': rg(self.w2.data),
            'b2': rg(self.b2.data)
        }

    def _forward(self, x, params):
        return mlp(x, params['w1'], params['b1'], params['w2'], params['b2'])

    def forward(self, x, y_true, optimizer_func, opt_state=None):
        params = self._params()
        losses = []

        for _ in range(1000):
            y_pred = self._forward(x, params)
            loss = F.binary_cross_entropy(y_pred, y_true)
            losses.append(loss.item())
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=True)
            grads = {k: g for k, g in zip(params.keys(), grads)}

            # Apply the optimizer
            if optimizer_func == sgd:
                params = optimizer_func(params, grads, lr=self.lr_inner)
            elif optimizer_func in [sgd_momentum, rmsprop]:
                params, opt_state = optimizer_func(params, grads, opt_state, lr=self.lr_inner)
            elif optimizer_func == adam:
                params, *opt_state = optimizer_func(params, grads, *opt_state, lr=self.lr_inner)

        final_output = self._forward(x, params)
        return final_output, losses

def run_experiment(optimizer_func, n_runs=5):
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y_true = torch.tensor([[0.], [1.], [1.], [0.]])

    all_losses = []
    for _ in range(n_runs):
        model = MetaModel()
        params = model._params()
        if optimizer_func == sgd:
            opt_state = None
        elif optimizer_func == sgd_momentum:
            opt_state = sgd_momentum_init(params)
        elif optimizer_func == rmsprop:
            opt_state = rmsprop_init(params)
        elif optimizer_func == adam:
            opt_state = adam_init(params)

        _, losses = model(x, y_true, optimizer_func, opt_state)
        all_losses.append(losses)

    return all_losses

# Run experiments for all optimizers
optimizers = [sgd, sgd_momentum, rmsprop, adam]
results = {opt.__name__: run_experiment(opt) for opt in optimizers}

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, len(optimizers)))

for (name, losses), color in zip(results.items(), colors):
    losses_array = np.array(losses)
    mean_loss = np.mean(losses_array, axis=0)
    std_loss = np.std(losses_array, axis=0)
    best_run = losses_array[np.argmin(losses_array[:, -1])]

    epochs = range(1, len(mean_loss) + 1)

    ax.plot(epochs, mean_loss, label=f"{name} (Mean)", color=color, linewidth=2)
    ax.plot(epochs, best_run, label=f"{name} (Best)", color=color, linestyle='--', linewidth=1.5)
    ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.2)

ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('XOR Problem: Loss vs Iteration for Different Optimizers')
ax.set_yscale('log')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

# Print final mean loss and standard deviation for each optimizer
print("\nFinal mean loss and standard deviation for each optimizer:")
for opt_name, losses in results.items():
    final_losses = [run[-1] for run in losses]
    mean_loss = np.mean(final_losses)
    std_loss = np.std(final_losses)
    print(f"\n{opt_name}:")
    print(f"  Mean: {mean_loss:.6f}")
    print(f"  Std:  {std_loss:.6f}")



# END_BLOCK_3
