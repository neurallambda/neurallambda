'''

Genetic Algorithm for Pytorch Models

TODO:
* Move this to `./experiments`
* mutation rate is agressive, randn-per-affected param
* similarity should only account for named params

'''

import torch
import torch.nn as nn
import copy
import random
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np


##################################################
# MUTATION

def mutate(tensor, mutation_rate):
    """
    Apply mutation to a tensor based on the specified mutation rate.

    Args:
        tensor (torch.Tensor): The tensor to be mutated.
        mutation_rate (float): The probability of mutation for each element in the tensor.

    Returns:
        torch.Tensor: The mutated tensor.
    """
    mask = torch.rand_like(tensor) < mutation_rate
    mutated_tensor = tensor.clone()
    mutated_tensor[mask] = torch.randn_like(mutated_tensor[mask]) * MUTATION_AMOUNT
    return mutated_tensor


def mutate_model(model, mutation_rate, param_names):
    """
    Apply mutation to specified parameters of a nn.Module model based on the mutation rate.

    Args:
        model (nn.Module): The model to be mutated.
        mutation_rate (float): The probability of mutation for each element in the specified parameters.
        param_names (list): A list of parameter names to be mutated.

    Returns:
        nn.Module: The mutated model.
    """
    assert isinstance(model, nn.Module), "Model must be an instance of nn.Module"
    assert 0 <= mutation_rate <= 1, "Mutation rate must be between 0 and 1"
    assert len(param_names) > 0, "At least one parameter name must be provided"

    # Create a copy of the model
    mutated_model = copy.deepcopy(model)

    # Get the state dictionary of the mutated model
    state_dict = mutated_model.state_dict()

    # Apply mutation to the specified parameters in the state dictionary
    for name in param_names:
        assert name in state_dict, f"Parameter '{name}' not found in the model"
        param = state_dict[name]
        mutated_param = mutate(param, mutation_rate)
        state_dict[name] = mutated_param

    # Load the mutated state dictionary into the mutated model
    mutated_model.load_state_dict(state_dict)

    return mutated_model


##################################################
# CROSSOVER


def crossover(parent1, parent2, cross_points):
    '''Flatten the tensors, and then cross_points represents an index in the
    flattened tensor at which the two tensors should be crossed over.'''
    assert parent1.shape == parent2.shape, "Parent tensors must have the same shape"
    assert len(cross_points) > 0, "At least one cross point is required"

    flat_parent1 = parent1.view(-1)
    flat_parent2 = parent2.view(-1)

    child = flat_parent2.clone()

    prev_point = 0
    for i, point in enumerate(cross_points):
        if i % 2 == 0:
            child[prev_point:point] = flat_parent1[prev_point:point]
        else:
            child[prev_point:point] = flat_parent2[prev_point:point]
        prev_point = point

    if len(cross_points) % 2 == 0:
        child[cross_points[-1]:] = flat_parent1[cross_points[-1]:]

    return child.view_as(parent1)


def crossover_models(model1, model2, crossover_points):
    """
    Crossover two nn.Module models based on the specified crossover points.

    Args:
        model1 (nn.Module): The first parent model.
        model2 (nn.Module): The second parent model.
        crossover_points (dict): A dictionary where the keys are parameter names
                                 and the values are lists of crossover points.

    Returns:
        tuple: A tuple containing the two child models after crossover.
    """
    assert isinstance(model1, nn.Module) and isinstance(model2, nn.Module), "Models must be instances of nn.Module"
    assert len(crossover_points) > 0, "At least one parameter crossover point is required"

    # Create copies of the parent models
    child = copy.deepcopy(model1)

    # Get the state dictionaries of the child models
    child_state_dict = child.state_dict()
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    # Perform crossover on the specified parameters
    for name, points in crossover_points.items():
        assert name in model1_state_dict and name in model2_state_dict, f"Parameter '{name}' not found in both models"
        assert len(points) > 0, f"At least one crossover point is required for parameter '{name}'"

        parent1_param = model1_state_dict[name]
        parent2_param = model2_state_dict[name]

        # Perform crossover on the parameters
        child_param = crossover(parent1_param, parent2_param, points)

        # Update the state dictionaries of the child models
        child_state_dict[name] = child_param

    # Load the updated state dictionaries into the child models
    child.load_state_dict(child_state_dict)
    return child


def generate_crossover_points(model, param_names, k_points):
    """
    Generate random crossover points for the specified parameters of a model.

    Args:
        model (nn.Module): The model for which to generate crossover points.
        param_names (list): A list of parameter names to generate crossover points for.
        k_points (int): The number of crossover points to generate for each parameter.

    Returns:
        dict: A dictionary where the keys are parameter names and the values are lists of crossover points.
    """
    crossover_points = {}
    state_dict = model.state_dict()

    for name in param_names:
        assert name in state_dict, f"Parameter '{name}' not found in the model"
        param_size = state_dict[name].numel()
        points = sorted(random.sample(range(1, param_size), k_points))
        crossover_points[name] = points

    return crossover_points


##################################################
# RUNNER

def evaluate_fitness(model, data_loader, n_batches, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if n_batches and i >= n_batches:
                break
            output = model(data.to(device))
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target.to(device)).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def evaluate_population_fitness(population, data_loader, n_batches, device):
    fitnesses = []
    for model, _ in population:
        fitness = evaluate_fitness(model, data_loader, n_batches, device)
        fitnesses.append(fitness)
    for i, (model, _) in enumerate(population):
        population[i] = (model, fitnesses[i])
    return population

def select_parents(population, num_parents):
    selected = sorted(population, key=lambda x: x[1], reverse=True)[:num_parents]
    return selected

# def tournament_selection(population, tournament_size):
#     selected = []
#     for _ in range(tournament_size):
#         individual = random.choice(population)
#         selected.append(individual)
#     return max(selected, key=lambda x: x[1])

def tournament_selection(population, tournament_size):
    best_individual = None
    best_fitness = float('-inf')

    for _ in range(tournament_size):
        individual = random.choice(population)
        fitness = individual[1]

        if fitness > best_fitness:
            best_individual = individual
            best_fitness = fitness

    return best_individual

def cosine_distance_similarity(model1: nn.Module, model2: nn.Module) -> float:
    """
    Compute the cosine distance between the parameters of two models.

    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model.

    Returns:
        float: The cosine distance between the two models.
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    distance = 0.0
    norm1 = 0.0
    norm2 = 0.0

    cossims = 0

    for param in state_dict1:
        vec1 = state_dict1[param].view(-1)
        vec2 = state_dict2[param].view(-1)
        cossims += torch.cosine_similarity(vec1, vec2, dim=0)

    return (1 - cossims / len(state_dict1))

def mse_similarity(model1: nn.Module, model2: nn.Module) -> float:
    """
    Compute the Euclidean distance between the parameters of two models.

    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model.

    Returns:
        float: The Euclidean distance between the two models.
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    distance = 0.0

    for param in state_dict1:
        vec1 = state_dict1[param].view(-1)
        vec2 = state_dict2[param].view(-1)
        distance += torch.sum((vec1 - vec2) ** 2)

    return torch.sqrt(distance).item()

# similarity = cosine_distance_similarity
similarity = mse_similarity

class UnionFind:
    def __init__(self, size: int):
        """
        Initialize the UnionFind data structure.

        Args:
            size (int): The number of elements in the UnionFind.

        Attributes:
            parent (List[int]): A list representing the parent of each element.
                                Initially, each element is its own parent.
            rank (List[int]): A list representing the rank (depth) of each subset.
                              Initially, all ranks are set to 0.
        """
        self.parent: List[int] = list(range(size))
        self.rank: List[int] = [0] * size

    def find(self, u: int) -> int:
        """
        Find the root (representative) element of the subset containing element u.

        Args:
            u (int): The element to find the root for.

        Returns:
            int: The root element of the subset containing u.

        Description:
            The find operation uses path compression to recursively find the root
            element of the subset containing u. It updates the parent of each
            visited element to directly point to the root, compressing the path
            for future find operations.
        """
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u: int, v: int) -> None:
        """
        Merge the subsets containing elements u and v.

        Args:
            u (int): The first element.
            v (int): The second element.

        Description:
            The union operation merges the subsets containing elements u and v.
            It finds the roots of both subsets using the find operation.
            If the roots are different, it merges the subsets by making one root
            the parent of the other based on their ranks.
            If the ranks are the same, it arbitrarily chooses one root as the
            parent and increments its rank.
        """
        root_u: int = self.find(u)
        root_v: int = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def test_union_find():
    uf = UnionFind(5)

    # Test find on initial state
    assert uf.find(0) == 0
    assert uf.find(1) == 1
    assert uf.find(2) == 2
    assert uf.find(3) == 3
    assert uf.find(4) == 4

    # Test union
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)

    uf.union(2, 3)
    assert uf.find(2) == uf.find(3)

    uf.union(1, 2)
    assert uf.find(0) == uf.find(1)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) == uf.find(3)

    # Test find after union
    assert uf.find(4) == 4

    # Test union with different ranks
    uf.union(3, 4)
    assert uf.find(0) == uf.find(4)

    # Test path compression
    assert uf.find(3) == uf.find(0)
    assert uf.parent[3] == uf.parent[0]
    print("UnionFind tests passed!")
test_union_find()



def fitness_sharing(population: List[Tuple[nn.Module, float]], sigma_share: float) -> Tuple[List[Tuple[nn.Module, float]], List[int], torch.Tensor]:
    """
    Apply fitness sharing to the population to maintain diversity and assign niche indices.

    Args:
        population (List[Tuple[nn.Module, float]]): The population of models with their fitness scores.
        sigma_share (float): The similarity threshold for fitness sharing.

    Returns:
        Tuple[List[Tuple[nn.Module, float]], List[int]]: The population with shared fitness scores and their niche indices.
    """
    shared_fitnesses = []
    num_individuals = len(population)
    uf = UnionFind(num_individuals)

    dists = []

    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            dist = similarity(population[i][0], population[j][0])
            dists.append(dist)
            if dist < sigma_share:
                uf.union(i, j)

    niche_dict = {}
    for i in range(num_individuals):
        niche = uf.find(i)
        if niche not in niche_dict:
            niche_dict[niche] = []
        niche_dict[niche].append(i)

    niche_indices = [0] * num_individuals
    for niche, members in niche_dict.items():
        sharing_sum = sum(1 for _ in members)
        for idx in members:
            niche_indices[idx] = niche
            shared_fitnesses.append(population[idx][1] / sharing_sum)

    for i in range(num_individuals):
        population[i] = (population[i][0], shared_fitnesses[i])

    return population, niche_indices, torch.tensor(dists)

def evolve_population(population: List[Tuple[nn.Module, float]],
                      mutation_rate: float,
                      crossover_rate: float,
                      param_names: List[str],
                      k_points: int,
                      retain_top_k: int,
                      sigma_share: float) -> Tuple[List[Tuple[nn.Module, float]], List[int]]:
    """
    Evolve the population using mutation, crossover, and fitness sharing.

    Args:
        population (List[Tuple[nn.Module, float]]): The current population of models with their fitness scores.
        mutation_rate (float): The probability of mutation for each parameter.
        crossover_rate (float): The probability of crossover between two models.
        param_names (List[str]): A list of parameter names to be mutated.
        k_points (int): The number of crossover points.
        retain_top_k (int): The number of top niches to retain the best individual from.
        sigma_share (float): The similarity threshold for fitness sharing.

    Returns:
        Tuple[List[Tuple[nn.Module, float]], List[int]]: The new evolved population and their niche indices.
    """
    # Get top individuals BEFORE fitness sharing
    population.sort(key=lambda x: x[1], reverse=True)
    top_individuals = population[:retain_top_k]

    # Apply fitness sharing and calculate niche indices
    population, niche_indices, dists = fitness_sharing(population, sigma_share)

    # Sort population by fitness in descending order within each niche
    niches = {}
    for i, (model, fitness) in enumerate(population):
        niche = niche_indices[i]
        if niche not in niches:
            niches[niche] = []
        niches[niche].append((model, fitness))

    for niche in niches:
        niches[niche].sort(key=lambda x: x[1], reverse=True)

    # Get the top retain_top_k niches based on the best fitness in each niche
    top_niches = sorted(niches.keys(), key=lambda x: niches[x][0][1], reverse=True)[:retain_top_k]

    # Initialize new population with the best individual from each of the top niches
    new_population = (
        top_individuals +
        [niches[niche][0] for niche in top_niches]
    )

    while len(new_population) < len(population):
        if random.random() < crossover_rate and len(population) > 1:
            # Perform crossover
            parent1 = tournament_selection(population, 3)
            parent2 = tournament_selection(population, 3)

            if parent1[0] != parent2[0]:
                crossover_points = generate_crossover_points(parent1[0], param_names, k_points)
                child = crossover_models(parent1[0], parent2[0], crossover_points)
            else:
                child = copy.deepcopy(parent1[0])
        else:
            # Select a parent for mutation
            parent = tournament_selection(population, 3)
            child = copy.deepcopy(parent[0])

        # Perform mutation
        mutated_child = mutate_model(child, mutation_rate, param_names)

        # Add mutated child to the new population
        new_population.append((mutated_child, 0.0))

    # Apply fitness sharing and calculate niche indices for the new population
    # new_population, new_niche_indices, dists = fitness_sharing(new_population, sigma_share)

    return new_population, niche_indices, dists


def apply_adam_optimizer(model, train_loader, device, learning_rate, n_batches):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i, (data, target) in enumerate(train_loader):
        if i >= n_batches:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

##################################################
# MNIST

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = 'cuda:1'

class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.fc1(x).relu()
        y = self.fc2(y).relu()
        y = self.fc3(y)
        return y

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True, download=False)
all_data = trainset.data.to(device=DEVICE)
all_data = all_data.float() / 255.0
all_data = (all_data - 0.5) / 0.5  # Normalize to [-1, 1]

all_targets = trainset.targets.to(device=DEVICE)
trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(all_data, all_targets),
                                          batch_size=1024,
                                          shuffle=True)


# Function to visualize data from the DataLoader
def visualize_data(loader, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(12, 2))
    for i, (images, labels) in enumerate(loader):
        if i>=num_images:
            break
        image = images[i].numpy().squeeze()
        label = labels[i].item()
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()


# Genetic Algorithm parameters
population_size = 40
generations = 100
mutation_rate = 0.01
MUTATION_AMOUNT = 1.0  # multiplies randn
mutation_rate_max = 0.1
mutation_rate_min = 0.001
crossover_rate = 0.7
param_names = ["fc1.weight", "fc1.bias",
               "fc2.weight", "fc2.bias",
               "fc3.weight", "fc3.bias"]
k_points = 2
retain_top_k = 3
sigma_share = 140
top_sigma = 200
delta_sigma = 0.1

N_BATCHES = 2  # don't train on entire DataLoader, just n_batches per eval

# Initialize population
population = []
for _ in range(population_size):
    model = FFNN()
    model.to(DEVICE)
    fitness = 0.0
    population.append((model, fitness))


##########
# GO

niche_history = []
accs = []
num_niches = []

# START_BLOCK_1
for gen in tqdm(range(generations)):
    print(f"Generation {gen+1}")

    with torch.no_grad():
        population = evaluate_population_fitness(population, trainloader, N_BATCHES, DEVICE)

        best_model, best_fitness = max(population, key=lambda x: x[1])
        accs.append(best_fitness)
        print(f"Best Fitness: {best_fitness:.2f}%")

        # Adaptive mutation rate
        avg_fitness = sum(fitness for _, fitness in population) / len(population)
        mutation_rate = mutation_rate_max - (best_fitness - avg_fitness) / best_fitness * (mutation_rate_max - mutation_rate_min)
        mutation_rate = max(mutation_rate_min, min(mutation_rate_max, mutation_rate))

        population, niche_indices, dists = evolve_population(population, mutation_rate, crossover_rate, param_names, k_points, retain_top_k, sigma_share)
        niche_history.append(niche_indices)

        # Adaptive Sigma
        ni = len(set(niche_indices))
        num_niches.append(ni)
        if ni < 6:
            sigma_share -= (top_sigma - sigma_share) * delta_sigma
        elif ni > population_size/4:
            sigma_share += (top_sigma - sigma_share) * delta_sigma

        print(f'mean dist: {dists.mean():>.2f}, std dist: {dists.std():>.2f}, ni: {ni}, sigma: {sigma_share:>.3f}, mutation_rate:{mutation_rate:>.3f}')

    # Backpropogation2
    if gen % 10 == 0:
        for i, (model,_) in enumerate(sorted(population, key=lambda x: x[1], reverse=True)):
            if i > 5:
                break
            apply_adam_optimizer(model, trainloader, DEVICE, learning_rate=0.001, n_batches=10)

population = evaluate_population_fitness(population, trainloader, N_BATCHES, DEVICE)
best_model, best_fitness = max(population, key=lambda x: x[1])
print(f"\nBest Model Fitness: {best_fitness:.2f}%")
# END_BLOCK_1


##########
# VIZ

# START_BLOCK_2
niche_array = np.array(niche_history)

# Create a figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 15))

# Plot the niche evolution
ax1.imshow(niche_array, aspect='auto', cmap='tab20')
ax1.set_xlabel('Individuals')
ax1.set_ylabel('Generations')
ax1.set_title('Niche Evolution')


# Plot the accuracy over time
ax2.plot(accs)
ax2.set_xlabel('Generations')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Over Time')

# Plot the number of niches over time
ax3.plot(num_niches)
ax3.set_xlabel('Generations')
ax3.set_ylabel('Number of Niches')
ax3.set_title('Number of Niches Over Time')

plt.tight_layout()
plt.show()
# END_BLOCK_2






##################################################
# Tests

import torch
from torch.testing import assert_close
import torch.nn as nn
from torch.testing import assert_close



def test_crossover():

    # Test case 1: Single point crossover
    parent1 = torch.tensor([1, 2, 3, 4, 5])
    parent2 = torch.tensor([6, 7, 8, 9, 10])
    cross_points = [2]
    child = crossover(parent1, parent2, cross_points)
    assert_close(child, torch.tensor([1, 2, 8, 9, 10]))

    # Test case 2: Two-point crossover
    parent1 = torch.tensor([1, 2, 3, 4, 5])
    parent2 = torch.tensor([6, 7, 8, 9, 10])
    cross_points = [1, 3]
    child = crossover(parent1, parent2, cross_points)
    assert_close(child, torch.tensor([1, 7, 8, 4, 5]))

    # Test case 3: Multi-dimensional tensors
    parent1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    parent2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
    cross_points = [3]
    child = crossover(parent1, parent2, cross_points)
    assert_close(child, torch.tensor([[1, 2], [3, 10], [11, 12]]))





def test_crossover_models():

    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

    # Create two parent models with different parameters
    model1 = TestModel()
    model2 = TestModel()

    model1.fc1.weight.data.fill_(1.0)
    model1.fc1.bias.data.fill_(0.1)
    model1.fc2.weight.data.fill_(2.0)
    model1.fc2.bias.data.fill_(0.2)

    model2.fc1.weight.data.fill_(3.0)
    model2.fc1.bias.data.fill_(0.3)
    model2.fc2.weight.data.fill_(4.0)
    model2.fc2.bias.data.fill_(0.4)

    # Define crossover points for each parameter
    crossover_points = {
        'fc1.weight': [50, 100],
        'fc1.bias': [5],
        'fc2.weight': [20, 40, 60],
        'fc2.bias': [2]
    }

    # Perform crossover
    child = crossover_models(model1, model2, crossover_points)

    assert_close(child.fc1.weight.data.flatten()[:50], torch.full((50,), 1.0))
    assert_close(child.fc1.weight.data.flatten()[50:100], torch.full((50,), 3.0))
    assert_close(child.fc1.weight.data.flatten()[100:], torch.full((100,), 1.0))
    assert_close(child.fc1.bias.data.flatten()[:5], torch.full((5,), 0.1))
    assert_close(child.fc1.bias.data.flatten()[5:], torch.full((15,), 0.3))

    assert_close(child.fc2.weight.data.flatten()[:20], torch.full((20,), 2.0))
    assert_close(child.fc2.weight.data.flatten()[20:40], torch.full((20,), 4.0))
    assert_close(child.fc2.weight.data.flatten()[40:60], torch.full((20,), 2.0))
    assert_close(child.fc2.weight.data.flatten()[60:], torch.full((40,), 4.0))
    assert_close(child.fc2.bias.data.flatten()[:2], torch.full((2,), 0.2))
    assert_close(child.fc2.bias.data.flatten()[2:], torch.full((3,), 0.4))


# Run the test
# test_crossover_models()
