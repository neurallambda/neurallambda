'''

Generate orthogonal vecs and compare different methods.

'''

##########
# Orthogonal Vecs

import torch
import matplotlib.pyplot as plt
import time

def generate_orthogonal_vectors(num_vectors, dim, seed_vector=None, dtype=torch.float32, method='householder'):
    '''householder: Householder Reflections, can generate a large number of
    orthogonal vectors. OK orthogonality, but still needs DIM close to N.

    mix: Recursively project a random vector through a matrix. Can generate inf
    examples, but they are considerably less orthogonal. Great if you want
    fast, infinite, ok-orthogonal vecs.

    qr: QR Decomposition, can only generate min(N, DIM) examples. Perfect
    orthogonality (in the limit of numerical precision).

    gs: Gram Schmidt, can generate more than QR, but if N isn't close to DIM,
    this will generate NaNs. Near perfect orthogonality. Maybe ideal if you
    need a lot of high quality vecs, and will preprocess out the NaNs. This is
    also super slow.

    torch_init: uses torch.nn.init.orthogonal_, fast, similar to householder in
    quality

    '''
    if method == 'qr':  # QR Decomposition
        matrix = torch.randn(dim, num_vectors, dtype=dtype)
        Q, _ = torch.linalg.qr(matrix)
        return Q.T

    elif method == 'gs':  # Gram-Schmidt process
        vectors = torch.randn(num_vectors, dim, dtype=dtype)
        for i in range(num_vectors):
            for j in range(i):
                vectors[i] -= torch.dot(vectors[i], vectors[j]) * vectors[j]
            vectors[i] /= torch.norm(vectors[i])
        return vectors

    elif method == 'mix':  # Project thru orthogonal matrix
        if seed_vector is None:
            seed_vector = torch.randn(dim, dtype=dtype)
        vectors = [seed_vector]
        random_matrix = torch.randn(dim, dim, dtype=dtype)
        Q, _ = torch.linalg.qr(random_matrix)
        for _ in range(num_vectors - 1):
            new_vector = torch.matmul(Q, vectors[-1])
            vectors.append(new_vector)
        return torch.stack(vectors)

    elif method == 'householder':  # Householder Reflections
        vectors = []
        num_batches = (num_vectors + dim - 1) // dim
        for _ in range(num_batches):
            batch_vectors = torch.eye(dim, dtype=dtype)
            for i in range(dim):
                x = torch.randn(dim, dtype=dtype)
                x /= torch.norm(x)
                e = torch.zeros(dim, dtype=dtype)
                e[i] = 1.0
                u = x - torch.sign(x[i]) * torch.norm(x) * e
                u /= torch.norm(u)
                H = torch.eye(dim, dtype=dtype) - 2 * torch.outer(u, u)
                batch_vectors = torch.matmul(H, batch_vectors.T).T
            vectors.append(batch_vectors)
        vectors = torch.cat(vectors, dim=0)
        return vectors[:num_vectors]

    elif method == 'torch_init':  # PyTorch's orthogonal initialization
        weight = torch.empty(num_vectors, dim, dtype=dtype)
        torch.nn.init.orthogonal_(weight)
        return weight


def cosine_similarities(U):
    similarities = torch.nn.functional.cosine_similarity(U.unsqueeze(1), U.unsqueeze(0), dim=2)
    mask = torch.triu(torch.ones(U.shape[0], U.shape[0]), diagonal=1)
    similarities = similarities[mask.bool()]
    return similarities


# Example usage
N = 1000
DIM = 256

start = time.time()
# orthogonal_vectors = generate_orthogonal_vectors(N, DIM, dtype=torch.float32, method='householder')
orthogonal_vectors = generate_orthogonal_vectors(N, DIM, dtype=torch.float32, method='torch_init')
end = time.time()
print(f'DURATION: {end - start:>.4f}')

start = time.time()
cs = cosine_similarities(orthogonal_vectors)
end = time.time()
print(f'COSSIM DURATION: {end - start:>.4f}')

# Plot histogram of cosine similarities
plt.figure(figsize=(8, 6))
plt.hist(cs.tolist(), bins=50, range=(-1, 1), alpha=0.8)
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Histogram of Cosine Similarities')
plt.grid(True)
plt.show()

print()
print(f'N: {len(orthogonal_vectors)}')
print(f'max sim: {max(abs(min(cs)), max(cs))}')
print(f'% nan: {orthogonal_vectors.sum(dim=1).isnan().sum() / orthogonal_vectors.shape[0] :>.3f}')

print()
cs = torch.tensor(cs)
print(f"Mean cosine similarity: {torch.mean(cs):.6f}")
print(f"Median cosine similarity: {torch.median(cs):.6f}")
print(f"Standard deviation of cosine similarities: {torch.std(cs):.6f}")




##########
# Big Comparison

import numpy as np
from tqdm import tqdm

N_values = [100, 1000, 2000]
DIM_values = [16, 64, 128, 1024]
methods = ['householder', 'qr', 'mix', 'torch_init']  # 'gs', ]
N_TRIALS = 5

fig, axs = plt.subplots(2, len(DIM_values), figsize=(12, 8))
pbar = tqdm(total=len(DIM_values) * len(methods) * len(N_values) * N_TRIALS)
for i, DIM in enumerate(DIM_values):
    # grid lines
    for N in N_values:
        axs[0, i].axvline(x=N, color='k', linestyle='-', alpha=0.1)
        axs[1, i].axvline(x=N, color='k', linestyle='-', alpha=0.1)
    for cs_line in [0, 0.1, 0.5, 1.0]:
        axs[0, i].axhline(y=cs_line, color='k', linestyle='-', alpha=0.1)
        axs[1, i].axhline(y=cs_line, color='k', linestyle='-', alpha=0.1)

    # calculate orthogonal vecs and cossims
    for j, method in enumerate(methods):
        means = []
        stds = []
        maxs = []

        for N in N_values:
            trial_means = []
            trial_stds = []
            trial_maxs = []
            for _ in range(N_TRIALS):
                orthogonal_vectors = generate_orthogonal_vectors(N, DIM, dtype=torch.float32, method=method)
                cs = cosine_similarities(orthogonal_vectors)

                if orthogonal_vectors.shape[0] != N or torch.any(torch.isnan(orthogonal_vectors)):
                    # QR Decomp ends early if N > DIM, and Gram Schmidt creates
                    # nans if N >> DIM
                    mean_cs = None
                    std_cs = None
                    max_cs = None
                else:
                    mean_cs = torch.mean(cs.abs()).item()
                    std_cs = torch.std(cs).item()
                    max_cs = max(torch.max(cs).item(), torch.abs(torch.min(cs)).item())

                trial_means.append(mean_cs)
                trial_stds.append(std_cs)
                trial_maxs.append(max_cs)

                pbar.update(1)

            if None in trial_maxs:
                means.append(None)
                stds.append(None)
                maxs.append(None)
            else:
                means.append(np.mean(trial_means))
                stds.append(np.std(trial_stds))
                maxs.append(np.mean(trial_maxs))

        # Plot maxs
        maxs = [x for x in maxs if x is not None]
        axs[0, i].plot(N_values[:len(maxs)], maxs, label=method, marker='x', markevery=max(len(maxs)-1, 1))

        axs[0, i].set_title(f'DIM={DIM}')
        axs[0, i].set_ylabel('Max Cosine Similarity')
        axs[0, i].set_xticks(N_values)
        axs[0, i].set_xlabel('# Vectors')

        # Plot means with ribbon
        means = [x for x in means if x is not None]
        stds = [x for x in stds if x is not None]
        axs[1, i].plot(N_values[:len(means)], means, label=method, marker='x', markevery=max(len(maxs)-1, 1))
        lo = [m - s for m, s in zip(means, stds) if m is not None]
        hi = [m + s for m, s in zip(means, stds) if m is not None]
        axs[1, i].fill_between(N_values[:len(stds)], lo, hi, alpha=0.1)

        axs[1, i].set_title(f'DIM={DIM}')
        axs[1, i].set_ylabel('Mean Cosine Similarity')
        axs[1, i].set_xticks(N_values)
        axs[1, i].set_xlabel('# Vectors')

axs[0, 0].legend(loc='lower right')

plt.tight_layout()
plt.show()
