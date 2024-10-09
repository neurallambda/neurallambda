'''

Just playin around with SVD

'''


# import torch



# # Create a random matrix
# A = torch.randn(256, 128)

# m, n = A.shape

# # Compute A^T A
# ATA = torch.matmul(A.t(), A)

# # Eigendecomposition of A^T A
# eigenvalues, eigenvectors = torch.linalg.eigh(ATA)

# # Sort eigenvalues and eigenvectors in descending order
# sorted_indices = torch.argsort(eigenvalues, descending=True)
# eigenvalues = eigenvalues[sorted_indices]
# eigenvectors = eigenvectors[:, sorted_indices]

# # Compute singular values
# singular_values = torch.sqrt(eigenvalues)

# # Compute left singular vectors
# U = torch.matmul(A, eigenvectors)
# U = U / torch.norm(U, dim=0, keepdim=True)

# # Compute right singular vectors (V^T)
# VT = eigenvectors.t()

# # Handle case when m < n
# if m < n:
#     S = torch.zeros(m, n, dtype=A.dtype, device=A.device)
#     S[:, :m] = torch.diag(singular_values)
# else:
#     S = torch.diag(singular_values)

# # low rank
# #   U[:, :k], S, VT[:k, :]

# # Reconstruct the original matrix
# k = 20
# print((A - torch.matmul(U, torch.matmul(S, VT))).abs().max())
# print((A - torch.matmul(U[:, :k], torch.matmul(S[:k, :k], VT[:k, :]))).abs().max())



import torch
import matplotlib.pyplot as plt

def low_rank_approx(A, k):
    U, S, Vt = torch.linalg.svd(A)
    U_k = U[:, :k]
    S_k = torch.diag(S[:k])
    Vt_k = Vt[:k, :]
    A_k = torch.einsum('ij,jj,jk->ik', U_k, S_k, Vt_k)
    return A_k

def plot_stats(A, max_k):
    m, n = A.shape
    k_values = range(1, min(m, n, max_k) + 1)
    errors = []
    explained_variances = []

    for k in k_values:
        A_k = low_rank_approx(A, k)
        error = torch.norm(A - A_k, p='fro')
        explained_variance = torch.sum(torch.diag(torch.einsum('ij,ij->i', A, A_k))) / torch.norm(A, p='fro') ** 2
        errors.append(error.item())
        explained_variances.append(explained_variance.item())

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, errors)
    plt.xlabel('k')
    plt.ylabel('Frobenius Norm Error')
    plt.title('Approximation Error')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, explained_variances)
    plt.xlabel('k')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance')

    plt.tight_layout()
    plt.show()

# # Example usage
# m, n = 100, 80
# A = torch.randn(m, n)
# max_k = 50
# plot_stats(A, max_k)


k = 40
U, S, Vt = torch.linalg.svd(A)
U_k = U[:, :k]
S_k = torch.diag(S[:k])
Vt_k = Vt[:k, :]
A_k = torch.einsum('ij,jj,jk->ik', U_k, S_k, Vt_k)

U_z = torch.zeros(200, k)
U_z[:100] = U_k

Vt_z = torch.zeros(k, 200)
Vt_z[:, :80] = Vt_k

A_z = torch.einsum('ij,jj,jk->ik', U_z, S_k, Vt_z)


plt.imshow(A_z.numpy())
plt.show()
