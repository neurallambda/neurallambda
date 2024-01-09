'''

Redefine Generalization of Complex, Quaternion, Etc., by representing numbers as matrices


STATUS:

RESOURCES:

Linear Algebra over Quaternions: https://arxiv.org/pdf/1311.7488.pdf

Complex/Quaternions as Matrices: http://www.zipcon.net/~swhite/docs/math/quaternions/matrices.html

Quaternion Book:
  https://math.dartmouth.edu/~jvoight/quat-book.pdf
  https://math.dartmouth.edu/~jvoight/quat.html

'''

import torch
from torch import einsum, allclose

# If you multiply this with a 2d complex number, you get a 2x2 complex number,
# which is the matrix form.
mat_complex = torch.tensor([
    # Real
    [[1, 0],
     [0, 1.]],
    # Imag
    [[0, -1],
     [1, 0.]],
])

def to_mat(*x):
    ''' Make Matrix representation of a Complex number '''
    if len(x) == 1 and isinstance(x[0], torch.Tensor):
        return einsum('...ijk, ...i -> ...jk', mat_complex.to(x[0].device), x[0])
    return einsum('...ijk, ...i -> ...jk', mat_complex, torch.tensor(*x, dtype=torch.float))

def from_mat(mx):
    ''' Convert matrix representation back to a 2-value tensor representation by averaging.'''
    a1, b1, b2, a2 = torch.flatten(mx, start_dim=-2, end_dim=-1).split(1, dim=-1)
    a1 = a1.squeeze(-1)
    b1 = b1.squeeze(-1)
    b2 = b2.squeeze(-1)
    a2 = a2.squeeze(-1)
    return torch.stack([(a1 + a2) / 2, (-b1 + b2) / 2], dim=-1)

def randn(size):
    ''' RandN, non-matrix version '''
    return torch.randn(size + (2,))
