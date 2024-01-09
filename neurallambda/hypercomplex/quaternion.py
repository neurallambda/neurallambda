'''.

Quaternions, defined in matrix form. The final 2 dimensions of any tensor are
`[4,4]`, which represents one Quaternion.

'''

import torch
from torch import einsum, allclose

# Quaternion for multiplication on the LEFT
mat_quaternion = torch.tensor([
    # w
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    # x
    [[0, -1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, -1],
     [0, 0, 1, 0]],
    # y
    [[0, 0, -1, 0],
     [0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, -1, 0, 0]],
    # z
    [[0, 0, 0, -1],
     [0, 0, -1, 0],
     [0, 1, 0, 0],
     [1, 0, 0, 0.]],
])

# # NOT QUATERNION, WEIRD EXPERIMENT
# print('NOT QUATERNION, WEIRD EXPERIMENT\n' * 50)

# # Quaternion for multiplication on the LEFT
# mat_quaternion = torch.tensor([
#     # w
#     [[1, 0, 0, 0],
#      [0, 1, 0, 0],
#      [0, 0, 1, 0],
#      [0, 0, 0, 1]],
#     # x
#     [[0, -1, 0, 0],
#      [1, 0, 0, 0],
#      [0, 0, 0, -1],
#      [0, 0, 1, 0]],
#     # y
#     [[0, 0, -1, 0],
#      [0, 0, 0, 1],
#      [1, 0, 0, 0],
#      [0, -1, 0, 0]],
#     # z
#     [[0, 0, 0, -1],
#      [0, 0, -1, 0],
#      [0, 1, 0, 0],
#      [1, 0, 0, 0.]],
# ])

def to_mat(*x):
    ''' Make Matrix representation of a Quaternion '''
    if len(x) == 1 and isinstance(x[0], torch.Tensor):
        return einsum('...ijk, ...i -> ...jk', mat_quaternion.to(x[0].device), x[0])
    return einsum('...ijk, ...i -> ...jk', mat_quaternion, torch.tensor(*x, dtype=torch.float))

def from_mat(qx):
    ''' Convert matrix representation back to a 4-value tensor representation. '''
    # Flatten the matrix and split into individual elements
    elements = torch.flatten(qx, start_dim=-2, end_dim=-1)

    # Adjust the indexing based on the number of dimensions
    if qx.ndim == 3:  # With batch dimension
        w_elements = elements[:, :, [0, 5, 10, 15]]
        x_elements = elements[:, :, [1, 4, 11, 14]]
        y_elements = elements[:, :, [2, 7, 8, 13]]
        z_elements = elements[:, :, [3, 6, 9, 12]]
    else:  # Without batch dimension
        w_elements = elements[..., [0, 5, 10, 15]]
        x_elements = elements[..., [1, 4, 11, 14]]
        y_elements = elements[..., [2, 7, 8, 13]]
        z_elements = elements[..., [3, 6, 9, 12]]

    # Average the components to get w, x, y, z
    w = torch.mean(w_elements, dim=-1, keepdim=True)

    x = (-x_elements[..., 0] + x_elements[..., 1] - x_elements[..., 2] + x_elements[..., 3]) / 4
    y = (-y_elements[..., 0] + y_elements[..., 1] + y_elements[..., 2] - y_elements[..., 3]) / 4
    z = (-z_elements[..., 0] - z_elements[..., 1] + z_elements[..., 2] + z_elements[..., 3]) / 4

    return torch.hstack([w, x, y, z])


def from_mat(qx):
    ''' Convert matrix representation back to a 4-value tensor representation. '''
    # Flatten the matrix and split into individual elements
    elements = torch.flatten(qx, start_dim=-2, end_dim=-1)

    # Adjust the indexing based on the number of dimensions
    if qx.ndim == 3:  # With batch dimension
        w_elements = elements[:, :, [0, 5, 10, 15]]
        x_elements = elements[:, :, [1, 4, 11, 14]]
        y_elements = elements[:, :, [2, 7, 8, 13]]
        z_elements = elements[:, :, [3, 6, 9, 12]]
    else:  # Without batch dimension
        w_elements = elements[..., [0, 5, 10, 15]]
        x_elements = elements[..., [1, 4, 11, 14]]
        y_elements = elements[..., [2, 7, 8, 13]]
        z_elements = elements[..., [3, 6, 9, 12]]

    # Average the components to get w, x, y, z
    w = torch.mean(w_elements, dim=-1, keepdim=True)

    if qx.ndim == 3:  # Ensure the tensors have the same number of dimensions in the batched case
        w = w.unsqueeze(-1)

    x = (-x_elements[..., 0] + x_elements[..., 1] - x_elements[..., 2] + x_elements[..., 3]) / 4
    y = (-y_elements[..., 0] + y_elements[..., 1] + y_elements[..., 2] - y_elements[..., 3]) / 4
    z = (-z_elements[..., 0] - z_elements[..., 1] + z_elements[..., 2] + z_elements[..., 3]) / 4

    return torch.cat([w, x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)



def randn(size):
    return torch.randn(size + (4,))
