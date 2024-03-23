'''

Can I get stacks into transformers?

'''

import torch
import torch.nn.functional as F
import torch.nn as nn

DEVICE = 'cuda'
torch.set_printoptions(precision=1, sci_mode=False)
torch.manual_seed(152)

pp = lambda x: print(x.squeeze(0).squeeze(-1))
ppp = pp

##################################################
# Soft Index, Experiment #1 using diff with arange block

def soft_index_2(vector, temperature=0.1):
    """
    Converts a binary vector into a soft index matrix.

    Args:
    - vector: A binary vector of shape (N,), where N is the number of elements.
    - temperature: A scalar to control the sharpness of softmax.

    Returns:
    - A matrix of shape (N, N) where each row indicates the soft position of 1s.
    """
    N = vector.shape[0]
    cum_positions = torch.cumsum(vector, dim=0)  # Cumulative sum to get positions

    # Expand cum_positions to a matrix for broadcasting
    cum_positions_matrix = cum_positions.unsqueeze(1).expand(-1, N)

    # Create a range matrix
    indices = torch.arange(1, N + 1).expand_as(cum_positions_matrix)

    # Calculate "distance" from each position to each potential index
    distances = torch.abs(cum_positions_matrix - indices)

    # Apply a softmax over distances with a temperature to make it "soft"
    soft_positions = torch.softmax(-distances / temperature, dim=1)

    return soft_positions

# Example usage
if False:
    vector = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.float)
    soft_matrix = soft_index_2(vector, temperature=0.2)
    print(soft_matrix)


##################################################
# Soft Index, Experiment #2 using grid_sample of ident mat

#####
# Non-differentiable Version

# # Fix the loop to correctly simulate the stack operations
# for i in range(seq_len):
#     current_push = pushes[:, i]
#     # Check if there's a push operation at this timestep
#     if current_push.item() == 1:
#         # Find the current stack height by checking how many pushes have been performed before this timestep
#         current_height = pushes[:, :i+1].sum(dim=1).item() - 1
#         # Update the stack with the new value at the current height
#         stacks[:, i, int(current_height), :] = sequence[:, i, :]
#     if i > 0:
#         # Carry forward the stack state from the previous timestep if no push occurred
#         stacks[:, i, :, :] = torch.where(stacks[:, i, :, :] == 0, stacks[:, i-1, :, :], stacks[:, i, :, :])

# # Remove unnecessary dimensions for clarity
# stacks = stacks.squeeze(0).squeeze(-1)
# print(stacks)


def soft_push(pushes, values, initial_pointer, initial_stack):
    """Convert an (soft) binary vector into a matrix where the **index** of a
    (soft) 1.0 tracks how many 1.0s have come before.

    This function takes into account an initial stack pointer position, allowing
    the starting point of the 'push' operations to be other than the first
    position of the stack. This is useful for operations on stacks that aren't
    empty at the beginning of the sequence of operations.

    Usage Example: This is useful for instance to track the `pointer` in a
    differentiable stack, according to how many PUSH operations have been
    performed.

    For instance:

    >>> soft_push([1, 1, 1, 1], [3, 5, 7, 11], [0, 0, 0, 0], [0, 0, 0, 0])
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

    [[3, 0, 0, 0],
     [3, 5, 0, 0],
     [3, 5, 7, 0],
     [3, 5, 7, 11]]

    >>> soft_push([0, 1, 0, 0], [3, 5, 7, 11], [0, 0, 0, 0], [0, 0, 0, 0])
    [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [1, 0, 0, 0],
     [1, 0, 0, 0]]

    [[0, 0, 0, 0],
     [5, 0, 0, 0],
     [5, 0, 0, 0],
     [5, 0, 0, 0]]

    >>> soft_push([0, 1, 0, 0], [3, 5, 7, 11], [0, 1, 0, 0], [2, 4, 0, 0])
    [[0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 1, 0],
     [0, 0, 1, 0]]

    [[2, 4, 0, 0],
     [2, 4, 5, 0],
     [2, 4, 5, 0],
     [2, 4, 5, 0]]

    This function takes a batch of binary vectors, each representing a sequence
    of 0s and 1s.  It then creates a "soft" matrix for each vector where the
    position of "soft" ones within each row is determined by the cumulative sum
    of ones up to that point in the vector. This is achieved using a grid
    sampling technique that samples from an identity matrix that allows for
    differentiable operations, making it suitable for gradient-based
    optimization.

    Args:

      pushes (torch.Tensor): A batch of binary vectors with shape
        [batch_size, N], where N is the length of each vector. Values are
        expected to be in [0,1].

      values (torch.Tensor): A batch of values to be pushed onto the stack according to
                             the push operations specified in `pushes`.
                             Shape: [batch_size, seq_len, value_size].

      initial_pointer (torch.Tensor): A batch of one-hot encoded vectors with shape
        [batch_size, N], where each vector represents the initial position of
        the stack pointer.

    Returns:

      torch.Tensor: A batch of soft matrices with shape [batch_size, N, N],
        where each matrix corresponds to its input vector. If all inputs had
        been either 0 or 1 exactly, each row of resulting matrix will have a 1.0
        at the index which corresponds to the sum of 1s seen so far.

    """

    batch_size, N = pushes.size()

    # Indexes of eye: pick an (interpolated) row from eye according to
    #   normalized cumsum of input, which represents array index. An "index" of
    #   -1.0 refers to the leftmost element, 0 is middle, 1.0 is right most.

    # cum_sum = torch.cumsum(pushes, dim=1)  # [batch_size, N]

    init_sum = torch.sum(initial_pointer * torch.arange(1, N+1), dim=1, keepdim=True)  # [batch_size, N]
    cum_sum = torch.cumsum(pushes, dim=1) + init_sum  # [batch_size, N]

    # Rest of your function stays the same up until you compute `ixs`
    ixs = (cum_sum - 1) / (N - 1) * 2.0 - 1.0  # Normalize to [-1, 1], [batch_size, N]

    grid_x = ixs.unsqueeze(-1).expand(-1, -1, N)  # [batch_size, N, N]

    # Grab whole row of eye via even interpolation from [-1, 1]
    grid_y = torch.linspace(-1, 1, steps=N).unsqueeze(0).expand(batch_size, N, -1)  # [batch_size, N, N]

    # grid represents (x,y) coords to grab out of eye
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [batch_size, N, N, 2]

    # Create an identity matrix (per batch elem) and reshape to 4D tensor (4D
    # because `grid_sample` is meant for imgs with channel dim)
    identity_matrix = torch.eye(N).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, N, N]

    # Apply grid_sample to the identity matrices using the grid, with bilinear interpolation and zero padding
    pointer = F.grid_sample(identity_matrix, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(1)

    # leading edge allows grabbing the value aligned with the original push
    # operation (ie and not the pointer, which continues at this location until
    # the next push).
    leading_edge = pointer.unsqueeze(-1) # [batch, seq, seq, 1]
    leading_edge = (leading_edge - leading_edge.roll(dims=1, shifts=1)).relu()

    # remove intiial pointer's influence on leading edge. ie if initial_pointer
    # is [0,1,0], it'll see that 1 and want to push the first value in the
    # sequence, so this prevents that.
    leading_edge = (leading_edge - initial_pointer.unsqueeze(1).unsqueeze(3).expand(-1, N, -1, vec_size)).relu()

    # multiply leading edge by values to place elements into the stack
    expanded_sequence = values.repeat(1, 1, seq_len).reshape(batch_size, seq_len, seq_len, vec_size) # [batch, seq, seq, vec]
    soft_placed_elements = expanded_sequence * leading_edge  # [batch, seq, seq, vec]
    soft_stack = soft_placed_elements.cumsum(dim=1) # [batch, seq, seq, vec]

    initial_stack_expanded = initial_stack.unsqueeze(1).expand(-1, seq_len, -1, -1)
    final_stack = initial_stack_expanded + soft_stack

    return (
        pointer, # [batch, N, N]
        final_stack  # [batch, N, N, vec]
    )


##########
# Sandbox

batch_size = 1
vec_size = 1
values  = torch.tensor([[[3], [5], [7], [11], [13], [17]]]) # [batch, seq, vec]
pushes  = torch.tensor([[ 0,   1,   0,   1,    0,    1  ]]) # [batch, seq]

seq_len = values.size(1)

stack_depth = seq_len
stacks = torch.zeros((batch_size, seq_len, stack_depth, vec_size)) # [batch, seq, stack, vec]

#####
# Expected output

'''
[[0, 0, 0, 0, 0, 0],    # Before any pushes, the stack is empty
 [5, 0, 0, 0, 0, 0],    # First push operation adds 5
 [5, 0, 0, 0, 0, 0],    # No push, stack remains the same
 [5, 11, 0, 0, 0, 0],   # Second push adds 11 on top of the stack
 [5, 11, 0, 0, 0, 0],   # No push, stack remains the same
 [5, 11, 17, 0, 0, 0]]  # Third push adds 17 on top of the stack
'''


##########
# Differentiable Version

#####
# Pushes

init_pointer = torch.tensor([[ 0,     0,     1,     0,   0,   0]], dtype=torch.float)
init_stack   = torch.tensor([[[0.1], [0.2], [0.3], [0], [0], [0],]], dtype=torch.float)
pointer, push_stack = soft_push(pushes, values, init_pointer, init_stack)  # Shape: [batch, seq, seq]


print('PUSH STACK:')
pp(push_stack)
'''
PUSH STACK:
tensor([[ 0.1,  0.2,  0.3,  0.0,  0.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0,  0.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0,  0.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0, 11.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0, 11.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0, 11.0, 17.0]])
'''


print('PUSH POINTERS:')
pp(pointer)
'''
PUSH POINTERS:
tensor([[0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]])
'''

#####
# Pops


def soft_pop(pops, initial_pointer, initial_stack):
    """Perform a soft pop operation on the stack.

    Args:
        pops (torch.Tensor): A batch of binary vectors with shape [batch_size, N],
            where N is the length of each vector. Values are expected to be in [0, 1].
            A value of 1 indicates a pop operation at that position.
        initial_pointer (torch.Tensor): A batch of soft vectors in [0,1] with shape [batch_size, N],
            representing the current stack pointers.
        initial_stack (torch.Tensor): A batch of vectors from the top of the stacks with shape [batch_size, N, vec_size],
            representing the current state of the stacks.

    Returns:
        torch.Tensor: A batch of updated pointers with shape [batch_size, N, N],
            after performing the pop operations.
        torch.Tensor: A batch of updated stacks with shape [batch_size, N, N, vec_size],
            after performing the pop operations.
        torch.Tensor: A batch of popped values with shape [batch_size, N, vec_size],
            containing the values that were popped from the stack.
    """

    batch_size, N = pops.size()

    # Indexes of eye: pick an (interpolated) row from eye according to
    #   normalized cumsum of input, which represents array index. An "index" of
    #   -1.0 refers to the leftmost element, 0 is middle, 1.0 is right most.

    init_sum = torch.sum(initial_pointer * torch.arange(1, N+1), dim=1, keepdim=True)  # [batch_size, N]
    cum_sum = init_sum - torch.cumsum(pops, dim=1)  # [batch_size, N]

    # Rest of your function stays the same up until you compute `ixs`
    ixs = (cum_sum - 1) / (N - 1) * 2.0 - 1.0  # Normalize to [-1, 1], [batch_size, N]

    grid_x = ixs.unsqueeze(-1).expand(-1, -1, N)  # [batch_size, N, N]

    # Grab whole row of eye via even interpolation from [-1, 1]
    grid_y = torch.linspace(-1, 1, steps=N).unsqueeze(0).expand(batch_size, N, -1)  # [batch_size, N, N]

    # grid represents (x,y) coords to grab out of eye
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [batch_size, N, N, 2]

    # Create an identity matrix (per batch elem) and reshape to 4D tensor (4D
    # because `grid_sample` is meant for imgs with channel dim)
    identity_matrix = torch.eye(N).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, N, N]

    # Apply grid_sample to the identity matrices using the grid, with bilinear
    # interpolation and zero padding
    pointer = F.grid_sample(identity_matrix, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(1)

    # new stack
    mask = pointer.flip(dims=[2]).cumsum(dim=2).flip(dims=[2]).unsqueeze(-1).expand(-1, -1, -1, vec_size)
    new_stack = mask * initial_stack.unsqueeze(1).expand(-1, N, -1, -1)

    # pops
    trailing_edge = pointer.unsqueeze(-1) # [batch, seq, seq, 1]
    trailing_edge = (1 * (trailing_edge - trailing_edge.roll(dims=1, shifts=1)))
    trailing_edge = (-1 * trailing_edge).relu()

    # remove final pointer's influence on trailing edge (which rolls around to
    # top of sequence). ie if final pointer is [0,1,0], it'll see that 1 and
    # want to pop the first value in the sequence after wrapping, so this
    # prevents that.
    last_pointer = pointer[:,-1]
    trailing_edge = (trailing_edge - last_pointer.unsqueeze(1).unsqueeze(3).expand(-1, N, -1, vec_size)).relu()

    popped_vals = initial_stack * trailing_edge.sum(dim=2)

    # breakpoint()

    return (
        pointer, # [batch, N]
        new_stack, # [batch, N, N, vec_size]
        popped_vals # [batch, N, vec_size]
    )


pops = torch.tensor([[0, 0, 1, 0, 1, 0]], dtype=torch.float)  # [batch_size, seq_len]
new_pointer, new_stack, popped_vals = soft_pop(pops, pointer[:, -1], push_stack[:, -1])

print('POP STACK:')
pp(new_stack)
'''
tensor([[ 0.1,  0.2,  0.3,  5.0, 11.0, 17.0],
        [ 0.1,  0.2,  0.3,  5.0, 11.0, 17.0],
        [ 0.1,  0.2,  0.3,  5.0, 11.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0, 11.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0,  0.0,  0.0],
        [ 0.1,  0.2,  0.3,  5.0,  0.0,  0.0]])
'''

print('POP POINTERS:')
pp(new_pointer)
'''
tensor([[0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 0.]])

'''
print('\n')

print('POP VALS:')
print(popped_vals)
'''
tensor([[[ 0.0],
         [ 0.0],
         [ 0.3],
         [ 0.0],
         [11.0],
         [ 0.0]]])
'''
