'''

Tests for neurallambda.transformer.stack

'''

import torch
import torch.nn.functional as F
from neurallambda.transformer.stack import soft_push, soft_pop


##################################################
# Test soft_push

# def test_soft_push_1():
#     """Test basic push functionality"""
#     pushes = torch.tensor([[1, 1, 0, 1.]])
#     values = torch.tensor([[[3], [5], [7], [11.]]])
#     initial_pointer = torch.tensor([[0, 0, 0, 0.]])
#     initial_stack = torch.tensor([[[0], [0], [0], [0.]]])
#     push_pointer, push_stack = soft_push(pushes, values, initial_pointer, initial_stack)
#     expected_pointer = torch.tensor([[[1, 0, 0, 0],
#                                       [0, 1, 0, 0],
#                                       [0, 1, 0, 0],
#                                       [0, 0, 1, 0.]]])
#     expected_stack = torch.tensor([[[[3], [0], [0], [0]],
#                                     [[3], [5], [0], [0]],
#                                     [[3], [5], [0], [0]],
#                                     [[3], [5], [11], [0.]]]])
#     assert torch.allclose(push_pointer, expected_pointer, atol=1e-6), f"\n{push_pointer.squeeze(0)}"
#     assert torch.allclose(push_stack, expected_stack, atol=1e-6), f"\n{push_stack.squeeze(0).squeeze(-1)}"


def assert_close(x, y, atol=1e-6, rtol=1e-6):
    try:
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
    except AssertionError:
        print('x:')
        print(x)
        print('y:')
        print(y)
        raise

def x_test_soft_push_2():
    """Push on t=0 with empty stack"""
    pushes = torch.tensor([[0, 0, 0, 0.]])
    values = torch.tensor([[[3], [5], [7], [11.]]])
    initial_pointer = torch.tensor([[0, 0, 0, 0.]])
    initial_stack = torch.tensor([[[0], [0], [0], [0.]]])
    push_pointer, push_stack = soft_push(pushes, values, initial_pointer, initial_stack)
    expected_pointer = torch.tensor([[[1, 0, 0, 0],
                                      [1, 0, 0, 0],
                                      [1, 0, 0, 0],
                                      [1, 0, 0, 0.]]])
    expected_stack = torch.tensor([[[[3], [0], [0], [0]],
                                    [[3], [0], [0], [0]],
                                    [[3], [0], [0], [0]],
                                    [[3], [0], [0], [0.]]]])

    assert_close(push_pointer, expected_pointer, atol=1e-6, rtol=1e-6)
    # print(f"\nActual Pointer:\n{push_pointer.squeeze(0)}")

    # f"\nActual Stack:\n{push_stack.squeeze(0).squeeze(-1)}"
    assert_close(push_stack, expected_stack, atol=1e-6, rtol=1e-6)



# def test_soft_push_with_initial_pointer_and_stack():
#     """Test push functionality with an initial stack and pointer."""
#     pushes = torch.tensor([[0, 1, 0, 1]])
#     values = torch.tensor([[[3], [5], [7], [11]]])
#     initial_pointer = torch.tensor([[0, 1, 0, 0]])
#     initial_stack = torch.tensor([[[2], [4], [0], [0]]])

#     _, push_stack = soft_push(pushes, values, initial_pointer, initial_stack)

#     expected_stack = torch.tensor([[[2, 4, 0, 0],
#                                     [2, 4, 5, 0],
#                                     [2, 4, 5, 0],
#                                     [2, 4, 5, 7]]])

#     assert torch.allclose(push_stack, expected_stack), "Push with initial pointer and stack failed."


# ##################################################
# # Test soft_pop

# def test_soft_pop_basic_pop():
#     """Test basic pop functionality."""
#     pops = torch.tensor([[0, 0, 1, 0]])
#     initial_pointer = torch.tensor([[0, 0, 0, 1]])  # Assuming the stack is full up to the 4th position
#     initial_stack = torch.tensor([[[3], [5], [7], [11]]])

#     _, new_stack, popped_vals = soft_pop(pops, initial_pointer, initial_stack)

#     expected_new_stack = torch.tensor([[[3, 5, 7, 0]]])
#     expected_popped_vals = torch.tensor([[[0], [0], [0], [11]]])

#     assert torch.allclose(new_stack, expected_new_stack), "New stack after pop is incorrect."
#     assert torch.allclose(popped_vals, expected_popped_vals), "Popped values are incorrect."

# def test_soft_pop_with_empty_pop():
#     """Test pop functionality with a no-op pop (popping from an empty position)."""
#     pops = torch.tensor([[0, 0, 0, 0]])
#     initial_pointer = torch.tensor([[0, 0, 0, 0]])  # Empty stack
#     initial_stack = torch.tensor([[[0], [0], [0], [0]]])

#     _, new_stack, popped_vals = soft_pop(pops, initial_pointer, initial_stack)

#     expected_new_stack = torch.tensor([[[0, 0, 0, 0]]])
#     expected_popped_vals = torch.tensor([[[0], [0], [0], [0]]])

#     assert torch.allclose(new_stack, expected_new_stack), "New stack should remain unchanged with empty pop."
#     assert torch.allclose(popped_vals, expected_popped)
