'''

Tie together Complex and Quaternions

'''

# try:
#     import os
#     p = os.getcwd()
#     print('getcwd(): ', p)
#     if 'hypercomplex' in p:
#         os.chdir('..')
#     import warnings
#     warnings.warn('TODO: remove this path hack in hypercomplex')
# except:
#     pass

import neurallambda.hypercomplex.complex as c
import neurallambda.hypercomplex.quaternion as q
import torch
from torch import einsum, allclose

RUN_COMPLEX_TESTS    = False
RUN_QUATERNION_TESTS = False


##################################################
# Types


class Real:
    dim = 1

    @staticmethod
    def to_mat(x):
        return x.unsqueeze(-1)

    @staticmethod
    def from_mat(x):
        return x.squeeze(-1)

    @staticmethod
    def randn(size):
        ''' Not in matrix form '''
        return torch.randn(size + (1,))

    @staticmethod
    def randn_mat(size):
        ''' RandN, matrix-formatted version '''
        return Real.to_mat(Real.randn(size))


class Complex:
    dim = 2

    @staticmethod
    def to_mat(x):
        return c.to_mat(x)

    @staticmethod
    def from_mat(x):
        return c.from_mat(x)

    @staticmethod
    def randn(size):
        ''' Not in matrix form '''
        return c.randn(size)

    @staticmethod
    def randn_mat(size):
        ''' RandN, matrix-formatted version '''
        return Complex.to_mat(Complex.randn(size))


class Quaternion:
    dim = 4

    @staticmethod
    def to_mat(x):
        return q.to_mat(x)

    @staticmethod
    def from_mat(x):
        return q.from_mat(x)

    @staticmethod
    def randn(size):
        ''' Not in matrix form '''
        return q.randn(size)

    @staticmethod
    def randn_mat(size):
        ''' RandN, matrix-formatted version '''
        return Quaternion.to_mat(Quaternion.randn(size))


##################################################
# Math

def assert_is_probably_mat_form(x):
    assert x.shape[-1] == x.shape[-2] and x.shape[-1] in {1, 2, 4}, f'expected tensor in matrix format (ie shape=[..., n, n]), but has shape={x.shape}'

def assert_is_probably_not_mat_form(x):
    assert x.shape[-1] != x.shape[-2] and x.shape[-1] in {1, 2, 4}, f'expected tensor in non-matrix format (ie shape=[..., 2]), but has shape={x.shape}'

def scale(q, c):
    ''' Multiply two hypercomplex tensors in matrix format '''
    return einsum('...sr, ... -> ...sr', q, c)

def conjugate(x):
    ''' Transpose is the same as conjugate, at least for Complex and Quaternon. '''
    return torch.transpose(x, dim0=-2, dim1=-1)

def hadamard(x1, x2):
    ''' Element-wise product. Notice the einsum script doesn't look like
    element-wise pdt. Matrix-form notation for hypercomplex numbers allows
    individual scalars to be multiplied together via dot product, which is what
    you're seeing here. Were it to be `...sr, ...sr -> ...sr` that would not be
    elem-wise mul of hypercomplex nums.'''
    return einsum('...sr, ...rq -> ...sq', x1, x2)

def dot_product(x1, x2, dim):
    ''' Dot Product. Elem-wise multiplication, then summing along some dim. '''
    assert_is_probably_mat_form(x1)
    assert_is_probably_mat_form(x2)

    mat_w = x1.shape[-1]
    xx = einsum('...sr, ...rq -> ...', x1, x2) / mat_w # div-n bc matrix form repeats elems
    return torch.sum(xx, dim=dim)

def norm(x, dim):
    ''' Compute the norm of a complex number in matrix format '''
    sq = hadamard(x, conjugate(x))
    return torch.sqrt(torch.sum(sq[..., 0, 0], dim=dim)) # return real

def cosine_similarity(x1, x2, dim, eps=1e-8):
    """
    Cosine similarity between two complex numbers in matrix format.
    """
    assert_is_probably_mat_form(x1)
    assert_is_probably_mat_form(x2)

    x1, x2 = torch.broadcast_tensors(x1, x2)
    dot = dot_product(x1, conjugate(x2), dim=dim)
    n1 = norm(x1, dim=dim)
    n1 = torch.maximum(n1, torch.tensor([eps], device=x1.device))
    n2 = norm(x2, dim=dim)
    n2 = torch.maximum(n2, torch.tensor([eps], device=x1.device))
    return dot / (n1 * n2)


##################################################
# Complex Tests

if RUN_COMPLEX_TESTS:
    C = Complex

    #####
    # Test round tripping MCmpx and UnMCmpx
    for _ in range(10):
        a = C.randn_mat((3,))
        b = C.from_mat(a)
        assert allclose(a, C.to_mat(b))

    #####
    # Multiply

    # >>> (1 + 3j) * (5 + 7j)
    #     (-16+22j)
    # >>> (-1 - 9j) * (-3 + 5j)
    #     (48+22j)
    # >>> (0 + 1j) * (3 + 0j)
    #     (0+3j)
    x1  = torch.stack([C.to_mat([1, 3]), C.to_mat([-1, -9]), C.to_mat([0, 1])])
    x2  = torch.stack([C.to_mat([5, 7]), C.to_mat([-3, 5]), C.to_mat([3, 0])])
    exp = torch.stack([C.to_mat([-16, 22]), C.to_mat([48, 22]), C.to_mat([0, 3])])
    mul = einsum('...sr, ...rq -> ...sq', x1, x2)
    assert allclose(mul, exp)

    #####
    # Conjugate
    x1 = C.to_mat([1, 3])
    out = conjugate(x1)
    exp = C.to_mat([1, -3])
    assert allclose(out, exp), f'\nOut:\n{out}\nExpected:\n{exp}'

    #####
    # Dot product

    # Test 1: Dot product of a complex number with its conjugate should be its squared magnitude
    x = C.to_mat([3, 4])
    y = dot_product(x, conjugate(x), dim=0)
    exp = torch.tensor(3 ** 2 + 4 ** 2.)
    assert allclose(y, exp), "Dot product Test 1 failed"

    # Test 2: Dot product of orthogonal complex numbers should be 0
    x1 = C.to_mat([1, 0])
    x2 = C.to_mat([0, 1])
    y = dot_product(x1, conjugate(x2), dim=0)
    assert allclose(y, torch.tensor(0.0)), "Dot product Test 2 failed"

    #####
    # Norm
    x = C.to_mat([3, 4])
    x_norm = norm(x, dim=0)
    assert torch.allclose(x_norm, torch.tensor([5.]))

    #####
    # Cosine Similarity

    # Test 1: Cosine similarity of a complex number with itself should be 1
    x = C.to_mat([3, 4])
    y = cosine_similarity(x, x, dim=0)
    assert allclose(y, torch.tensor(1.0)), "Test 1 failed"

    # Test 2: Cosine similarity between orthogonal complex numbers should be close to 0
    x1 = C.to_mat([1, 0])
    x2 = C.to_mat([0, 1])
    assert allclose(cosine_similarity(x1, x2, dim=0), torch.tensor(0.0)), "Test 2 failed"

    # Test 3: Cosine similarity between non-orthogonal complex numbers
    x1 = C.to_mat([1, 2])
    x2 = C.to_mat([3, 4])
    expected_result = (1*3 + 2*4) / (torch.sqrt(torch.tensor(1**2 + 2**2)) * torch.sqrt(torch.tensor(3**2 + 4**2)))
    assert allclose(cosine_similarity(x1, x2, dim=0), expected_result), "Test 3 failed"


    #####
    # Bilinearity of Dot Product with Conjugation
    x = C.randn_mat((3,))
    y = C.randn_mat((3,))
    z = C.randn_mat((3,))
    a = torch.randn((1,))

    # Linearity in the first argument with conjugation
    out1 = dot_product(x, conjugate(y + z), dim=0)
    out2 = dot_product(x, conjugate(y), dim=0) + dot_product(x, conjugate(z), dim=0)
    assert allclose(out1, out2), "Should be linear in the first argument with conjugation"

    # Scalar multiplication with conjugation
    assert allclose(dot_product(x, conjugate(a * y), dim=0),
                    a * dot_product(x, conjugate(y), dim=0)), "Should respect scalar multiplication with conjugation"

    # Symmetry with Conjugation
    assert allclose(dot_product(x, conjugate(y), dim=0),
                    dot_product(y, conjugate(x), dim=0)), "Should be symmetric with conjugation"

    #####
    # Cauchy-Schwarz Inequality
    # assert torch.all(torch.abs(dot_product(x, conjugate(y))) <= norm(x) * norm(y)).item(), "Cauchy-Schwarz inequality should hold"

    # Cauchy-Schwarz Inequality
    x = C.randn_mat((128,))
    y = C.randn_mat((128,))

    # Calculate dot products
    dot_x_y = dot_product(x, conjugate(y), dim=0)
    dot_x_x = dot_product(x, conjugate(x), dim=0)
    dot_y_y = dot_product(y, conjugate(y), dim=0)

    # Check the inequality |<x, y>|^2 ≤ <x, x> * <y, y>
    cauchy_schwarz_holds = torch.square(torch.abs(dot_x_y)) <= (dot_x_x * dot_y_y)

    assert torch.all(cauchy_schwarz_holds).item(), "Cauchy-Schwarz inequality failed"


    #####
    # Range of Cosine Similarity

    # cos sim each individual scalar
    similarity = cosine_similarity(x, y, dim=0)
    assert torch.all(torch.logical_and(-1 <= similarity,
                                       similarity <= 1)).item(), "Cosine similarity should be within the range [-1, 1]"

    similarity = cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=1)
    assert torch.all(torch.logical_and(-1 <= similarity,
                                       similarity <= 1)).item(), "Cosine similarity should be within the range [-1, 1]"

    #####
    # Identity of Indiscernibles
    similarity = cosine_similarity(x, x, dim=0)
    assert torch.allclose(similarity, torch.tensor(1.0)), "Cosine similarity of a vector with itself should be 1"

    #####
    # Orthogonality
    # Generating orthogonal complex vectors is more intricate than real ones. Here we use a simpler approach

    for _ in range(10):
        x1 = C.randn_mat((1,1024))
        x2 = C.randn_mat((1,1024))
        similarity = cosine_similarity(x1, x2, dim=1)
        assert allclose(similarity, torch.tensor(0.0), atol=1e-1), "Cosine similarity should be 0 for orthogonal vectors"

    #####
    # Symmetry of Cosine Similarity with Conjugation

    # Per scalar
    similarity_xy = cosine_similarity(x, y, dim=0)
    similarity_yx = cosine_similarity(y, x, dim=0)
    assert torch.allclose(similarity_xy, similarity_yx), "Cosine similarity should respect conjugation symmetry"

    # Per vector
    similarity_xy = cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=1)
    similarity_yx = cosine_similarity(y.unsqueeze(0), x.unsqueeze(0), dim=1)
    assert torch.allclose(similarity_xy, similarity_yx), "Cosine similarity should respect conjugation symmetry"

    # Per list of vectors
    for _ in range(10):
        x1 = C.randn_mat((10, 256))
        x2 = C.randn_mat((10, 256))
        out = cosine_similarity(x1, x2, dim=1)
        assert out.shape == torch.Size((10,))
        assert torch.all(out.abs() <= 1)
        assert torch.all(out.abs() >= 0)

    print('Complex Tests Passed')

    #####
    # More

    # x1 = torch.tensor([[[ 2,  1],
    #                     [-1,  2.]]])
    # x2 = torch.tensor([[[-0.01,  1.363e-03],
    #                     [-2.279e-03,  0.01]]])

    # dim = 0
    # dot = dot_product(x1, conjugate(x2), dim=dim)
    # n1 = norm(x1, dim)
    # n2 = norm(x2, dim)
    # out = dot / (n1 * n2)
    # print(out)

    # BR

    # s1 = cosine_similarity(x1, x2, dim=0)
    # s2 = cosine_similarity(x1, x2, dim=1)
    # print(s1)
    # print(s2)
    # BR


##################################################
# Quaternion Tests

if RUN_QUATERNION_TESTS:
    Q = Quaternion
    #####
    # Syntax

    # Example usage
    q1 = Q.to_mat([1, 2, 3, 4])
    mq = torch.tensor([
        [1, -2, -3, -4],
        [2, 1, -4, 3],
        [3, 4, 1, -2],
        [4, -3, 2, 1.],
    ])
    assert allclose(q1, mq), f'\nExpected:\n{mq}\nActual:\n{q1}'

    # Round trip test for MQuat and UnMQuat
    q1 = Q.to_mat([1, 2, 3, 4])
    uq = Q.from_mat(q1)
    assert torch.allclose(uq, torch.tensor([1,2,3,4.])), f'wrong: {uq}'
    muq = Q.to_mat(uq)
    assert torch.allclose(q1, muq)

    ##########

    # Test quaternion multiplication for known values
    q1 = Q.to_mat([1, 0, 0, 0])  # Identity quaternion
    q2 = Q.to_mat([1, 2, 3, 4])
    expected = q2  # Multiplying with identity should give the same quaternion
    result = hadamard(q1, q2)
    assert allclose(result, expected), f"Quaternion multiplication failed: {result} != {expected}"

    # Test commutativity: q1 * q2 != q2 * q1 generally for quaternions
    q1 = Q.to_mat([1, 2, 3, 4])
    q2 = Q.to_mat([5, 6, 7, 8])
    result1 = hadamard(q1, q2)
    result2 = hadamard(q2, q1)
    assert not allclose(result1, result2), "Quaternion multiplication should not be commutative"

    # Test quaternion conjugation
    q1 = Q.to_mat([1, 2, 3, 4])
    expected = Q.to_mat([1, -2, -3, -4])
    result = conjugate(q1)
    assert allclose(result, expected), f"Quaternion conjugation failed: {result} != {expected}"

    # Test quaternion norm
    q1 = Q.to_mat([1, 2, 3, 4])
    expected_norm = torch.sqrt(torch.tensor(1**2 + 2**2 + 3**2 + 4**2))
    result_norm = norm(q1, dim=0)
    assert allclose(result_norm, expected_norm), f"Quaternion norm calculation failed: {result_norm} != {expected_norm}"

    # Test dot product for orthogonal quaternions
    q1 = Q.to_mat([1, 0, 0, 0])
    q2 = Q.to_mat([0, 1, 0, 0])
    expected_dot = torch.tensor(0.0)
    result_dot = dot_product(q1, q2, dim=0)
    assert allclose(result_dot, expected_dot), f"Dot product for orthogonal quaternions failed: {result_dot} != {expected_dot}"

    # Test dot product for the same quaternion
    q1 = Q.to_mat([1, 2, 3, 4])
    expected_dot = torch.sum(torch.tensor([1, 2, 3, 4.]) ** 2)
    result_dot = dot_product(q1, conjugate(q1), dim=0)
    assert allclose(result_dot, expected_dot), f"Dot product for the same quaternion failed: {result_dot} != {expected_dot}"

    # Test conversion from tensor to matrix form and back
    q_tensor = torch.tensor([1, 2, 3, 4.])
    q_matrix = Q.to_mat(q_tensor)
    q_tensor_converted = Q.from_mat(q_matrix)
    assert torch.allclose(q_tensor, q_tensor_converted), f"Round trip conversion failed: {q_tensor} != {q_tensor_converted}"

    #####
    # Cosine Similarity

    # Test 1: Cosine similarity of a quaternion with itself should be 1
    q1 = Q.to_mat([1, 2, 3, 4])
    cos_sim = cosine_similarity(q1, q1, dim=0)
    assert allclose(cos_sim, torch.tensor(1.0)), "Cosine similarity with itself should be 1"

    # Test 2: Cosine similarity between orthogonal quaternions should be 0
    q1 = Q.to_mat([1, 0, 0, 0])
    q2 = Q.to_mat([0, 1, 0, 0])
    cos_sim = cosine_similarity(q1, q2, dim=0)
    assert allclose(cos_sim, torch.tensor(0.0)), "Cosine similarity between orthogonal quaternions should be 0"

    # Test 3: Cosine similarity between non-orthogonal quaternions
    q1 = Q.to_mat([1, 2, 3, 4])
    q2 = Q.to_mat([-2, 1, 4, 3])
    expected_result = (1*(-2) + 2*1 + 3*4 + 4*3) / (torch.sqrt(torch.tensor(1**2 + 2**2 + 3**2 + 4**2)) * torch.sqrt(torch.tensor((-2)**2 + 1**2 + 4**2 + 3**2)))
    cos_sim = cosine_similarity(q1, q2, dim=0)
    assert allclose(cos_sim, expected_result), f"Cosine similarity failed: Expected {expected_result}, got {cos_sim}"

    # Test 4: Vector
    q1 = torch.stack([Q.to_mat([1, 2, 3, 4]), Q.to_mat([1, 2, 3, 4]), Q.to_mat([1, 2, 3, 4])]).unsqueeze(0)
    q2 = torch.stack([Q.to_mat([1, 2, 3, 4]), Q.to_mat([1, 2, 3, 4]), Q.to_mat([1, 2, 3, 4])]).unsqueeze(0)
    cos_sim = cosine_similarity(q1, q2, dim=1)
    assert allclose(cos_sim, torch.tensor(1.0)), "Cosine similarity with itself should be 1"


    # Per list of vectors
    for _ in range(10):
        x1 = Q.to_mat(Q.randn((10, 256)))
        x2 = Q.to_mat(Q.randn((10, 256)))
        out = cosine_similarity(x1, x2, dim=1)
        assert out.shape == torch.Size((10,))
        assert torch.all(out.abs() <= 1)
        assert torch.all(out.abs() >= 0)

    print('Quaternion Tests Passed.')
