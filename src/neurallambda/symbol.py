'''

Vector Symbols

'''


from torch import einsum, tensor, allclose
from typing import Dict, Union, List, Tuple, Union, List, Any, Type
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


##################################################
# Build symbol map

nums = list(range(-1000, 1000 + 1))
chars = (
    'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ') +
    'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' ')
)
default_symbols = nums + chars

class SymbolMapper:
    def __init__(self, vec_size, symbols=default_symbols, device='cpu'):
        self.device = device
        self.symbols_i2v = {i: v for i, v in enumerate(symbols)}
        self.symbols_v2i = {v: i for i, v in self.symbols_i2v.items()}


        # nbits = math.ceil(torch.log(torch.tensor([len(symbols) * 1.0])) / torch.log(torch.tensor([2])))
        # n_projections = math.ceil(vec_size / nbits)
        # self.int_mapper = IntMapper(nbits, n_projections)

        self.symbols_vec = torch.stack([

            # self.int_mapper.project(i)[:vec_size] # the nbit-n_projections stuff might go over, so trim back
            #   if isinstance(i, int)
            #   else torch.randn((vec_size,))

            torch.randn((vec_size,))

            for i in range(len(self.symbols_i2v))
        ]).to(self.device)

    def project(self, val):
        """Projects an integer to a vector space."""
        index = self.symbols_v2i[val]
        return self.symbols_vec[index]

    def unproject(self, vector, return_sim=False):
        """Unprojects a vector from the vector space back to an integer.
        """
        cs = torch.cosine_similarity(vector.unsqueeze(0), self.symbols_vec, dim=1)
        am = torch.argmax(cs)
        sim_ix = am.item()
        if return_sim:
            return self.symbols_i2v[sim_ix], cs.max()
        else:
            return self.symbols_i2v[sim_ix]


##################################################
# IntMapper

def initialize_projection_matrix(bit_string_size, n_projections):
    """
    Initializes a binary projection matrix with 50% ones and 50% zeros.
    """
    # For each projection, create a flat array with half ones and half zeros
    half_size = bit_string_size // 2
    # int8 is fine because it's just binary
    balanced_array = torch.cat((torch.ones(half_size, dtype=torch.int8), torch.zeros(half_size, dtype=torch.int8)))

    # Initialize an empty tensor for the projection matrix
    projection_matrix = torch.empty((n_projections, bit_string_size), dtype=torch.int8)

    for i in range(n_projections):
        # Randomly shuffle the balanced array to ensure a random distribution of ones and zeros
        projection_matrix[i] = balanced_array[torch.randperm(bit_string_size)]

    return projection_matrix

def int_to_binary_vector(integer, bit_string_size):
    """
    Converts an integer to a binary vector, supports negative numbers using two's complement.
    """
    # Adjust for Python's unlimited precision integers: mask to simulate fixed-width
    mask = 2**(bit_string_size) - 1
    # Convert to two's complement binary representation
    twos_complement = (integer + (1 << bit_string_size)) % (1 << bit_string_size)
    binary_representation = [(twos_complement >> bit) & 1 for bit in range(bit_string_size-1, -1, -1)]
    return torch.tensor(binary_representation, dtype=torch.int8)

def binary_vector_to_int(binary_vector):
    """
    Converts a binary vector back to an integer, assumes two's complement for negative numbers.
    """
    bit_string_size = binary_vector.size(0)
    # Convert from binary vector to integer
    integer = sum([bit.item() * (2**idx) for idx, bit in enumerate(reversed(binary_vector))])
    # Adjust for two's complement if the sign bit is set
    if binary_vector[0] == 1:  # If the sign bit is set
        integer -= 2**bit_string_size
    return int(integer)

class IntMapper:
    def __init__(self, bit_string_size, n_projections):
        self.bit_string_size = bit_string_size
        self.projection_matrix = initialize_projection_matrix(bit_string_size, n_projections)

    def project(self, integer, dtype=torch.float32):
        binary_vector = int_to_binary_vector(integer, self.bit_string_size)
        projected = torch.zeros((self.projection_matrix.size(0) * self.bit_string_size), dtype=dtype)
        for i in range(self.projection_matrix.size(0)):
            xor_result = binary_vector ^ self.projection_matrix[i]
            if dtype == torch.float32:
                projected[i * self.bit_string_size:(i + 1) * self.bit_string_size] = xor_result.float() * 2 - 1
            else:
                projected[i * self.bit_string_size:(i + 1) * self.bit_string_size] = xor_result
        return projected

    def unproject(self, projected_vector):
        if projected_vector.dtype == torch.float32:
            projected_vector = (projected_vector > 0).int()

        binary_vector = projected_vector.view(self.projection_matrix.size(0), -1)
        recovered_vectors = torch.zeros_like(self.projection_matrix, dtype=torch.int8)
        for i in range(self.projection_matrix.size(0)):
            recovered_vectors[i] = binary_vector[i] ^ self.projection_matrix[i]

        avg_vector = torch.round(recovered_vectors.float().mean(dim=0))
        return binary_vector_to_int(avg_vector)



##################################################
# String Encodings (DEPRECATED)

# Letters
chars = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'.split(' ')
char_to_int_ = {c: i for i, c in enumerate(chars)}
def char_to_int(c):
    if c in char_to_int_:
        return char_to_int_[c]
    return -666  # an error, but still an int

int_to_char_ = {i: c for i, c in enumerate(chars)}
def int_to_char(i):
    if i in int_to_char_:
        return int_to_char_[i]
    return '#'  # an error, but still a char

# ArithOp
arithops = '+ - / *'.split(' ')
arithop_to_int = {c: i for i, c in enumerate(arithops)}
int_to_arithop = {i: c for i, c in enumerate(arithops)}
