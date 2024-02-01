'''

Vector Symbols

'''


from torch import einsum, tensor, allclose
from typing import Dict, Union, List, Tuple, Union, List, Any, Type
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################
# Build symbol map

nums = list(range(-1000, 1000 + 1))
chars = (
    'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ') +
    'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' ')
)
default_symbols = nums + chars

def symbol_map(vec_size, symbols=default_symbols, device='cpu'):

    symbols_i2v = {i:v for i,v in enumerate(symbols)}
    symbols_v2i = {v:i for i,v in symbols_i2v.items()}

    symbols_vec = torch.stack([
        torch.randn((vec_size,))
        for _ in range(len(symbols_i2v))
    ]).to(device)


    def project(val):
        """Projects an integer to a vector space."""
        index = symbols_v2i[val]
        return symbols_vec[index]

    def unproject(vector):
        """Unprojects a vector from the vector space back to an integer.

        Assumes matrix formatted `vector`.
        """
        cs = torch.cosine_similarity(vector.unsqueeze(0), symbols_vec, dim=1)
        sim_ix = torch.argmax(cs).item()
        return symbols_i2v[sim_ix]

    # Test round tripping
    for v in symbols:
        rv = unproject(project(v))
        assert v == rv, f'Value {v} round-tripped to {rv}'

    return project, unproject, symbols_i2v, symbols_v2i, symbols_vec


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
