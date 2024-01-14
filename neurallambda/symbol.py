'''

Vector Symbols

'''


from torch import einsum, tensor, allclose
from typing import Dict, Union, List, Tuple, Union, List, Any, Type
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


##########
# String Encodings

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
