'''

Train 1D Hypernets 02



'''


import os
import warnings
import random
from typing import Callable, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, LayerNorm
import torch.optim as optim
from einops.layers.torch import Rearrange
from torch import einsum

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as Q
import tokenizers
from tokenizers import Tokenizer

from neurallambda.lab.common import print_model_info
import t02_train_1d_serial_attn as Attention
import importlib

try:
    importlib.reload(arc_like)
    importlib.reload(Horizontal)
    print('RELOADED MODULES')
except NameError:
    import t01_data_arc_like as arc_like
    import neurallambda.model.recurrent_transformer as Recurrent

SEED = 152
torch.manual_seed(152)
random.seed(SEED)

# Recurrent.MultiHeadAttention
# Recurrent.DecoderLayer
