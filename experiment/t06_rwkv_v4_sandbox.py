'''

Start adding NLambda to RWKV

RESULTS:
 incomplete


DEPENDENCIES:
  git clone https://github.com/BlinkDL/RWKV-LM
  ln -s RWKV-LM/RWKV-v5/src/ neurallambda/rwkv-v5



----------
INFERENCE MEM USAGE:

| 1b5  | float32 | 6.3GB |
| 1b5  | float16 | 3.5GB |
| 169M | float32 | 1GB   |
| 169M | float16 | 0.6GB |
| 0.4B | float32 | 2.9GB |

----------
RWKV-5 1.5B

Rwkv5ForCausalLM(
  (rwkv): Rwkv5Model(
    (embeddings): Embedding(65536, 2048)
    (blocks): ModuleList(
      (0): RwkvBlock(
        (pre_ln): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (ln1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (attention): RwkvSelfAttention(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (key): Linear(in_features=2048, out_features=2048, bias=False)
          (value): Linear(in_features=2048, out_features=2048, bias=False)
          (receptance): Linear(in_features=2048, out_features=2048, bias=False)
          (gate): Linear(in_features=2048, out_features=2048, bias=False)
          (output): Linear(in_features=2048, out_features=2048, bias=False)
          (ln_x): GroupNorm(32, 2048, eps=1e-05, affine=True)
        )
        (feed_forward): RwkvFeedForward(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (key): Linear(in_features=2048, out_features=7168, bias=False)
          (receptance): Linear(in_features=2048, out_features=2048, bias=False)
          (value): Linear(in_features=7168, out_features=2048, bias=False)
        )
      )
      (1-23): 23 x RwkvBlock(
        (ln1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (attention): RwkvSelfAttention(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (key): Linear(in_features=2048, out_features=2048, bias=False)
          (value): Linear(in_features=2048, out_features=2048, bias=False)
          (receptance): Linear(in_features=2048, out_features=2048, bias=False)
          (gate): Linear(in_features=2048, out_features=2048, bias=False)
          (output): Linear(in_features=2048, out_features=2048, bias=False)
          (ln_x): GroupNorm(32, 2048, eps=1e-05, affine=True)
        )
        (feed_forward): RwkvFeedForward(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (key): Linear(in_features=2048, out_features=7168, bias=False)
          (receptance): Linear(in_features=2048, out_features=2048, bias=False)
          (value): Linear(in_features=7168, out_features=2048, bias=False)
        )
      )
    )
    (ln_out): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (head): Linear(in_features=2048, out_features=65536, bias=False)
)



PROVENANCE:
  https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v5_demo.py


'''

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = 'cuda'

model_path = os.path.expanduser('~/_/models/rwkv/RWKV-5-World-0.4B-v2-20231113-ctx4096.pth')
# model_path = os.path.expanduser('~/_/models/rwkv/RWKV-5-World-0.1B-v1-20230803-ctx4096.pth')
# model_path = os.path.expanduser('~/_/models/rwkv/RWKV-5-World-3B-v2-20231113-ctx4096.pth')
# model_path = os.path.expanduser('~/_/models/rwkv/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth')
model_path = (model_path)

vocab_path = os.path.expanduser('~/_/models/rwkv/rwkv_vocab_v20230424.txt')


##################################################

def log_object_structure(obj, indent=0):
    """
    Recursively logs the structure of a Python object. It focuses on nn.Module,
    types.SimpleNamespace, and tensors, and handles nesting within common Python
    containers like lists and dictionaries. For tensors, only their shapes are printed.

    Parameters:
    - obj: The Python object to log.
    - indent: The current indentation level (used for recursive calls).
    """
    if isinstance(obj, nn.Module):
        print("\t" * indent + f"{obj.__class__.__name__} (nn.Module):", )
        for name, module in obj.named_children():
            log_object_structure(module, indent + 1)
        for name, param in obj.named_parameters(recurse=False):
            print("\t" * (indent + 1) + f"{name}: Tensor of shape {param.size()}", )
    elif isinstance(obj, types.SimpleNamespace):
        # print("\t" * indent + "SimpleNamespace:", )
        for name, value in vars(obj).items():
            print("\t" * (indent + 1) + f"{name}:", )
            log_object_structure(value, indent + 1)
    elif torch.is_tensor(obj):
        print("\t" * indent + f"tensor of shape {obj.size()}", )
        print()
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            print("\t" * indent + f"[{i}]:", )
            log_object_structure(item, indent + 1)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            print("\t" * indent + f"{key}:", )
            log_object_structure(value, indent + 1)
    elif hasattr(obj, "__dict__"):
        print("\t" * indent + f"{obj.__class__.__name__}:", )
        for name, value in vars(obj).items():
            log_object_structure(value, indent + 1)
    else:
        print("\t" * indent + str(obj), )

# log_object_structure(model.w)
# BRK


##################################################

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()


########################################################################################################

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out, dim=-1).cpu().numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out

########################################################################################################


tokenizer = RWKV_TOKENIZER(vocab_path)

# THIS IS NOW UPDATED TO SUPPORT LATEST RWKV-5 WORLD v2 MODELS

args = types.SimpleNamespace()
args.MODEL_NAME = model_path
args.n_layer = 24
args.n_embd = 1024  # 768
args.vocab_size = 65536

context = "\nA neurallambda is"
NUM_TRIALS = 3
LENGTH_PER_TRIAL = 100
TEMPERATURE = 1.0
TOP_P = 0.7

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        w = torch.load(args.MODEL_NAME, map_location=DEVICE)
        for k in w.keys():
            w[k] = w[k].float() # convert to f32 type
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = torch.exp(-torch.exp(w[k])).unsqueeze(-1)
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

        self.n_head = w['blocks.0.att.time_decay'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head

        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        i0 = (2+self.head_size)*i+0
        xk = x * time_mix_k + state[i0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[i0] * (1 - time_mix_r)
        state[i0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    @MyFunction
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_mix_g, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
        H = self.n_head
        S = self.head_size

        i1 = (2+S)*i+1
        xk = x * time_mix_k + state[i1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[i1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[i1] * (1 - time_mix_r)
        xg = x * time_mix_g + state[i1] * (1 - time_mix_g)
        state[i1] = x

        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = F.silu(gw @ xg)

        s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (time_first * a + s)
        s = a + time_decay * s

        state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1)
        x = x.flatten()

        x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze(0) * g # same as gn(x/8, eps=1e-5)
        return ow @ x

    def forward(self, token, state):
        if state == None:
            state = torch.zeros(self.args.n_layer * (2+self.head_size), self.args.n_embd, device=DEVICE)

        x = self.w.emb.weight[token]
        x = self.layer_norm(x, self.w.blocks[0].ln0)
        for i in range(self.args.n_layer):
            if hasattr(self.w.blocks[i], 'pre_fn'):
                x, state = self.w.blocks[i].pre_fn(x, state)
            att = self.w.blocks[i].att
            x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i,
                att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_mix_g, att.time_faaaa, att.time_decay,
                att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                att.ln_x.weight, att.ln_x.bias)
            ffn = self.w.blocks[i].ffn
            x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i,
                ffn.time_mix_k, ffn.time_mix_r,
                ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

        x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
        return x.float(), state

print(f'Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)


#####
# Replace an RWKV block
block_id = 2

def pre_fn(x, state):
    '''
    (Pdb) x.shape
    torch.Size([1024])
    (Pdb) state.shape
    torch.Size([1584, 1024])
    '''
    return x, state

model.w.blocks[block_id].pre_fn = pre_fn


##########
# Go

model.eval()

with torch.no_grad():
    print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
    init_state = None
    for token in tokenizer.encode(context):
        init_out, init_state = model.forward(token, init_state)

    for TRIAL in range(NUM_TRIALS):
        print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
        all_tokens = []
        out_last = 0
        out, state = init_out.clone(), init_state.clone()
        for i in range(LENGTH_PER_TRIAL):
            token = sample_logits(out, TEMPERATURE, TOP_P)
            all_tokens += [token]
            try:
                tmp = tokenizer.decode(all_tokens[out_last:])
                if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                    print(tmp, end="", flush=True)
                    out_last = i + 1
            except:
                pass
            out, state = model.forward(token, state)
    print('\n')
