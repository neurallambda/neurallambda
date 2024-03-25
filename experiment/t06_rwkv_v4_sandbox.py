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

'''

import torch
import torch.nn as nn

# from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["RWKV_JIT_ON"] = "0"
import neurallambda.rwkv_v5.model as rwkv


torch.manual_seed(42)


# os.environ["RWKV_JIT_ON"] = '1'
# os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
# from rwkv.model import RWKV                         # pip install rwkv
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16')

# out, state = model.forward([187, 510, 1563, 310, 247], None)   # use 20B_tokenizer.json
# print(out.detach().cpu().numpy())                   # get logits
# out, state = model.forward([187, 510], None)
# out, state = model.forward([1563], state)           # RNN has state (use deepcopy if you want to clone it)
# out, state = model.forward([310, 247], state)
# print(out.detach().cpu().numpy())

BREAK

##################################################
# Load model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_prompt(prompt):
    prompt = prompt.strip().replace('\r\n','\n').replace('\n\n','\n')

    return f'''User: {prompt}

Assistant:'''

try:
    already_loaded
except:
    # model_name, revision = "RWKV/rwkv-5-world-1b5", "a2d9eeb70aa2095fb81a05b0a306b1fb0e5a6547"
    model_name, revision = "RWKV/rwkv-4-world-169m", "598b039e6be5298de3f6f82ee373b6acdfae2858"

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, revision=revision, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    already_loaded = True


##################################################
# Modify


class CustomRwkvBlock(nn.Module):
    def __init__(self, original_module, model_name):
        super(CustomRwkvBlock, self).__init__()
        self.original_module = original_module
        self.model_name = model_name

    def forward(self, *args, **kwargs):
        '''
        NOTES on 169m:
          x: torch.Size([1, 15, 768])
          state: list, len == 5; all shapes == torch.Size([1, 768, 12])
          use_cache == True
          output_attentions == False
        '''

        if '169m' in model_name:
            x = args[0]
            state, use_cache, output_attentions = kwargs.values()
            return self.original_module(x, state, use_cache, output_attentions)

        elif '1b5' in self.model_name:
            x = args[0]
            state, use_cache, output_attentions, seq_mode = kwargs.values()

            new_state = []
            for s in state:
                s_ = s + torch.randn_like(s) * 1e-1
                new_state.append(s_)
            state = new_state

            x = x + torch.randn_like(x) * 1e-1

            return self.original_module(x, state, use_cache, output_attentions, seq_mode)

    def __getattr__(self, name):
        """If an attribute is not found in this custom module, try to find it
        in the original_module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_module, name)

#####
# Replace an RWKV block
block_id = 10

# replace last experimental block with the original (handy if experimenting in
# an interpreter)
if str(type(model.rwkv.blocks[block_id])) == "<class '__main__.CustomRwkvBlock'>":
    print('replacing')
    model.rwkv.blocks[block_id] = model.rwkv.blocks[block_id].original_module

# sub in new experimental block
original_block = model.rwkv.blocks[block_id]
custom_block = CustomRwkvBlock(original_block, model_name)
model.rwkv.blocks[block_id] = custom_block


##################################################
# Run

text = "What's the weather like on the Sun?"
prompt = generate_prompt(text)

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(**inputs, # ["input_ids"],
                        max_new_tokens=128,
                        do_sample=False,
                        # temperature=0.1,
                        # top_p=0.3,
                        # top_k=2,
                        )
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
