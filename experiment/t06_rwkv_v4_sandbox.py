'''

Start adding NLambda to RWKV

RESULTS: incomplete


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
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""

try:
    already_loaded
except:
    model_name, revision = "RWKV/rwkv-5-world-1b5", "a2d9eeb70aa2095fb81a05b0a306b1fb0e5a6547"
    # model_name, revision = "RWKV/rwkv-4-world-169m", "598b039e6be5298de3f6f82ee373b6acdfae2858"

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, revision=revision, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    already_loaded = True

text = "Hi review of product plz. Apple iPhone v18."
prompt = generate_prompt(text)

inputs = tokenizer(prompt, return_tensors="pt").to(0)
output = model.generate(inputs["input_ids"],
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.0,
                        top_p=0.3,
                        top_k=0, )
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
