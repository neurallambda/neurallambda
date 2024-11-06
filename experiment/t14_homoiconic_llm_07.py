'''.

Apply Metalearning-Hypernet stuff to Transformer arch

This version incorporates METALEARNING.

PROVENANCE:
- t14_homoiconic_llm_05.py

NaN CAUSES:
- cross_entropy loss where everything is masked out
- ragged batch sizes
- bfloat16?
- clip grads?


TODO: STABILIZE TRAINING
- [ ] teacher forcing
- [ ] test time training
- [ ] init lormodules to output top eigenvectors of matched linear layers?
- [X] curriculum learning (simpler data (fewer variable indirections), for starters)
- [X] increasing dataset size helped, there's seems to be a minimal count necessary
- [X] adding problem text to loss seems to stabilize training, and it appears
      to still be able to overfit training. I suspect an issue with this though,
      that problem syntax might be predictable, but not specifics, so this might
      hurt the quality

NOTES:
- LayerNorm on end of LORModule encouraged weights to move more, without, they imperceptibly moved (also used high lr (1e-2))
- Biases in LORModule can cause non-lor layers (batch effect), which receive 0 values, to suddenly have values, and corrupt performance. LORModules should have the property that they cannot alter non-LOR'd samples.
- swiglu lor modules seem more stable to train than linear-only, and better val loss


DESIGN CHOICES:  these should be followed up on
  - [X] init LORProject for low impact initially. Setting the final output linear to 0s.
  - [X] LoRs should probably be scaled by sqrt(2/dim) (see t14_homoiconic_llm_adding_data_to_mlp)
  - [X] How should RMSNorm be initialized? For now, we're setting it low. By setting it even to 0, this effectively removes a layer from the net, but residuals still pass through.
  - [ ] uncausal masking of metatokens?
  - [ ] position encodings on metatokens?


POSSIBLE BUGS:
  - [ ] Test uncausal masking in `model`, that it's not buggy, or off by 1. Has complexity with cache_position, position_ids, past_key_values, and attention_mask.
  - [ ] Test uncausal masking data generation side
  - [ ] dbl check loss_mask, and off-by-one, and include_in_loss
  - [ ] dbl check off-by-one shifts in data/parsing lors/loss

QUESTIONS
  - [ ] How does a metatoken/metaweight know what to focus on, to succeed?
  - [ ] Separate loss for metatokens/metaweights? (IE train-time inner-loop SGD to construct part of the dataset on-the-go?)


LIBRARIFY
  - [ ] dataset construction tools
  - [ ] good test suite


TODO FIX:

- [X] Custom embeds per metaweight pipe | type
  - [X] experiment with active forgetting: fuzz new embeddings every n-steps. RESULTS: preliminary results suggest this helped reduce overfitting early, but loss plateaud at worse value, and later, overfit anyway.

- [ ] Delta Rule in LorProject

- [ ] Tie U&D "intermediate"

- [ ] METALEARNING
  - [ ] multiview reconstruction, using delta rule
    does it need to see only metatokens (which presumably have stuff routed to em), or previous tokens too?
  - [ ] online dataset: metalearn weights in inner-loop, MSE of these weights vs initial in outer loop
  - [ ] parallelize layers


- [ ] Instead of parsing ixs, parse spans (cleaner).

- [ ] Test LORProjection batch effect, and LORNorm
  - 'LORProjection after training should have an effect'
  - 'LORProjection after training should still return 0 for 0-inputs'

- [ ] COMMON WEIGHT GENERATION
  - [X] separate the QKVOGUD projection stuff from the normalization model. 1 projection model, separate norm models per QKVOGUD.
  - [X] all parses go into one weight head together
  - [X] fix failing batch vs single assertions: the cause was Dropout
  - [X] make lorproject mlpmixer
  - [X] different LRs per parameter
  - [X] lorproject has to be causally unmasked
  - [X] save and restore models
  - [ ] columnwise_collate_fn: ix parse keys have wrong representation (should be list, num_layers long)
  - [ ] should metatokens have position encodings?
  - [ ] separate loss for metatokens?
  - [ ] Does LORProject support GUD "sharing" an intermediate value to tie the KV storage together (ie, results from t14_homoiconic_llm_adding_data_to_mlp)
  - [ ] upgrade parseixs to parse spans instead of individual ixs
- [X] TEST that a lor module can have a KV stored, then queried back out
- [X] loss_fn is currently unbatched
- [X] need to iterate over list of per-layer ixs
- [X] later, how to connect lor_cache with lor_model? in the model module, its now just passing in the module instead of the cache
- [X] replace `lor` dep in model
- [X] add changes to `generate`
- [X] general clean up
- [ ] add correctness tests for lor stuff
  - [X] batch mode gives same output as non-batched
  - [X] parse ixs don't cause weirdness in non-parsed samples)
  - [?] ensure cache looks good (with test)
  - [ ] double check that lormodule init produces LRx=0 before any training, make it a test. Actually LRx=0 isn't enough if a Norm follows.
- [X] `generate` should use `forward_columns`
- [ ] train RMSNorm separately, first?
- [ ] trained model gets low loss, then outputs "^@Q" as next token?!
- [ ] OFF BY ONE ERROR IN LOSS?!
- [X] don't init lor projection to 0 (bc of rmsnorm)
- [ ] non-parsed ixs could be masked out at cache, instead of projecting the 0s
- [ ] MAP EVERYTHING OUT, like, how does `var\\_\\1` map attention to the right place? what should it be saving to the cache?
- [ ] dont need norm after D lor

- [X] smaller dataset (fewer vars)
- [X] only train gud for now
- [X] overfit a single sample. If this doesn't work, test time training won't, nor teacher forcing.
- [X] LORModule weights not updating much
- [ ] when training loss hits 0, @ inference it gets training probs wrong
- [X] right singular vectors not updating
- [X] make use of lor_qkvo in model
- [X] replace F.interpolate with a Linear Forward in LORModule
- [X] i flipped lor_i and lor_o from lor_gud, fix
- [X] make sure to properly initialize params in LORModule


TODO:
- [ ] teacher forcing by taking preceding clause, ex x=1, and setting QKVOGUD to relevant portions of that
- [ ] teacher forcing (via training lor weights one step, then using those)
- [ ] implement LOR parsing during inference's autoregressive portion
- [ ] lor_ixs makes lor_mask irrelevant? maybe not...
- [X] aggregate LORs instead of replace
- [X] if not training text tokens, consider tracing what the outputs actually are

BATCH EFFECTS
- non-parsed batch sample affected by being in a batch with some parsed samples. Means cache gets populated with 0 for non-parsed samples.

OPTIM:
- [ ] multi-gpu
- [ ] AMP mixed precision


IDEAS:

- [ ] after a LOR block, forbid attention to preceding tokens (or just make
      RNN). This could provide an info bottleneck, and force lors to be highly
      relevant.
- [ ] instead of interpretting hidden state as metaweights, interpret traces of Q/K/V/etc projections?
- [ ] query back out matrix->vector. We have vec->mat, but not the reverse.

'''

PAUSED, pending experiment with t14_homoiconic_llm_test_metalearning_2


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # let cuda give debug info

# # python interpreter, why do you start in the dir you do??
# here = os.getcwd()
# if here.endswith('neurallambda'):
#     print('changing working dir to ./experiment')
#     os.chdir('./experiment')

# import re
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.cache_utils import Cache

# from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
# from datasets import load_dataset

# from typing import Optional, Tuple, Union, List, Dict, Any
# import warnings
# import math

# from datasets import load_dataset, Dataset
# from torch.utils.data import DataLoader, random_split
# from torch.optim import AdamW

# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import random
# import time
# import warnings
# import itertools

# from torch.utils.tensorboard import SummaryWriter

# from neurallambda.lab.common import print_model_info
# from neurallambda.torch import get_last_attended_token

# import t14_homoiconic_llm_add_tokens as AT

# from datasets import Dataset as HFDataset

# from dataclasses import dataclass
# import importlib
# from functools import partial

# from datetime import datetime
# current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# writer = SummaryWriter(f'log/t14_homoiconic_llm_05/{current_time}')
# LOG = True


# try:
#     importlib.reload(C)
#     print('RELOADING C')
# except NameError:
#     import t14_homoiconic_llm_columnize_06 as C

# try:
#     importlib.reload(Q)
#     print('RELOADING Q')
# except NameError:
#     import t14_homoiconic_llm_model_03 as Q


# SEED = 152
# torch.manual_seed(SEED)
# random.seed(SEED)

# # # Extra determinism
# # torch.cuda.manual_seed_all(SEED)
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False
# # torch.use_deterministic_algorithms(True)
# # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

# DEVICE = 'cuda:1'


# ##################################################
# # Saving/Loading

# save_path = 't14_homoiconic_llm_07'

# def get_next_run_number(save_path):
#     if os.path.exists(save_path):
#         files = os.listdir(save_path)
#         run_numbers = [int(re.search(r'run(\d+)', f).group(1)) for f in files if re.search(r'run(\d+)', f)]
#         return max(run_numbers, default=0) + 1
#     else:
#         return 0


# def save_model(name, run_number, model, save_path, val_loss, epoch):
#     os.makedirs(save_path, exist_ok=True)
#     filename = f'model_run{run_number}_{name}_epoch{epoch}_valloss{val_loss:.4f}.pth'
#     full_path = os.path.join(save_path, filename)

#     save_dict = {
#         'model_state_dict': model.state_dict(),
#         'architecture': str(model),  # Save architecture as string
#         'val_loss': val_loss,
#         'epoch': epoch,
#         'run_number': run_number
#     }

#     torch.save(save_dict, full_path)
#     print(f"Model saved to {full_path}")

# def load_model(model, load_path):
#     checkpoint = torch.load(load_path)

#     # Check if architecture matches
#     if str(model) != checkpoint['architecture']:
#         print("Warning: Current model architecture does not match saved architecture.")
#         print(f"Saved architecture: {checkpoint['architecture']}")
#         print(f"Current architecture: {str(model)}")

#         user_input = input("Do you want to proceed with loading? (y/n): ")
#         if user_input.lower() != 'y':
#             print("Model loading aborted.")
#             return

#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f"Model loaded from {load_path}")
#     print(f"Validation loss: {checkpoint['val_loss']:.4f}, Epoch: {checkpoint['epoch']}")
#     return checkpoint['epoch']


# ##################################################

# # Ex:
# #
# # "Hi there^Q^1||^K^1|| how are you?"
# #          ^     ^
# #          |     |
# #          |     At K weights of layer 1, 2 pipes define the L and R singular vectors
# #          At Q weights of layer 1, 2 pipes define the L and R singular values
# #

# # Pairs of new tokens with their initialization
# new_token_pairs = [
#     # Attention
#     ("^@Q", "&nbsp"),  # Q proj
#     ("^@Ql", "&nbsp"),  # left singular value
#     ("^@Qr", "&nbsp"),  # right singular value

#     ("^@K", "&nbsp"),  # K proj
#     ("^@Kl", "&nbsp"),
#     ("^@Kr", "&nbsp"),

#     ("^@V", "&nbsp"),  # V proj
#     ("^@Vl", "&nbsp"),
#     ("^@Vr", "&nbsp"),

#     ("^@O", "&nbsp"),  # O proj
#     ("^@Ol", "&nbsp"),
#     ("^@Or", "&nbsp"),

#     # MLP
#     ("^@G", "&nbsp"),  # Gate proj
#     ("^@Gl", "&nbsp"),
#     ("^@Gr", "&nbsp"),

#     ("^@U", "&nbsp"),  # Up proj
#     ("^@Ul", "&nbsp"),
#     ("^@Ur", "&nbsp"),

#     ("^@D", "&nbsp"),  # Down proj
#     ("^@Dl", "&nbsp"),
#     ("^@Dr", "&nbsp"),


#     # Layer identifier
#     #   Qwen1.5B has 28 layers, for testing we'll target
#     ("^@3", "&nbsp"),
#     ("^@24", "&nbsp"),

#     # # Dummy weight
#     # ("^@|", "&nbsp"),
# ]


# ##################################################
# # Lor stuff

# def create_position_ids(attention_mask):
#     """
#     Create position IDs based on the attention mask.
#     Set position to -1 for padding tokens.
#     """
#     position_ids = attention_mask.long().cumsum(-1) - 1
#     position_ids.masked_fill_(attention_mask == 0, -1)
#     return position_ids


# def empty_lors(num_layers):
#     '''per-layer cache of all lor blocks parsed during generation, and used in
# future generation. The None values at a given layer index will be replaced with
# a tuple of tensors shaped like ([BATCH, DIM, N], [BATCH, N, DIM]). N is the
# rank of the LOR matrix. In the current implementation, these are just
# concatenated, and never erased (within a batch, ofc). '''
#     lors = {
#         # low rank attention params
#         "lor_qs": [None] * num_layers,
#         "lor_ks": [None] * num_layers,
#         "lor_vs": [None] * num_layers,
#         "lor_os": [None] * num_layers,

#         # low rank mlp params
#         "lor_gs": [None] * num_layers,
#         "lor_us": [None] * num_layers,
#         "lor_ds": [None] * num_layers,
#     }
#     return lors


# def partially_apply_models(lor_models, lor_cache):
#     '''deep in the transformer stack, the LORModules get applied, but they need to
# reference the current lor cache. This is where they get it from.'''
#     fs = {}
#     for k in lor_ix_keys:  # TODO: rm global
#         fs[k] = []
#         for m, c in zip(lor_models[k], lor_cache[k]):
#             if m is not None:
#                 f = partial(m, c)
#             else:
#                 f = None
#             fs[k].append(f)
#     return fs


# def forward_columns(model, lor_models, lor_cache, input_idss, attention_masks, position_idss, lor_ixss, uncausal_masks):
#     '''Recurrently process column blocks of ids, concatenating attention_mask and
# past_key_values across generations. '''
#     col_out_logits = []
#     past_key_values = None
#     attention_mask = None
#     uncausal_mask = None

#     # Iterate over each column in the batch
#     for i, (input_ids, new_attention_mask, position_ids, lor_ixs, new_uncausal_mask) in enumerate(zip(input_idss, attention_masks, position_idss, lor_ixss, uncausal_masks)):

#         # skip empty batches. this can happen because of padded blocks.
#         if input_ids.numel() == 0:
#             continue

#         # `attention_mask` will continue to grow as the entire sequence length
#         # seen so far (even though input_ids will only be new inputs, not past)
#         if attention_mask is not None:
#             attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
#         else:
#             attention_mask = new_attention_mask

#         # `uncausal_mask` will continue to grow as the entire sequence length
#         # seen so far (even though input_ids will only be new inputs, not past)
#         if uncausal_mask is not None:
#             uncausal_mask = torch.cat([uncausal_mask, new_uncausal_mask], dim=-1)
#         else:
#             uncausal_mask = new_uncausal_mask

#         # give the lor models the current cache
#         lorm = partially_apply_models(lor_models, lor_cache)

#         # run column
#         out = model(input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     past_key_values=past_key_values,
#                     return_dict=True,
#                     use_cache=True,
#                     position_ids=position_ids,
#                     output_hidden_states=True,  # OPTIM: we don't need this for non-lor blocks
#                     uncausal_mask=uncausal_mask,
#                     **lorm,
#                     )
#         col_out_logits.append(out.logits)

#         past_key_values = out.past_key_values
#         assert past_key_values is not None, "past_key_values cannot be None. Typically `transformers` models won't return past_key_values during training, so, you need to modify the underlying model class."

#         lor_cache = update_lors(lor_models, lor_cache, lor_ixs, out.hidden_states)

#     return col_out_logits, attention_mask, past_key_values, lor_cache


# ##################################################
# # Training Fns


# def unfreeze_embeddings(grads, ixs):
#     """
#     Zero out gradients for all embeddings except those specified by ixs.

#     Args:
#     grads (torch.Tensor): Gradients of the embedding layer
#     ixs (list or torch.Tensor): Indices of the embeddings to keep unfrozen

#     Returns:
#     torch.Tensor: Modified gradients with appropriate values zeroed out
#     """
#     # Create a mask of zeros with the same shape as grads
#     mask = torch.zeros_like(grads)

#     # Set the mask to 1 for the indices we want to keep unfrozen
#     mask[ixs] = 1

#     # Multiply the original gradients by the mask
#     # This zeroes out gradients for all embeddings except those in ixs
#     out = grads * mask
#     return out


# # START_BLOCK_7

# def loss_fn(
#     col_batch_in: List[torch.Tensor],  # list of blocks of TOKENS IDS (dtype=int)
#     col_batch_out: List[torch.Tensor],  # list of blocks of EMBEDDINGS (dtype=float)
#     loss_mask: List[torch.Tensor]  # list of blocks of masks (dtype=bool)
# ):
#     vocab_size = col_batch_out[0].shape[2]

#     # Concatenate, shift, flatten
#     labels = torch.cat(col_batch_in, dim=1)[..., 1:].contiguous().view(-1)  # [B, S-1] -> [B * (S-1)]
#     logits = torch.cat(col_batch_out, dim=1)[..., :-1, :].contiguous().view(-1, vocab_size)  # [B, S-1, D] -> [B * (S-1), D]
#     # m = torch.cat(loss_mask, dim=1)[..., :-1].contiguous().view(-1)  # TODO: is this right?
#     m = torch.cat(loss_mask, dim=1)[..., 1:].contiguous().view(-1)

#     # Calculate masked loss
#     #
#     # NOTE: we're averaging per unmasked element here, bc that's efficient. An
#     # alternative might be averaging loss within samples, then averaging across
#     # samples. The difference shows up if samples have different counts of
#     # unmasked elements.
#     loss = F.cross_entropy(logits[m], labels[m], reduction='mean')
#     return loss



# def loss_fn_loop(
#     col_batch_in: List[torch.Tensor],  # list of blocks of TOKENS IDS (dtype=int)
#     col_batch_out: List[torch.Tensor],  # list of blocks of EMBEDDINGS (dtype=float)
#     loss_mask: List[torch.Tensor]  # list of blocks of masks (dtype=bool)
# ):
#     ''' TODO DEBUG for loop to make sure view isn't messing things up '''

#     # Concatenate, shift, flatten
#     labels = torch.cat(col_batch_in, dim=1)[..., 1:].contiguous()  # [B, S-1]
#     logits = torch.cat(col_batch_out, dim=1)[..., :-1, :].contiguous()  # [B, S-1, D]
#     m = torch.cat(loss_mask, dim=1)[..., 1:].contiguous()

#     B = labels.shape[0]

#     loss = 0
#     for b in range(B):
#         loss = loss + F.cross_entropy(logits[b][m[b]], labels[b][m[b]], reduction='mean')

#     loss = loss / B
#     return loss



# # @@@@@@@@@@
# # Test that the 2 loss functions are roughly equivalent (they're slightly different because of averaging per unmasked item vs averaging means across the entire batch).

# # Generate random input data
# batch_size = 2
# seq_length = 10
# vocab_size = 1000
# num_blocks = 3

# for _ in range(100):
#     col_batch_in = [torch.randint(0, vocab_size, (batch_size, seq_length)) for _ in range(num_blocks)]
#     col_batch_out = [torch.randn(batch_size, seq_length, vocab_size) for _ in range(num_blocks)]
#     loss_mask = [torch.randint(0, 2, (batch_size, seq_length)).bool() for _ in range(num_blocks)]

#     # Compute losses using both functions
#     loss_v1 = loss_fn(col_batch_in, col_batch_out, loss_mask)
#     loss_v2 = loss_fn_loop(col_batch_in, col_batch_out, loss_mask)
#     tolerance = 0.5
#     assert torch.isclose(loss_v1, loss_v2, atol=tolerance), f"Losses are not equal: {loss_v1} vs {loss_v2}"

# print("Test passed: Both loss functions produce the same result.")

# # @@@@@@@@@@

# # END_BLOCK_7




# def run_epoch(model, lor_models, dataloader, optimizer, device, train=True, debug=False):
#     model.train() if train else model.eval()
#     total_loss = 0
#     total_samples = 0

#     with torch.set_grad_enabled(train):
#         for batch_cols in tqdm(dataloader, desc="Training" if train else "Evaluating"):
#             B = batch_cols[0]['input_ids'].shape[0]
#             device = model.device

#             #####
#             # Move to cuda

#             # list of columns
#             input_idss = [x['input_ids'].to(device) for x in batch_cols]
#             attention_masks = [x['attention_mask'].to(device) for x in batch_cols]
#             position_idss = [x['position_ids'].to(device) for x in batch_cols]
#             loss_masks = [x['loss_mask'].to(device) for x in batch_cols]
#             uncausal_masks = [x['uncausal_mask'].to(device) for x in batch_cols]

#             # lor_ixs per column
#             lor_ixss = []
#             for col in batch_cols:
#                 lor_ixs = {k: (v[0].to(device), v[1].to(device)) for k, v in col.items() if k in lor_ix_keys}
#                 # v[0]: [layer, batch_size]; parse ixs, 1 (optional) index per layer for each sample
#                 # v[1]: [layer, batch_size]
#                 lor_ixss.append(lor_ixs)


#             #####
#             # Go

#             lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch
#             out_logits, cat_attention_mask, _, lor_cache = forward_columns(model, lor_models, lor_cache, input_idss, attention_masks, position_idss, lor_ixss, uncausal_masks)

#             loss = loss_fn(input_idss,
#                            out_logits,
#                            loss_masks)

#             if torch.isnan(loss):
#                 print('NaN encountered:')
#                 print(f'  all loss was masked out?: {sum([x.to(dtype=torch.long).sum() for x in loss_masks]) == 0}')
#                 print(f'  nan in out_logits?: {torch.cat(out_logits, dim=1).isnan().sum() > 0}')
#                 print('  ragged batch sizes? (double check by hand if necessary)')
#                 breakpoint()

#             if train:
#                 optimizer.zero_grad()
#                 loss.backward()

#                 # mask out all tokens except the new meta tokens
#                 if model.model.embed_tokens.weight.grad is not None:
#                     warnings.warn('freezing all token embeddings except new meta tokens')
#                     with torch.no_grad():
#                         model.model.embed_tokens.weight.grad[:] = unfreeze_embeddings(model.model.embed_tokens.weight.grad, new_token_ids)

#                 MAX_GRAD_NORM = 1.0
#                 nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
#                 nn.utils.clip_grad_norm_(lor_models.parameters(), max_norm=MAX_GRAD_NORM)

#                 optimizer.step()

#             total_loss += loss.item() * B
#             total_samples += B

#     avg_loss = total_loss / total_samples

#     # Log weight histogram
#     if LOG and train:
#         try:
#             # for name, param in itertools.chain(model.named_parameters(), lor_models.named_parameters()):
#             for name, param in lor_models.named_parameters():
#                 if param.requires_grad:
#                     writer.add_histogram(f'weights/{name}', param.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
#                     if param.grad is not None:
#                         writer.add_histogram(f'grads/{name}', param.grad.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
#         except Exception as e:
#             warnings.warn(f'Failed to write to tensorboard: {e}')

#     if debug:
#         warnings.warn('only debugging the last batch, values are not accumulated across batches (loss *is* averaged though)')
#         return avg_loss, out_logits, cat_attention_mask, lor_cache
#     else:
#         return avg_loss


# ##################################################
# # LOAD MODEL

# model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# # model_name = os.path.expanduser("~/_/models/Qwen2.5-0.5B")
# # model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
# # model_name = os.path.expanduser("~/_/models/Qwen2-7B")


# def hook_fn(module, input, output):
#     if isinstance(output, torch.Tensor):
#         isnan = torch.isnan(output).any()
#         isinf = torch.isinf(output).any()
#         if isnan or isinf:
#             print(f"NaN or inf detected in {module.__class__.__name__}: {isnan=}, {isinf=}")
#             print(f"Module: {module}")
#             print(f"Module name: {module_names.get(module, 'Unknown')}")
#             print(f"Input shape: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}")
#             print(f"Output shape: {output.shape}")
#             breakpoint()
#             # raise RuntimeError("NaN or inf detected: {isnan=}, {isinf=}")

# module_names = {}

# def add_hooks(model):
#     for name, module in model.named_modules():
#         module_names[module] = name
#         module.register_forward_hook(hook_fn)

# try:
#     fail
#     already_loaded
# except:
#     print('Loading model')
#     model = Q.Qwen2ForCausalLM.from_pretrained(
#         model_name,
#         # torch_dtype=torch.bfloat16,  # HERE BE DRAGONS
#         torch_dtype=torch.float32,
#         # torch_dtype=torch.float64,
#         device_map=DEVICE,
#         _attn_implementation='eager',
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = 'left'

#     num_layers = model.config.num_hidden_layers

#     # add new tokens
#     if False:
#         AT.assert_unique_tokens(tokenizer, [x[0] for x in new_token_pairs])

#     AT.add_and_initialize_tokens(model, tokenizer, new_token_pairs)

#     new_token_ids = [tokenizer.convert_tokens_to_ids(x[0]) for x in new_token_pairs]

#     if False:
#         AT.test_token_behavior(model, tokenizer, new_token_pairs)


#     # fuzz new tokens
#     with torch.no_grad():
#         for i, t in enumerate(new_token_ids):

#             # random fuzz
#             # model.model.embed_tokens.weight[t, :] += torch.randn_like(model.model.embed_tokens.weight[t]) * 1e-2

#             # # fuzz by rolling; preserves statistics, but eliminates semantixs
#             # model.model.embed_tokens.weight[t, :] += torch.roll(model.model.embed_tokens.weight[t], shifts=i)

#             # ignore inits, make random
#             model.model.embed_tokens.weight[t, :] += torch.randn_like(model.model.embed_tokens.weight[t]) * 1e-3


#     # ensure parsability
#     if False:
#         print('Testing that new tokens are parsable')
#         toks = [x[0] for x in new_token_pairs]
#         AT.assert_parsability_1(tokenizer, toks)
#         AT.assert_parsability_2(tokenizer, toks)
#         AT.assert_parsability_3(tokenizer, toks)

#     if False:
#         print_model_info(model)

#     add_hooks(model)

#     already_loaded = True


# ##########
# # LOR stuff

# # Layer index to target, hard coded
# LOR_LAYER = 14  # must be positive num (can't index bwd)

# WHICH_LOR = 1

# ##########
# # Do QKVOGUD Lors

# if WHICH_LOR == 1:

#     # These are metatokens that get interjected throughout training data
#     #
#     # Q|| K|| V|| O|| G|| U|| D||
#     lor_block = "^@Q^@Ql^@Qr^@K^@Kl^@Kr^@V^@Vl^@Vr^@O^@Ol^@Or^@G^@Gl^@Gr^@U^@Ul^@Ur^@D^@Dl^@Dr"

#     # metatoken mask, for masking loss. metatokens should be predicted, thus make
#     # it to loss. meta weights should not be affected directly by the cross entropy
#     # loss.

#     # TODO: i think this needs to be shifted one left, into the preceding block (??)

#     # lor_mask = [1, 0, 0] * 7  # note: should be list, not tensor. `datasets` converts it back to a list (then tensor again) anyway
#     lor_mask = [0, 0, 0] * 7  # TODO: rm, this mask does NOT learn output of metatokens

#     # for blocks that don't contain any lors
#     #   note: right now this assumes only a single layer is getting targeted, but someday could have separate values per index
#     empty_lor_ixs = {
#         'lor_qs': [None] * num_layers,
#         'lor_ks': [None] * num_layers,
#         'lor_vs': [None] * num_layers,
#         'lor_os': [None] * num_layers,
#         'lor_gs': [None] * num_layers,
#         'lor_us': [None] * num_layers,
#         'lor_ds': [None] * num_layers,
#     }

#     # where to parse to collect metaweights, tied to `lor_block`'s implementation.
#     #
#     # note: this might look shifted left, but consider "Q||". The token emitted
#     #       after Q will be in location 0 and represent the first predicted |.
#     parse_lor_ixs = {
#         'lor_qs': [None] * num_layers,
#         'lor_ks': [None] * num_layers,
#         'lor_vs': [None] * num_layers,
#         'lor_os': [None] * num_layers,
#         'lor_gs': [None] * num_layers,
#         'lor_us': [None] * num_layers,
#         'lor_ds': [None] * num_layers,
#     }

#     parse_lor_ixs['lor_qs'][LOR_LAYER] = (0, 1)  # (left singular value, right singular value)
#     parse_lor_ixs['lor_ks'][LOR_LAYER] = (3, 4)
#     parse_lor_ixs['lor_vs'][LOR_LAYER] = (6, 7)
#     parse_lor_ixs['lor_os'][LOR_LAYER] = (9, 10)
#     parse_lor_ixs['lor_gs'][LOR_LAYER] = (12, 13)
#     parse_lor_ixs['lor_us'][LOR_LAYER] = (15, 16)
#     parse_lor_ixs['lor_ds'][LOR_LAYER] = (18, 19)

#     lor_ix_keys = set(parse_lor_ixs.keys())



# ##########
# # Just do GUD Lors

# elif WHICH_LOR == 2:

#     # These are metatokens that get interjected throughout training data
#     #
#     # G|| U|| D||
#     lor_block = "^@G^@Gl^@Gr^@U^@Ul^@Ur^@D^@Dl^@Dr"

#     # metatoken mask, for masking loss. metatokens should be predicted, thus make
#     # it to loss. meta weights should not be affected directly by the cross entropy
#     # loss.

#     # TODO: i think this needs to be shifted one left, into the preceding block (??)

#     # lor_mask = [1, 0, 0] * 3  # note: should be list, not tensor. `datasets` converts it back to a list (then tensor again) anyway
#     lor_mask = [0, 0, 0] * 3  # TODO: rm, this mask does NOT learn output of metatokens

#     # for blocks that don't contain any lors
#     #   note: right now this assumes only a single layer is getting targeted, but someday could have separate values per index
#     empty_lor_ixs = {
#         'lor_gs': [None] * num_layers,
#         'lor_us': [None] * num_layers,
#         'lor_ds': [None] * num_layers,
#     }

#     # where to parse to collect metaweights, tied to `lor_block`'s implementation.
#     #
#     # note: this might look shifted left, but consider "Q||". The token emitted
#     #       after Q will be in location 0 and represent the first predicted |.

#     parse_lor_ixs = {
#         'lor_gs': [None] * num_layers,
#         'lor_us': [None] * num_layers,
#         'lor_ds': [None] * num_layers,
#     }
#     parse_lor_ixs['lor_gs'][LOR_LAYER] = (0, 1)  # (left singular value, right singular value)
#     parse_lor_ixs['lor_us'][LOR_LAYER] = (3, 4)
#     parse_lor_ixs['lor_ds'][LOR_LAYER] = (6, 7)

#     lor_ix_keys = set(parse_lor_ixs.keys())


# ##################################################
# # Data

# BATCH_SIZE = 32
# split = '_4'  # 4 variable version of dataset

# warnings.warn('training data is severely truncated during r&d')

# train_small = load_dataset("neurallambda/arithmetic_dataset", split=f"train{split}").select(range(BATCH_SIZE * 80))  # todo: rm, this truncates training
# test_small = load_dataset("neurallambda/arithmetic_dataset", split=f"test{split}").select(range(BATCH_SIZE * 4))

# # train_small = load_dataset("neurallambda/arithmetic_dataset", split=f"train{split}").select(range(BATCH_SIZE * 5))  # todo: rm, this truncates training
# # test_small = load_dataset("neurallambda/arithmetic_dataset", split=f"test{split}").select(range(BATCH_SIZE * 1))


# def process_row(row, lor_block, lor_mask):
#     prepared_data = []
#     for item in row['input']:

#         # # warnings.warn('including whole sample in loss mask')
#         # prepared_data.append({"type": "text", "content": item, 'include_in_loss': True, **empty_lor_ixs})  # TODO: including text in loss

#         prepared_data.append({"type": "text", "content": item, 'include_in_loss': False, **empty_lor_ixs})  # TODO: including text in loss

#         prepared_data.append({"type": "lor", "content": lor_block, 'include_in_loss': True, 'loss_mask': lor_mask, **parse_lor_ixs})

#     # Add the output
#     prepared_data.append({"type": "text", "content": row['output'], 'include_in_loss': True, **empty_lor_ixs})

#     return {"prepared_data": prepared_data}


# def insert_lor_blocks(dataset: HFDataset, lor_block, lor_mask) -> HFDataset:
#     """
#     Insert LOR blocks between text elements in the dataset.
#     """
#     # NOTE: the `dataset.map` process injects missing keys found among any rows, ie loss_mask
#     out = dataset.map(
#         lambda row: process_row(row, lor_block, lor_mask),
#         remove_columns=["input", "output"])
#     return out


# # Apply the preparation to a specific split
# train_small = insert_lor_blocks(train_small, lor_block, lor_mask)

# '''

# # Sample
# for example in train_small.select(range(2)):  # Show first 5 examples
#     print(example)
#     print()

# '''

# USE_UNCAUSAL_METATOKENS = True

# train_dl = C.create_dataloader(
#     dataset=train_small,
#     num_layers=num_layers,
#     lor_ix_keys=lor_ix_keys,
#     batch_size=BATCH_SIZE,
#     tokenizer=tokenizer,
#     use_uncausal_metatokens=USE_UNCAUSAL_METATOKENS,
#     device='cpu',
#     shuffle=False,
#     num_workers=0
# )


# # Apply the preparation to a specific split
# test_small = insert_lor_blocks(test_small, lor_block, lor_mask)
# # # Sample
# # for example in test_small.select(range(5)):  # Show first 5 examples
# #     print(example)
# #     print()
# test_dl = C.create_dataloader(
#     dataset=test_small,
#     num_layers=num_layers,
#     lor_ix_keys=lor_ix_keys,
#     batch_size=BATCH_SIZE,
#     tokenizer=tokenizer,
#     use_uncausal_metatokens=USE_UNCAUSAL_METATOKENS,
#     device='cpu',
#     shuffle=False,
#     num_workers=0
# )


# ##################################################
# # LoR Models

# @dataclass
# class LoR:
#     ''' this isn't used yet, cache/parse ixs/lor-modules are still a dict of this shape '''
#     lor_qs: List[Any]  # For Q blocks, a per-layer list of things (parse indexes, LoRModule, or lor cache)
#     lor_ks: List[Any]
#     lor_vs: List[Any]
#     lor_os: List[Any]
#     lor_gs: List[Any]
#     lor_us: List[Any]
#     lor_ds: List[Any]


# def assert_lor_shape(x):
#     pass


# def assert_no_biases(model):
#     '''Because of how batching interacts with parsing LoR weights, the LORModule
# must not have biases. See LORModule for more details.'''
#     bias_info = []

#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             bias_info.append(f"Linear bias found in {name}")

#         elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.bias is not None:
#             bias_info.append(f"Convolutional bias found in {name}")

#         elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.bias is not None:
#             bias_info.append(f"BatchNorm bias found in {name}")

#         elif isinstance(module, nn.LayerNorm) and module.bias is not None:
#             bias_info.append(f"LayerNorm bias found in {name}")

#         elif isinstance(module, (nn.LSTM, nn.GRU)):
#             for param_name, param in module.named_parameters():
#                 if 'bias' in param_name:
#                     bias_info.append(f"{type(module).__name__} bias found in {name}.{param_name}")

#         elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
#             bias_info.append(f"Embedding bias (padding_idx) found in {name}")

#     if bias_info:
#         error_message = "The model contains biases:\n" + "\n".join(f"- {info}" for info in bias_info)
#         raise AssertionError(error_message)



# def apply_lor(x, lorl, lorr) -> torch.Tensor:
#     ''' Low rank "matrix" multiplication

#     args:
#       x: [B, S, D]
#       lorl: [B, rank, out_features]
#       lorr: [B, in_features, rank]

#     '''
#     x = torch.einsum('bsd, bdr -> bsr', x, lorr)
#     x = torch.einsum('bsr, bdr -> bsd', x, lorl)

#     return x

# # START_BLOCK_5


# class LORProject(nn.Module):
#     '''
#     LOR weights need to be projected to fit the shapes of the underlying
#     Linear layers they're matching. These LORModules solve this, and there
#     are 2 modules per target linear layer, (left singular values, right
#     singular values). They multiply like: out=LRx, to match the usual out=Wx,
#     so the new calculation becomes out=Wx + LRx.

#     R are the input vectors, and L are the output vectors. The first
#     dimension of the LORModules must match the embedding that we're
#     projecting, so the 1st values are all `dim`. The 2nd dim of R is the
#     input dimension of the matched linear weights. The 2nd dim of L is the
#     output dimension of the same linear layer.

#     '''
#     def __init__(self, dropout_rate=0.2):
#         super().__init__()

#         dim = model.model.embed_tokens.weight.shape[1]  # embedding dimension
#         k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
#         v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
#         ff_dim = model.config.intermediate_size

#         self.dim = dim
#         self.k_dim = k_dim
#         self.v_dim = v_dim
#         self.ff_dim = ff_dim

#         self.input_dims = [dim] * 14  # All inputs are 'dim'-dimensional
#         self.output_dims = [
#             dim, dim,  # lor_qs_l, lor_qs_r
#             k_dim, dim,  # lor_ks_l, lor_ks_r
#             v_dim, dim,  # lor_vs_l, lor_vs_r
#             dim, dim,  # lor_os_l, lor_os_r
#             ff_dim, dim,  # lor_gs_l, lor_gs_r
#             ff_dim, dim,  # lor_us_l, lor_us_r
#             dim, ff_dim  # lor_ds_l, lor_ds_r
#         ]

#         self.token_mixing_dim = 128
#         self.channel_mixing_dim = 128

#         # Token-mixing MLP
#         self.token_mixing_mlp = nn.Sequential(
#             nn.RMSNorm(14),
#             nn.Linear(14, self.token_mixing_dim, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(self.token_mixing_dim, 14, bias=False),
#             nn.Dropout(dropout_rate)
#         )

#         # Channel-mixing MLP (same for all inputs)
#         self.channel_mixing_mlp = nn.Sequential(
#             nn.RMSNorm(dim),
#             nn.Linear(dim, self.channel_mixing_dim, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(self.channel_mixing_dim, dim, bias=False),
#             nn.Dropout(dropout_rate)
#         )

#         # Final projection layers
#         self.final_projections = nn.ModuleDict({
#             'lor_qs_l': nn.Linear(dim, dim, bias=False),
#             'lor_qs_r': nn.Linear(dim, dim, bias=False),
#             'lor_ks_l': nn.Linear(dim, k_dim, bias=False),
#             'lor_ks_r': nn.Linear(dim, dim, bias=False),
#             'lor_vs_l': nn.Linear(dim, v_dim, bias=False),
#             'lor_vs_r': nn.Linear(dim, dim, bias=False),
#             'lor_os_l': nn.Linear(dim, dim, bias=False),
#             'lor_os_r': nn.Linear(dim, dim, bias=False),
#             'lor_gs_l': nn.Linear(dim, ff_dim, bias=False),
#             'lor_gs_r': nn.Linear(dim, dim, bias=False),
#             # 'lor_us_l': nn.Linear(dim, ff_dim, bias=False),
#             'lor_us_r': nn.Linear(dim, dim, bias=False),
#             'lor_ds_l': nn.Linear(dim, dim, bias=False),
#             # 'lor_ds_r': nn.Linear(dim, ff_dim, bias=False),

#             # 'lor_us_l_ds_r': nn.Linear(dim, ff_dim, bias=False),

#         })


#         # LORModule must play nicely in a batch situation, where some samples
#         # of the batch imply lor parses and others don't. Non LoR'd samples
#         # should not be affected by sharing a batch with LoR'd samples. Biases
#         # corrupt this property. 0-valued lors (from samples without lor
#         # parses) must produce 0-valued outputs here. Checking for biases is
#         # not the whole solution, you must take care.
#         #
#         # This is not necessarily necessary. For instance, clever masking of
#         # non-parsed samples might obviate this.
#         assert_no_biases(self)


#     def forward(self, **kwargs):
#         # Stack all inputs
#         inputs = [
#             kwargs['lor_qs_l'], kwargs['lor_qs_r'],
#             kwargs['lor_ks_l'], kwargs['lor_ks_r'],
#             kwargs['lor_vs_l'], kwargs['lor_vs_r'],
#             kwargs['lor_os_l'], kwargs['lor_os_r'],
#             kwargs['lor_gs_l'], kwargs['lor_gs_r'],
#             kwargs['lor_us_l'], kwargs['lor_us_r'],
#             kwargs['lor_ds_l'], kwargs['lor_ds_r']
#         ]
#         # x shape: [batch, 14, dim]
#         x = torch.stack(inputs, dim=1)
#         B = x.shape[0]
#         device = x.device

#         # Token-mixing
#         residual = x
#         # x_token shape: [batch, dim, 14]
#         x_token = x.transpose(1, 2)
#         # Normalize across token dimension
#         x_token = self.token_mixing_mlp(x_token)
#         x_token = x_token.transpose(1, 2)  # [batch, 14, dim]
#         x = residual + x_token

#         # Channel-mixing
#         residual = x
#         x = self.channel_mixing_mlp(x)
#         x = residual + x

#         ##########
#         # Original attempt

#         # # Final projections to adjust dimensions
#         # outputs = []
#         # for i, proj in enumerate(self.final_projections.values()):
#         #     outputs.append(proj(x[:, i, :]))


#         ##########
#         # Tie intermediates. Adopting results from (ie, results from t14_homoiconic_llm_adding_data_to_mlp)

#         # ud_intermediate = self.final_projections['lor_us_l_ds_r']


#         # NOTE: statistics of randn are likely way off
#         # TODO: bc of this shrink, shrink token_mixing_mlp from dim=14 to dim=12, since those outputs go unused
#         ud_intermediate = torch.randn(B, self.ff_dim, device=device)

#         outputs = (
#             self.final_projections['lor_qs_l'](x[:, 0, :]),
#             self.final_projections['lor_qs_r'](x[:, 1, :]),
#             self.final_projections['lor_ks_l'](x[:, 2, :]),
#             self.final_projections['lor_ks_r'](x[:, 3, :]),
#             self.final_projections['lor_vs_l'](x[:, 4, :]),
#             self.final_projections['lor_vs_r'](x[:, 5, :]),
#             self.final_projections['lor_os_l'](x[:, 6, :]),
#             self.final_projections['lor_os_r'](x[:, 7, :]),
#             self.final_projections['lor_gs_l'](x[:, 8, :]),
#             # ud_intermediate * -1,
#             self.final_projections['lor_gs_r'](x[:, 9, :]),
#             # self.final_projections['lor_us_l'](x[:, 10, :]),
#             ud_intermediate,
#             self.final_projections['lor_us_r'](x[:, 11, :]),
#             self.final_projections['lor_ds_l'](x[:, 12, :]),
#             # self.final_projections['lor_ds_r'](x[:, 13, :]),
#             ud_intermediate,
#         )

#         return outputs













# class LORNorm(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()

#         self.norm = nn.RMSNorm(out_dim)

#         # NOTE: there is probably a principled way of setting this, or
#         #   pre-learning this. As it stands, the norm can be initialized even
#         #   to 0 which effectively removes this entire layer from the
#         #   transformer stack EXCEPT residuals still pass forward. Strangely
#         #   enough, the network can do ok with a layer removed. I'm thinking
#         #   here that we can limit the initial impact of LoR stuff by setting
#         #   this low. This has a SIDE-EFFECT though of small grads to this layer.
#         with torch.no_grad():
#             self.norm.weight[:] = self.norm.weight * 1e-2

#         # This is akin to He initialization. TODO: worth it?
#         self.scale = (2 / in_dim) ** 0.5

#         # LORModule must play nicely in a batch situation, where some samples
#         # of the batch imply lor parses and others don't. Non LoR'd samples
#         # should not be affected by sharing a batch with LoR'd samples. Biases
#         # corrupt this property. 0-valued lors (from samples without lor
#         # parses) must produce 0-valued outputs here. Checking for biases is
#         # not the whole solution, you must take care.
#         #
#         # This is not necessarily necessary. For instance, clever masking of
#         # non-parsed samples might obviate this.
#         assert_no_biases(self)

#     def forward(self, lor_cache, original, hidden_state):
#         '''This gets applied separately per QKVOGUD, and really just allows
#         Normalization to be handled from this module, instead of say adding a
#         new norm layer throughout the underlying model.

#         The `project` function is where more magic happens; it takes in ALL QKVOGUD parses for a layer, and generates a cache for each together.

#         Args:
#           original: model's original values of eg QKVOGUD within this layer
#           hidden_state: hidden_state at this layer, that projects through this layer's associated linear QKVOGUD block, and will be used to project through the LoR version too.

#         '''

#         if lor_cache is not None:
#             lorl, lorr = lor_cache
#             l = apply_lor(hidden_state, lorl, lorr)
#             # return self.norm(original + l * self.scale)  # TODO: revisit if `scale` is good
#             return self.norm(original + l)
#         else:
#             return self.norm(original)








# # @@@@@@@@@@
# # Test for no batch effect

# if False:
#     B = 1
#     S = 5
#     emb_dim = 128
#     in_dim = 128
#     out_dim = 256
#     rank = 3

#     lm = LORModule(emb_dim, out_dim, emb_dim, in_dim)

#     lorl_ = torch.randn(B, out_dim, rank)
#     lorr_ = torch.randn(B, in_dim, rank)
#     lor_cache = (lorl_, lorr_)
#     zlor_cache = (0 * lorl_, 0 * lorr_)

#     hidden_state = torch.randn(B, S, emb_dim)
#     orig = torch.randn(B, S, out_dim)

#     no_cache = lm(None, orig, hidden_state)
#     zero_cache = lm(zlor_cache, orig, hidden_state)
#     has_cache = lm(lor_cache, orig, hidden_state)

#     assert torch.allclose(no_cache, zero_cache), 'None cache should have same effect as zero cache. max diff: {(no_cache - zero_cache).max().item()}'
#     assert (no_cache - has_cache).abs().sum() > 1.0, 'lor cache should have an effect'

# # @@@@@@@@@@


# # END_BLOCK_5


# def select_ixs(x, indices):
#     """Selects values from a 3D tensor (`[batch, seq, dim]`) along dim=1 using
#     provided indices.

#     Perform indexed selection on a 3D tensor. If `indices` contains -1, that
#     location will be filled with 0s.

#     Args:
#         x (torch.Tensor): Input tensor of shape [batch, xs, ys] where:

#         indices (torch.Tensor): 1D tensor of indices for selection. Its length
#             should match the batch size of x, and it's max value must be
#             smaller than the length of xs. Negative values in indices indicate
#             that the location should be filled with default 0 values, ie no
#             parsing performed.

#     Returns:
#         torch.Tensor: [batch, dim]

#     Example:
#         >>> x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#         ...                   [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
#         >>> indices = torch.tensor([1, -1])
#         >>> select_ixs(x, indices, dim=1)
#         tensor([[[ 4,  5,  6]],
#                 [[ 0,  0,  0]]])

#     """

#     B = x.shape[0]  # batch

#     # a mask for valid indices (non-negative)
#     mask = indices >= 0

#     # replace negative indices (indicating default 0 values) with 0
#     safe_indices = torch.where(mask, indices, torch.zeros_like(indices))

#     # gather the embeddings along the sequence dimension
#     gathered_values = x[torch.arange(B), safe_indices]

#     # a zero tensor, for default values where index == -1
#     zeros = torch.zeros_like(gathered_values)

#     # combine selected indices with defaults
#     return torch.where(mask.unsqueeze(1), gathered_values, zeros)


# def update_lors(
#         lor_models,  # shaped like `empty_lors`
#         lor_cache,  # shaped like `empty_lors`
#         lor_ixs: Dict[str, List[Tuple[int, int]]],  # column ixs, eg: {'lor_qs': [None, None, (left_ix, right_ix), None, ...n_layers]}
#         hidden_states,  # from out.hidden_states, so, for all layers
# ):
#     ''' Update the LORs by interpreting hidden states as new lor blocks '''

#     for k in lor_ixs.keys():
#         assert (
#             len(lor_models[k]) ==  # one (optional) lor module per layer
#             len(lor_cache[k]) ==  # cache per layer
#             len(lor_ixs[k][0]) ==  # left parse ixs
#             len(lor_ixs[k][1])  # right parse ixs
#         ), (f'''
# {len(lor_models[k])=} ==  # one (optional) lor module per layer
# {len(lor_cache[k])=} ==  # cache per layer
# {len(lor_ixs[k][0])=} ==  # left parse ixs
# {len(lor_ixs[k][1])=}  # right parse ixs
# ''')

#     num_layers = len(lor_models[list(lor_ix_keys)[0]])  # TODO: hacky way to get this

#     h_emb = hidden_states[-1]  # final layer states

#     for layer_ix in range(num_layers):
#         # skip non-lor'd layers
#         if lor_models['lor_proj'][layer_ix] is None:
#             continue

#         parses = {}
#         is_first_pass = False  # TODO: this logic is janky
#         for lor_key, (l_layer_batch_ixs, r_layer_batch_ixs) in lor_ixs.items():
#             #####
#             # Assertions and Prep

#             # skip layers that aren't targeted by LoR modules
#             if lor_models[lor_key][layer_ix] is None:
#                 continue
#             # lor_model = lor_models[lor_key][layer_ix]

#             # no parse ixs identified
#             l_empty = (l_layer_batch_ixs[layer_ix] >= 0)
#             r_empty = (r_layer_batch_ixs[layer_ix] >= 0)
#             assert (l_empty == r_empty).all().item(), 'left parse indexes and right parse indexes should always accompany each other'
#             if l_empty.sum() == 0:
#                 continue  # no parses in any rows in this column

#             #####
#             # Parse ixs, apply LoR projections to parsed embeddings, cache

#             # left singular values
#             pl = select_ixs(h_emb, l_layer_batch_ixs[layer_ix])
#             parses[f'{lor_key}_l'] = pl
#             # left_lors = lor_model.left_project(pl)
#             # if is_first_pass:
#             #     new_l = left_lors.unsqueeze(2)  # [B, DIM, RANK]
#             # else:
#             #     new_l = torch.cat([lor_cache[lor_key][layer_ix][0], left_lors.unsqueeze(2)], dim=2)

#             # right singular values
#             pr = select_ixs(h_emb, r_layer_batch_ixs[layer_ix])
#             parses[f'{lor_key}_r'] = pr
#             # right_lors = lor_model.right_project(pr)
#             # if is_first_pass:
#             #     new_r = right_lors.unsqueeze(2)  # [B, DIM, RANK]
#             # else:
#             #     new_r = torch.cat([lor_cache[lor_key][layer_ix][1], right_lors.unsqueeze(2)], dim=2)

#             is_first_pass = is_first_pass or lor_cache[lor_key][layer_ix] is None


#         # no parses implied anywhere
#         if len(parses) == 0:
#             continue

#         # TODO: assert all keys present

#         if WHICH_LOR == 1:
#             (lor_ql,
#              lor_qr,
#              lor_kl,
#              lor_kr,
#              lor_vl,
#              lor_vr,
#              lor_ol,
#              lor_or,
#              lor_gl,
#              lor_gr,
#              lor_ul,
#              lor_ur,
#              lor_dl,
#              lor_dr
#              ) = lor_models['lor_proj'][layer_ix](**parses)

#             if is_first_pass:
#                 lor_cache['lor_qs'][layer_ix] = (lor_ql.unsqueeze(2), lor_qr.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_ks'][layer_ix] = (lor_kl.unsqueeze(2), lor_kr.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_vs'][layer_ix] = (lor_vl.unsqueeze(2), lor_vr.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_os'][layer_ix] = (lor_ol.unsqueeze(2), lor_or.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_gs'][layer_ix] = (lor_gl.unsqueeze(2), lor_gr.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_us'][layer_ix] = (lor_ul.unsqueeze(2), lor_ur.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_ds'][layer_ix] = (lor_dl.unsqueeze(2), lor_dr.unsqueeze(2))  # [B, DIM, RANK]
#             else:
#                 lor_cache['lor_qs'][layer_ix] = (torch.cat([lor_cache['lor_qs'][layer_ix][0], lor_ql.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_qs'][layer_ix][1], lor_qr.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_ks'][layer_ix] = (torch.cat([lor_cache['lor_ks'][layer_ix][0], lor_kl.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_ks'][layer_ix][1], lor_kr.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_vs'][layer_ix] = (torch.cat([lor_cache['lor_vs'][layer_ix][0], lor_vl.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_vs'][layer_ix][1], lor_vr.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_os'][layer_ix] = (torch.cat([lor_cache['lor_os'][layer_ix][0], lor_ol.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_os'][layer_ix][1], lor_or.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_gs'][layer_ix] = (torch.cat([lor_cache['lor_gs'][layer_ix][0], lor_gl.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_gs'][layer_ix][1], lor_gr.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_us'][layer_ix] = (torch.cat([lor_cache['lor_us'][layer_ix][0], lor_ul.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_us'][layer_ix][1], lor_ur.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_ds'][layer_ix] = (torch.cat([lor_cache['lor_ds'][layer_ix][0], lor_dl.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_ds'][layer_ix][1], lor_dr.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#         elif WHICH_LOR == 2:
#             (lor_gl,
#              lor_gr,
#              lor_ul,
#              lor_ur,
#              lor_dl,
#              lor_dr
#              ) = lor_models['lor_proj'][layer_ix](**parses)

#             if is_first_pass:
#                 lor_cache['lor_gs'][layer_ix] = (lor_gl.unsqueeze(2), lor_gr.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_us'][layer_ix] = (lor_ul.unsqueeze(2), lor_ur.unsqueeze(2))  # [B, DIM, RANK]
#                 lor_cache['lor_ds'][layer_ix] = (lor_dl.unsqueeze(2), lor_dr.unsqueeze(2))  # [B, DIM, RANK]
#             else:
#                 lor_cache['lor_gs'][layer_ix] = (torch.cat([lor_cache['lor_gs'][layer_ix][0], lor_gl.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_gs'][layer_ix][1], lor_gr.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_us'][layer_ix] = (torch.cat([lor_cache['lor_us'][layer_ix][0], lor_ul.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_us'][layer_ix][1], lor_ur.unsqueeze(2)], dim=2))  # [B, DIM, RANK]
#                 lor_cache['lor_ds'][layer_ix] = (torch.cat([lor_cache['lor_ds'][layer_ix][0], lor_dl.unsqueeze(2)], dim=2), torch.cat([lor_cache['lor_ds'][layer_ix][1], lor_dr.unsqueeze(2)], dim=2))  # [B, DIM, RANK]


#     return lor_cache


# def add_noise_keep_stats(x, noise_scale=0.1):
#     ''' Add randn noise, preserve mean and norm. Used for "active forgetting" experiment. '''
#     mean = x.mean()

#     noisy = x + torch.randn_like(x) * noise_scale
#     centered = noisy - noisy.mean()

#     factor = (x - mean).norm() / centered.norm()
#     return centered * factor + mean


# ##########
# #

# if True:
#     num_epochs = 50

#     # LOR Models: same structure as LORs
#     dim = model.model.embed_tokens.weight.shape[1]
#     k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
#     v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
#     ff_dim = model.config.intermediate_size

#     # Note: there must be at least a None per each QKVOGUD block per layer
#     lor_models = nn.ModuleDict(
#         {

#             #####
#             # Projection
#             'lor_proj': nn.ModuleList([None] * num_layers),

#             #####
#             # Norms

#             # low rank attention params
#             "lor_qs": nn.ModuleList([None] * num_layers),
#             "lor_ks": nn.ModuleList([None] * num_layers),
#             "lor_vs": nn.ModuleList([None] * num_layers),
#             "lor_os": nn.ModuleList([None] * num_layers),

#             # low rank mlp params
#             "lor_gs": nn.ModuleList([None] * num_layers),
#             "lor_us": nn.ModuleList([None] * num_layers),
#             "lor_ds": nn.ModuleList([None] * num_layers),
#         }
#     )

#     lor_models['lor_proj'][LOR_LAYER] = LORProject()

#     if WHICH_LOR == 1:
#         lor_models['lor_qs'][LOR_LAYER] = LORNorm(dim, dim)
#         lor_models['lor_ks'][LOR_LAYER] = LORNorm(dim, k_dim)
#         lor_models['lor_vs'][LOR_LAYER] = LORNorm(dim, v_dim)
#         lor_models['lor_os'][LOR_LAYER] = LORNorm(dim, dim)

#         lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim)
#         lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim)
#         lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim)
#     elif WHICH_LOR == 2:
#         lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim)
#         lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim)
#         lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim)
#     lor_models = lor_models.to(DEVICE, dtype=model.dtype)


#     print_model_info(lor_models)
#     add_hooks(lor_models)


#     ##########
#     # Load state

#     LOAD_MODEL = False

#     if LOAD_MODEL:

#         # lor_path = "t14_homoiconic_llm_06/model_run1_lor_models_epoch19_valloss0.4399.pth"
#         # emb_path = "t14_homoiconic_llm_06/model_run1_embed_tokens_epoch19_valloss0.4399.pth"

#         lor_path = "t14_homoiconic_llm_07/model_run0_lor_models_epoch4_valloss0.3802.pth"
#         emb_path = "t14_homoiconic_llm_07/model_run0_embed_tokens_epoch4_valloss0.3802.pth"

#         global_epoch = load_model(lor_models, lor_path)
#         _ = load_model(model.model.embed_tokens, emb_path)
#     else:
#         global_epoch = 0


#     # [print(model.model.embed_tokens.weight[i, :5]) for i in tokenizer(lor_block)['input_ids']]

#     fast_params = [
#         'lor_proj.14.token_mixing_mlp.0.weight',
#         'lor_proj.14.token_mixing_mlp.1.weight',
#         'lor_proj.14.token_mixing_mlp.4.weight',
#         'lor_proj.14.channel_mixing_mlp.0.weight',
#         'lor_proj.14.channel_mixing_mlp.1.weight',
#         'lor_proj.14.channel_mixing_mlp.4.weight',
#         'lor_proj.14.final_projections.lor_qs_l.weight',
#         'lor_proj.14.final_projections.lor_qs_r.weight',
#         'lor_proj.14.final_projections.lor_ks_l.weight',
#         'lor_proj.14.final_projections.lor_ks_r.weight',
#         'lor_proj.14.final_projections.lor_vs_l.weight',
#         'lor_proj.14.final_projections.lor_vs_r.weight',
#         'lor_proj.14.final_projections.lor_os_l.weight',
#         'lor_proj.14.final_projections.lor_os_r.weight',
#         'lor_proj.14.final_projections.lor_gs_l.weight',
#         'lor_proj.14.final_projections.lor_gs_r.weight',
#         'lor_proj.14.final_projections.lor_us_l.weight',
#         'lor_proj.14.final_projections.lor_us_r.weight',
#         'lor_proj.14.final_projections.lor_ds_l.weight',
#         'lor_proj.14.final_projections.lor_ds_r.weight',
#     ]

#     models = {
#         'main': model,
#         'lor': lor_models,
#     }

#     param_groups = [
#         {
#             'name': 'main_model',
#             'model': 'main',
#             'lr': 1e-6,
#             'weight_decay': 0,
#         },
#         {
#             'name': 'lor_standard',
#             'model': 'lor',
#             'lr': 1e-3,
#             'weight_decay': 1e-2,
#             'exclude': fast_params,
#         },
#         {
#             'name': 'lor_high',
#             'model': 'lor',
#             'lr': 1e-3,
#             'weight_decay': 1e-2,
#             'include': fast_params,
#         },
#     ]

#     optimizer_param_groups = []

#     for group in param_groups:
#         params = []
#         for name, param in models[group['model']].named_parameters():
#             if 'include' in group and name in group['include']:
#                 params.append(param)
#             elif 'exclude' in group and name not in group['exclude']:
#                 params.append(param)
#             elif 'include' not in group and 'exclude' not in group:
#                 params.append(param)

#         if params:
#             optimizer_param_groups.append({
#                 'params': params,
#                 'lr': group['lr'],
#                 'weight_decay': group.get('weight_decay', 0)
#             })

#     optimizer = optim.AdamW(optimizer_param_groups)

#     '''

# new_lr = 1e-2
# for param_group in optimizer.param_groups: param_group['lr'] = new_lr

# new_lr = 1e-3
# for param_group in optimizer.param_groups: param_group['lr'] = new_lr

# new_lr = 1e-4
# for param_group in optimizer.param_groups: param_group['lr'] = new_lr

#     '''

#     train_losses = []
#     test_losses = []
#     best_loss = float('inf')

#     run_number = get_next_run_number(save_path)

#     # START_BLOCK_2

#     model.train()
#     for epoch in range(num_epochs):
#         global_epoch += 1
#         train_loss = run_epoch(model, lor_models, train_dl, optimizer, DEVICE, train=True)
#         train_losses.append(train_loss)
#         writer.add_scalars('loss', {'train': train_loss}, global_epoch)

#         if epoch % 1 == 0:
#             test_loss = run_epoch(model, lor_models, test_dl, optimizer, DEVICE, train=False)
#             # test_loss = 0
#             test_losses.append(test_loss)
#             writer.add_scalars('loss', {'test': test_loss}, global_epoch)
#             print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

#             if test_loss < best_loss:
#                 best_loss = test_loss
#                 try:
#                     save_model('lor_models', run_number, lor_models, save_path, test_loss, global_epoch)
#                     save_model('embed_tokens', run_number, model.model.embed_tokens, save_path, test_loss, global_epoch)
#                 except Exception as e:
#                     print(f'EXCEPTION SAVING: {e}')
#                     breakpoint()
#                     save_model('lor_models', run_number, lor_models, save_path, test_loss, global_epoch)
#                     save_model('embed_tokens', run_number, model.model.embed_tokens, save_path, test_loss, global_epoch)

#         else:
#             print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")


#         # # ACTIVE FORGETTING EXPERIMENT (TODO, worth it?)
#         # if epoch % 1 == 0:
#         #     print('active forgetting of new embeddings')
#         #     with torch.no_grad():
#         #         for i, t in enumerate(new_token_ids):
#         #             model.model.embed_tokens.weight[t, :] = add_noise_keep_stats(model.model.embed_tokens.weight[t], noise_scale=1e-2)


#     # END_BLOCK_2

# if num_epochs > 0:
#     epochs = list(range(len(train_losses)))
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, train_losses, label='Training Loss', color='blue')
#     plt.plot(epochs, test_losses, label='Test Loss', color='red')

#     # Customize the plot
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Show the plot
#     plt.show()


# ####################################################################################################
# ####################################################################################################
# ####################################################################################################


# # Inference, ie Autoregressive Continuation

# # START_BLOCK_1

# def generate(model, lor_models, col_inputs, max_new_tokens):
#     '''Recurrently process column blocks of ids, concatenating attention_mask and
#     past_key_values across generations. Can generate new tokens if max_new_tokens > 0.'''
#     all_output_ids = []
#     tokens_generated = 0

#     lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch

#     #####
#     # Move to cuda

#     device = model.device

#     # list of columns
#     input_idss = [x['input_ids'].to(device) for x in col_inputs]
#     attention_masks = [x['attention_mask'].to(device) for x in col_inputs]
#     position_idss = [x['position_ids'].to(device) for x in col_inputs]

#     # lor_ixs per column
#     lor_ixss = []
#     for col in col_inputs:
#         lor_ixs = {k: (v[0].to(device), v[1].to(device)) for k, v in col.items() if k in lor_ix_keys}
#         # v[0]: [layer, batch_size]; parse ixs, 1 (optional) index per layer for each sample
#         # v[1]: [layer, batch_size]
#         lor_ixss.append(lor_ixs)


#     #####
#     # Handle input columns

#     lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch
#     out_logits, cat_attention_mask, past_key_values, lor_cache = forward_columns(model, lor_models, lor_cache, input_idss, attention_masks, position_idss, lor_ixss)

#     max_position_ids = torch.cat([x['position_ids'] for x in col_inputs], dim=1).max(dim=1).values
#     last_token = get_last_attended_token(torch.cat(out_logits, dim=1).argmax(dim=-1), cat_attention_mask)


#     #####
#     # Continue successive tokens

#     attention_mask = cat_attention_mask

#     # Sample final prediction as first output token
#     input_ids = last_token.unsqueeze(1)
#     all_output_ids.append(input_ids)
#     tokens_generated += 1

#     # If max_new_tokens > 0, continue generating new tokens
#     position_ids = max_position_ids.unsqueeze(1)
#     while tokens_generated < max_new_tokens:

#         # Update position_ids
#         position_ids = position_ids + 1

#         # Update attention_mask
#         attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=-1)

#         lorm = partially_apply_models(lor_models, lor_cache)

#         # Generate the next token
#         out = model(input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     past_key_values=past_key_values,
#                     return_dict=True,
#                     use_cache=True,
#                     position_ids=position_ids,
#                     output_hidden_states=True,
#                     **lorm,
#                     )

#         # TODO: lor updates aren't implemented during autoregression
#         past_key_values = out.past_key_values

#         # sample the next token
#         input_ids = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
#         all_output_ids.append(input_ids)

#         tokens_generated += 1

#     return torch.cat(all_output_ids, dim=-1), attention_mask, past_key_values, lor_cache, position_ids, attention_mask


# def t(x):
#     return {'type': 'text', 'content': x, 'include_in_loss': False, **empty_lor_ixs}

# def w(x, lor_ixs):
#     # one parse per type can be used per column/block, eg LOR Q left
#     # singular value, LOR Q right singular value, LOR K etc...
#     return {'type': 'lor', 'content': x, 'include_in_loss': False, **lor_ixs}

# def p(x):
#     return {'type': 'pad_block', 'content': x, 'include_in_loss': False, **empty_lor_ixs}


# model.eval()
# with torch.no_grad():

#     lor_ = w(lor_block, parse_lor_ixs)
#     le = empty_lor_ixs

#     training_prob = [
#         {'content': 'var_0=-2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_1=var_0', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_2=var_1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_3=var_2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_4=var_3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_5=4 - 3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_6=1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_7=var_2', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_8=-4', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_9=var_0 * 8', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'solve(var_7)=', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le},
#         # {'content': '-', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **le},
#         # {'content': '-2', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **le}
#     ]

#     training_prob_2 = [
#         {'content': 'var_0=9', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_1=var_0', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_2=1 - var_1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_3=-1', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_4=-3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_5=-7', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_6=7', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_7=-6 * var_4', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_8=var_1 + var_3', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'var_9=var_8', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le}, lor_,
#         {'content': 'solve(var_5)=', 'include_in_loss': False, 'loss_mask': None, 'type': 'text', **le},
#         # {'content': '-', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **le},
#         # {'content': '-7', 'include_in_loss': True, 'loss_mask': None, 'type': 'text', **le}
#     ]

#     test_prompts = [
#         training_prob,
#         training_prob_2,

#         [t("var_0=1"), lor_, t("var_1=2"), lor_, t("var_3=var_0 + var_1"), lor_, t("solve(var_3)=")],  # 3
#         [t("var_0=3"), lor_, t("var_1=5"), lor_, t("var_3=var_0 - var_1"), lor_, t("solve(var_3)=")],  # -2
#         [t("var_0=2"), lor_, t("var_1=3"), lor_, t("var_3=var_0 * var_1"), lor_, t("solve(var_3)=")],  # 6

#         [t("Once when")],
#         [t("Let me tell you about the time")],
#         [t("Let me tell you about the time I was golfing outside one sunny afternoon when suddenly")],

#     ]

#     inputs = C.create_column_batch_inputs(test_prompts, num_layers, lor_ix_keys, tokenizer, USE_UNCAUSAL_METATOKENS, device=DEVICE)

#     output_ids, col_attention_mask, col_past_key_values, elor_cache, position_ids, attention_mask = generate(model, lor_models, inputs, max_new_tokens=10)
#     # for display add in col outputs
#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
#     print('---------- gen results')
#     for o in outputs:
#         print(f'"""{o}"""')

# # END_BLOCK_1



# ##################################################
# # Assert that batch mode and single-row mode are equivalent
# #
# #   NOTE: batch vs single is incredibly sensitive to dtype. float32 is ok. bfloat16 has wild discrepancies.

# # START_BLOCK_4

# if False:
#     def remove_zero_columns(matrix):
#         column_sums = matrix.abs().sum(dim=0)
#         non_zero_mask = column_sums >= 1e-8
#         result = matrix[:, non_zero_mask]
#         return result

#     def get_single_sample(inputs, ix):
#         ''' out of a list of columns of batches '''
#         one_sample = []
#         for column in inputs:
#             new_column = {}
#             for k in column.keys():
#                 if isinstance(column[k], tuple):  # eg parse lor ixs
#                     tup = []
#                     for t in column[k]:
#                         tup.append(t[:, ix:ix + 1])  # shape=[layer, batch]
#                     new_column[k] = tuple(tup)
#                 else:
#                     new_column[k] = column[k][ix:ix + 1]  # preserve shape
#             one_sample.append(new_column)
#         return one_sample

#     inputs = C.create_column_batch_inputs(test_prompts, num_layers, lor_ix_keys, tokenizer, USE_UNCAUSAL_METATOKENS, device=DEVICE)
#     for inp in inputs:
#         inp['loss_mask'] = torch.ones_like(inp['loss_mask'])


#     # Run batch mode
#     batch_avg_loss, batch_out_logits, batch_cat_attention_mask, batch_lors = run_epoch(model, lor_models, [inputs], optimizer=None, device=DEVICE, train=False, debug=True)
#     batch_out_logits = torch.cat(batch_out_logits, dim=1)

#     # Collect results for each sample individually
#     all_single_avg_loss = []
#     all_single_out_logits = []
#     all_single_cat_attention_mask = []
#     all_single_lors = []
#     B = inputs[0]['input_ids'].shape[0]  # batch size
#     for ix in range(B):
#         one_sample = get_single_sample(inputs, ix)
#         single_avg_loss, single_out_logits, single_cat_attention_mask, single_lors = run_epoch(model, lor_models, [one_sample], optimizer=None, device=DEVICE, train=False, debug=True)
#         all_single_avg_loss.append(single_avg_loss)
#         all_single_out_logits.append(single_out_logits)
#         all_single_cat_attention_mask.append(single_cat_attention_mask)
#         all_single_lors.append(single_lors)

#     # Check Losses
#     bl = torch.tensor(batch_avg_loss)
#     sl = torch.tensor(all_single_avg_loss).mean()
#     assert torch.allclose(bl, sl, atol=1e-1), f'batch loss = {bl:>3f}, single loss = {sl:>3f}'

#     # Check attention masks
#     sa = torch.cat(all_single_cat_attention_mask, dim=0)
#     assert torch.allclose(batch_cat_attention_mask, sa)

#     # Check LoRs
#     lors_same = True
#     for batch_ix in range(B):
#         for k in lor_ix_keys:
#             for layer_ix in range(num_layers):
#                 blor = batch_lors[k][layer_ix]
#                 slor = all_single_lors[batch_ix][k][layer_ix]
#                 if blor is None:
#                     assert slor is None
#                 else:
#                     bllor = blor[0][batch_ix]  # left singular value cache
#                     brlor = blor[1][batch_ix]  # right singular value cache
#                     is_zero = bllor.abs().sum() < 1e-8
#                     if is_zero:
#                         # right values are zero too
#                         assert brlor.abs().sum() < 1e-8
#                         assert slor is None
#                     else:
#                         bllor_rm = remove_zero_columns(bllor)  # bllor might be bigger than sllor, bc of "non-parse" partners in batch
#                         sllor = slor[0][0]  # batch of 1, left singular value cache
#                         assert bllor_rm.shape == sllor.shape
#                         assert torch.allclose(bllor_rm, sllor, atol=1e-1), f'lors arent close, max diff: {(bllor_rm - sllor).abs().max()}, {batch_ix=}, {k=}, {layer_ix=}'
#                         brlor_rm = remove_zero_columns(brlor)  # bllor might be bigger than sllor, bc of "non-parse" partners in batch
#                         srlor = slor[1][0]  # batch of 1, right singular value cache
#                         assert brlor_rm.shape == srlor.shape
#                         assert torch.allclose(brlor_rm, srlor, atol=1e-1), f'lors arent close, max diff: {(brlor_rm - srlor).abs().max()}, {batch_ix=}, {k=}, {layer_ix=}'

#     # Check Logits
#     # reorient to shape like batch version

#     # len(all_single_out_logits)
#     # 8  # batch size
#     # >>> len(all_single_out_logits[0])
#     # 22  # column count
#     # >>> all_single_out_logits[0][0].shape
#     # torch.Size([1, 17, 151936])  # [1, sequence, dim]

#     # transposed = zip(*all_single_out_logits)
#     # so = torch.cat([  # cat along sequence dim
#     #     torch.cat(x, dim=0)  # cat along batch dim
#     #     for x in transposed
#     # ], dim=1)

#     so = []
#     for sample in all_single_out_logits:
#         so.append(torch.cat([column.squeeze(0) for column in sample], dim=0))
#     so = torch.stack(so, dim=0)
#     assert torch.allclose(batch_out_logits, so, atol=5e-3), f'{(batch_out_logits - so).abs().max().item()=:>.4f},  {(batch_out_logits - so).std()=:>.4f}'

#     print('assertions passed')


# # END_BLOCK_4
