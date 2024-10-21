import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import itertools
import t14_homoiconic_llm_model as Q

# from transformers.cache_utils import Cache
# import transformers.models.qwen2.modeling_qwen2 as Q

# from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

# from typing import Optional, Tuple, Union, List, Dict, Any
# import warnings
# import math

# from datasets import load_dataset, Dataset
# from torch.utils.data import DataLoader, random_split
# from torch.optim import AdamW

# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import os
# import random
# import time
# from neurallambda.lab.common import print_model_info

# SEED = 152
# torch.manual_seed(152)
# random.seed(SEED)

# DEVICE = 'cuda:1'
# BATCH_SIZE = 32

# model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")

def check_vocabulary(tokenizer, file_path):
    ''' print a vocab to a file '''
    voc = tokenizer.get_vocab()
    voc = sorted(list(voc.items()), key=lambda kv: kv[1])
    out = ''
    for k, v in voc:
        out += f'{v}: {k}\n'
    with open(file_path, 'w') as f:
        f.write(out)


def assert_unique_tokens(tokenizer, toks):
    ''' Check that the tokens we're adding are unique. '''
    vocab = tokenizer.get_vocab()
    for nt in toks:
        assert nt not in vocab, f'proposed new token already found in vocab: {nt}'
    print('new tokens are all unique')


def add_and_initialize_tokens(model: Q.Qwen2ForCausalLM, tokenizer: AutoTokenizer, new_token_pairs: list[tuple[str, str]]):
    '''Add new tokens to a tokenizer's vocabulary, and a model's embedding
    matrix. Initialize new embeddings to a paired extant token.

    args:
      new_token_pairs: [('new_token', 'token to initialize embedding to')]
    '''
    new_tokens = [pair[0] for pair in new_token_pairs]

    # Add new tokens to the tokenizer
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_tokens} new tokens to the tokenizer.")

    # Get the current sizes
    current_embed_size = model.model.embed_tokens.weight.shape[0]
    current_vocab_size = len(tokenizer)

    print(f"Current embed size: {current_embed_size}")
    print(f"Current vocab size: {current_vocab_size}")

    if current_vocab_size > current_embed_size:
        # you can do `model.resize_token_embeddings`, but it appears typical
        # for LLMs to have excess slots in the embeddings to make tensors a
        # good size for gpus perhaps
        raise ValueError("Tokenizer vocab size exceeds model embed size. This scenario is not handled.")

    # Initialize new token embeddings
    assert id(model.model.embed_tokens.weight) == id(model.lm_head.weight), "this function expects tied weights, but embed_tokens and lm_head are different"
    for new_token, init_word in new_token_pairs:
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        init_word_id = tokenizer.convert_tokens_to_ids(init_word)
        if init_word_id is None:
            raise ValueError("init token was actually multiple tokens/an invalid token")

        # copy the init token embeddings to the new token
        init_embedding = model.model.embed_tokens.weight[init_word_id]
        model.model.embed_tokens.weight.data[new_token_id] = init_embedding
        print(f"Initialized '{new_token}' (id: {new_token_id}) with embedding of '{init_word}' (id: {init_word_id})")



def test_token_behavior(model, tokenizer, new_token_pairs: list[tuple[str, str]], is_lor_version=True):

    ''' Build and run a prompt using new tokens, and the tokens they were initialized from. They should produce identical outputs.'''

    if is_lor_version:
        num_layers = model.config.num_hidden_layers

        lors = {
            # low rank attention params
            "lor_qs": [None] * num_layers,
            "lor_ks": [None] * num_layers,
            "lor_vs": [None] * num_layers,
            "lor_os": [None] * num_layers,

            # low rank mlp params
            "lor_us": [None] * num_layers,
            "lor_gs": [None] * num_layers,
            "lor_ds": [None] * num_layers,
        }
    else:
        lors = None

    # Smoke test
    print("Performing smoke test for new token embeddings weights...")
    for new_token, init_word in new_token_pairs:
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        init_word_id = tokenizer.convert_tokens_to_ids(init_word)

        # Check embedding similarities
        new_embedding = model.model.embed_tokens.weight.data[new_token_id]
        embed_similarities = F.cosine_similarity(new_embedding.unsqueeze(0), model.model.embed_tokens.weight, dim=1)
        print(f"\nSmoke test for '{new_token}':")

        # Check if new token matches itself in embeddings.
        #   Note: F.cosine_similarity is sometimes >1 (!?), maybe a bfloat16 issue? There's a pytorch issue tracking it
        assert (embed_similarities[new_token_id] >= 0.99).item(), f'new_token_id ({new_token}) is not self similar (weird), similarity={embed_similarities[new_token_id]}'
        assert (embed_similarities[init_word_id] >= 0.99).item(), f'new_token_id ({new_token}) is not similar to init_word_id ({init_word})'

        top_similarities, top_indices = embed_similarities.topk(6)
        print(f"  Top 5 similar tokens to '{new_token}':")
        for i, sim in zip(top_indices, top_similarities):
            token = tokenizer.convert_ids_to_tokens(i.item())
            print(f"    {token}: {sim.item():.4f}")

    # Integration test
    device = model.model.embed_tokens.weight.device
    new_ids  = torch.tensor([tokenizer.convert_tokens_to_ids(x[0]) for x in new_token_pairs], device=device).unsqueeze(0)
    orig_ids = torch.tensor([tokenizer.convert_tokens_to_ids(x[1]) for x in new_token_pairs], device=device).unsqueeze(0)
    attention_mask = torch.ones_like(new_ids).unsqueeze(0)
    with torch.no_grad():
        original_logits = model(input_ids=orig_ids, attention_mask=attention_mask, **lors).logits[0, -1]
        new_logits = model(input_ids=new_ids, attention_mask=attention_mask, **lors).logits[0, -1]
    assert torch.allclose(original_logits, new_logits), 'New tokens dont produce identical outputs as original tokens'
    print('New tokens pass the identical-output test')


##################################################
# Parsability tests: Just a sanity check that the new tokens will be parsed back out appropriately
#

def find_tokens_with_shared_chars(tokenizer, special_tokens):
    vocab = tokenizer.get_vocab()
    special_chars = set(''.join(special_tokens))
    shared_tokens = []

    for token, token_id in vocab.items():
        if any(char in token for char in special_chars):
            shared_tokens.append(token_id)

    return shared_tokens

def assert_parsability_1(tokenizer, special_tokens):
    ''' Check that new tokens play nicely with other tokens '''
    from tqdm import tqdm
    # Add special tokens to the tokenizer
    # num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    # print(f"Number of tokens added: {num_added_tokens}")

    # Get vocabulary and special token IDs
    vocab = tokenizer.get_vocab()
    vocab_ids = list(vocab.values())
    new_token_ids = [vocab[token] for token in special_tokens]

    # Find tokens with shared characters
    extra_tokens = [' ']
    shared_tokens = find_tokens_with_shared_chars(tokenizer, special_tokens + extra_tokens)
    print(f"Number of tokens with shared characters: {len(shared_tokens)}")

    def assert_correct_parsing(special_token_id, other_ids, insert_position):
        input_ids = list(other_ids)
        input_ids.insert(insert_position, special_token_id)

        text = tokenizer.decode(input_ids)
        parsed_ids = tokenizer.encode(text, add_special_tokens=False)

        special_token_preserved = special_token_id in parsed_ids

        assert special_token_preserved, (
            f"Failed: input {input_ids}, got {parsed_ids}. "
            f"Special token {special_token_id} should be present. "
            f"{[tokenizer.decode([x]) for x in input_ids]} != {[tokenizer.decode([x]) for x in parsed_ids]}. "
            f"Full text: '{text}' vs '{tokenizer.decode(parsed_ids)}'"
        )

    # Calculate total iterations for tqdm
    test_vocab = list(set(vocab_ids[:1000] + shared_tokens))  # Use first 1000 vocab ids + shared tokens
    pair_tests = len(test_vocab) * len(new_token_ids) * 2  # prefix and suffix
    special_pair_tests = len(new_token_ids) * (len(new_token_ids) - 1) * 2  # permutations of special tokens
    total_tests = pair_tests + special_pair_tests

    space = tokenizer.encode(' ')[0]
    with tqdm(total=total_tests, desc="Testing token combinations") as pbar:
        # Test pairs (prefix and suffix scenarios)
        for v, t in itertools.product(test_vocab, new_token_ids):
            assert_correct_parsing(t, [v], 0)  # Prefix test
            assert_correct_parsing(t, [v], 1)  # Suffix test

            assert_correct_parsing(t, [space, v], 0)
            assert_correct_parsing(t, [space, v], 1)
            assert_correct_parsing(t, [v, space], 0)
            assert_correct_parsing(t, [v, space], 1)
            pbar.update(2)

        # Test combinations of special tokens
        for t1, t2 in itertools.permutations(new_token_ids, 2):
            assert_correct_parsing(t1, [t2], 0)  # Prefix test
            assert_correct_parsing(t1, [t2], 1)  # Suffix test

            assert_correct_parsing(t1, [space, t2], 0)
            assert_correct_parsing(t1, [space, t2], 1)
            assert_correct_parsing(t1, [t2, space], 0)
            assert_correct_parsing(t1, [t2, space], 1)
            pbar.update(2)

    print("All tests passed successfully.")


def assert_parsability_2(tokenizer, special_tokens, num_tests=1000, max_sequence_length=50):
    ''' Test that a token can be parsed back out of a randomly generated string '''
    from tqdm import tqdm
    import random
    # Add special tokens to the tokenizer
    # num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    # print(f"Number of tokens added: {num_added_tokens}")

    # Get vocabulary and special token IDs
    vocab = tokenizer.get_vocab()
    vocab_ids = list(vocab.values())
    new_token_ids = [vocab[token] for token in special_tokens]

    def generate_random_sequence(length):
        return random.choices(vocab_ids, k=length)

    def insert_special_token(sequence, special_token_id):
        insert_position = random.randint(0, len(sequence))
        sequence.insert(insert_position, special_token_id)
        return sequence

    def assert_special_token_preserved(sequence, special_token_id):
        text = tokenizer.decode(sequence)
        parsed_ids = tokenizer.encode(text, add_special_tokens=False)

        special_token_preserved = special_token_id in parsed_ids

        assert special_token_preserved, (
            f"Failed: Special token {special_token_id} not found in parsed sequence. "
            f"Original: {sequence} "
            f"Parsed: {parsed_ids} "
            f"Original text: '{text}' "
            f"Parsed text: '{tokenizer.decode(parsed_ids)}'"
        )

    with tqdm(total=num_tests * len(special_tokens), desc="Testing random insertions") as pbar:
        for special_token_id in new_token_ids:
            for _ in range(num_tests):
                # Generate a random sequence
                sequence_length = random.randint(1, max_sequence_length)
                sequence = generate_random_sequence(sequence_length)

                # Insert the special token
                sequence_with_special = insert_special_token(sequence, special_token_id)

                # Assert that the special token is preserved
                assert_special_token_preserved(sequence_with_special, special_token_id)

                pbar.update(1)

    print("All random insertion tests passed successfully.")



def assert_parsability_3(tokenizer, special_tokens, num_tests=1000, max_sequence_length=50):
    ''' Test that a token can be parsed back out of a randomly generated string built out of characters present in the tokens'''
    from tqdm import tqdm
    import random

    # Add special tokens to the tokenizer
    # num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    # print(f"Number of tokens added: {num_added_tokens}")

    # Get vocabulary and special token IDs
    vocab = tokenizer.get_vocab()
    new_token_ids = [vocab[token] for token in special_tokens]

    # Extract unique characters from special tokens
    special_chars = list(set(''.join(special_tokens)))
    print(f"Unique characters in special tokens: {special_chars}")

    def generate_random_string(length):
        return ''.join(random.choices(special_chars, k=length))

    def insert_special_token(string, special_token):
        insert_position = random.randint(0, len(string))
        return string[:insert_position] + special_token + string[insert_position:]

    def assert_special_token_preserved(string, special_token_id):
        parsed_ids = tokenizer.encode(string, add_special_tokens=False)

        special_token_preserved = special_token_id in parsed_ids

        assert special_token_preserved, (
            f"Failed: Special token {special_token_id} not found in parsed sequence. "
            f"Original string: '{string}' "
            f"Parsed IDs: {parsed_ids} "
            f"Parsed text: '{tokenizer.decode(parsed_ids)}'"
        )

    with tqdm(total=num_tests * len(special_tokens), desc="Testing with special char sampling") as pbar:
        for special_token, special_token_id in zip(special_tokens, new_token_ids):
            for _ in range(num_tests):
                # Generate a random string
                string_length = random.randint(1, max_sequence_length)
                random_string = generate_random_string(string_length)

                # Insert the special token
                string_with_special = insert_special_token(random_string, special_token)

                # Assert that the special token is preserved
                assert_special_token_preserved(string_with_special, special_token_id)

                pbar.update(1)

    print("All special character sampling tests passed successfully.")


##################################################
# Sandbox

if False:
    import os
    import random

    SEED = 152
    torch.manual_seed(152)
    random.seed(SEED)

    DEVICE = 'cuda:1'
    BATCH_SIZE = 32

    # model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
    model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
    # model_name = os.path.expanduser("~/_/models/Qwen2-7B")

    try:
        fail
        already_loaded
    except:
        print('Loading model')
        model = Q.Qwen2ForCausalLM.from_pretrained(
            model_name,
            # torch_dtype="auto",
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            _attn_implementation='eager',
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Add new tokens
        new_token_pairs = [("[A]", "cat"), ("[B]", "dog"), ("[C]", "bird")]
        add_and_initialize_tokens(model, tokenizer, new_token_pairs)
        test_token_behavior(model, tokenizer, new_token_pairs)

        toks = [x[0] for x in new_token_pairs]
        assert_parsability_1(tokenizer, toks)
        assert_parsability_2(tokenizer, toks)
        assert_parsability_3(tokenizer, toks)

        already_loaded = True



# print()
# a = tokenizer("there was a^@Q")['input_ids']
# b = tokenizer("there was a&nbsp")['input_ids']
# print(a)
# print(b)
# print([tokenizer.decode(x) for x in a])
# print([tokenizer.decode(x) for x in b])
