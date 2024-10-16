'''

Prepare dataset for Binding Attention experiment

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
import transformers.models.qwen2.modeling_qwen2 as Q

from typing import Optional, Tuple
import warnings
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_dataset(dataset: Dataset):
    def process_example(example):
        # Create the prompt
        prompt = f"""Below is a fill-in-the-blank problem. Choose the correct option to complete the sentence.

Problem: {example['sentence']}

Options:
1. {example['option1']}
2. {example['option2']}

The correct answer is: """

        # Select the correct answer
        answer = example['option1'] if example['answer'] == '1' else example['option2']

        return {
            'prompt': prompt,
            'answer': answer
        }

    # Apply the processing function to the dataset
    processed_dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    return processed_dataset

def create_dataloader(dataset: Dataset, tokenizer, batch_size):
    def collate_fn(batch):
        tokenized_prompts = [tokenizer(item['prompt'], return_tensors="pt")['input_ids'].squeeze(0) for item in batch]
        tokenized_answers = [tokenizer(item['answer'], return_tensors="pt")['input_ids'].squeeze(0) for item in batch]

        # Find max lengths
        max_prompt_length = max([len(x) for x in tokenized_prompts])
        max_answer_length = max([len(x) for x in tokenized_answers]) + 1  # +1 for eos token

        # Prepare tensors
        prompt_ids = torch.full((len(batch), max_prompt_length), tokenizer.pad_token_id)
        answer_ids = torch.full((len(batch), max_answer_length), tokenizer.pad_token_id)
        prompt_attention_mask = torch.zeros((len(batch), max_prompt_length))
        answer_attention_mask = torch.zeros((len(batch), max_answer_length))

        # Fill tensors
        for i in range(len(batch)):
            prompt_length = len(tokenized_prompts[i])
            answer_length = len(tokenized_answers[i])

            # Left-pad prompts
            start_idx = max_prompt_length - prompt_length
            prompt_ids[i, start_idx:] = tokenized_prompts[i]
            prompt_attention_mask[i, start_idx:] = 1

            # Left-align answers and right-pad
            answer_ids[i, :answer_length] = tokenized_answers[i]
            answer_attention_mask[i, :answer_length] = 1

            # Ensure EOS token at the end of each answer
            if answer_ids[i, answer_length - 1] != tokenizer.eos_token_id:
                answer_ids[i, answer_length] = tokenizer.eos_token_id
                answer_attention_mask[i, answer_length] = 1

        return {
            'prompt_ids': prompt_ids,
            'answer_ids': answer_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'answer_attention_mask': answer_attention_mask
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

def debug_dataloader(dataloader, tokenizer, num_samples=3):
    total = 0
    for i, batch in enumerate(dataloader):
        if total >= num_samples:
            break

        prompt_ids = batch['prompt_ids']
        answer_ids = batch['answer_ids']

        for j in range(min(num_samples - total, prompt_ids.shape[0])):
            total += 1
            print(f"\nSample {i * dataloader.batch_size + j + 1}:")
            print("Prompt:")
            print(tokenizer.decode(prompt_ids[j][prompt_ids[j] != tokenizer.pad_token_id]))
            print("\nAnswer:")
            print(tokenizer.decode(answer_ids[j][answer_ids[j] != tokenizer.pad_token_id]))
            print("-" * 50)

        if total >= num_samples:
            break
