'''

Load ARC-AGI data

A JSON file looks like:
{
  "train": [
    {"input": [[1, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
    {"input": [[0, 0], [4, 0]], "output": [[4, 4], [4, 4]]},
    {"input": [[0, 0], [6, 0]], "output": [[6, 6], [6, 6]]}
  ],
  "test": [
    {"input": [[0, 0], [0, 8]], "output": [[8, 8], [8, 8]]}
  ]
}


We format it to:

# Example 1
## Input
1 0
0 0
## Output
1 1
1 1

# Example 2
## Input
0 0
4 0
## Output
4 4
4 4

# Example 3
## Input
0 0
6 0
## Output
6 6
6 6

# Example 4
## Input
0 0
0 8
## Output
8 8
8 8

'''

import os
import json
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import transformers.models.qwen2.modeling_qwen2 as Q
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'
BATCH_SIZE = 1

# class TokenizedJsonDataset(Dataset):
#     def __init__(self, data_dir, tokenizer):
#         self.data = []
#         self.tokenizer = tokenizer
#         for filename in os.listdir(data_dir):
#             if filename.endswith('.json'):
#                 file_path = os.path.join(data_dir, filename)
#                 with open(file_path, 'r') as f:
#                     json_data = json.load(f)
#                     formatted_data = self.format_json(json_data)
#                     tokenized = self.tokenizer.encode(
#                         formatted_data,
#                         return_tensors='pt'
#                     )
#                     self.data.append(tokenized.squeeze(0))

#         # Sort data by length in descending order
#         self.data.sort(key=lambda x: len(x), reverse=True)

#     def format_json(self, json_data):
#         formatted = ""
#         example_num = 1
#         for section in ['train', 'test']:
#             for item in json_data.get(section, []):
#                 formatted += f"# Example {example_num}\n"
#                 formatted += "## Input\n"
#                 for row in item['input']:
#                     formatted += " ".join(map(str, row)) + "\n"
#                 formatted += "## Output\n"
#                 for row in item['output']:
#                     formatted += " ".join(map(str, row)) + "\n"
#                 formatted += "\n"
#                 example_num += 1
#         return formatted.strip()

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# class TokenizedJsonDataset(Dataset):
#     def __init__(self, data_dir, tokenizer):
#         self.data = []
#         self.tokenizer = tokenizer
#         for filename in os.listdir(data_dir):
#             if filename.endswith('.json'):
#                 file_path = os.path.join(data_dir, filename)
#                 with open(file_path, 'r') as f:
#                     json_data = json.load(f)
#                     formatted_data_list = self.format_json(json_data)
#                     for formatted_data in formatted_data_list:
#                         tokenized = self.tokenizer.encode(
#                             formatted_data,
#                             return_tensors='pt'
#                         )
#                         self.data.append(tokenized.squeeze(0))

#         # Sort data by length in descending order
#         self.data.sort(key=lambda x: len(x), reverse=True)

#     def format_json(self, json_data):
#         formatted_list = []
#         example_num = 1
#         for section in ['train', 'test']:
#             for item in json_data.get(section, []):
#                 formatted = f"# Example {example_num}\n"
#                 formatted += "## Input\n"
#                 for row in item['input']:
#                     formatted += "".join(map(str, row)) + "\n"
#                 formatted += "## Output\n"
#                 for row in item['output']:
#                     formatted += "".join(map(str, row)) + "\n"
#                 formatted_list.append(formatted.strip())
#                 example_num += 1
#         return formatted_list

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


class TokenizedJsonDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    formatted_data = self.format_json(json_data)
                    tokenized = self.tokenizer.encode(
                        formatted_data,
                        return_tensors='pt'
                    )
                    self.data.append(tokenized.squeeze(0))

        # Sort data by length in descending order
        self.data.sort(key=lambda x: len(x), reverse=True)

    def format_json(self, json_data):
        formatted = ""
        example_num = 1
        for section in ['train', 'test']:
            for item in json_data.get(section, []):
                formatted += f"# Example {example_num}\n"
                formatted += "## Input\n"
                for row in item['input']:
                    formatted += "".join(map(str, row)) + "\n"
                formatted += "## Output\n"
                for row in item['output']:
                    formatted += "".join(map(str, row)) + "\n"
                formatted += "\n"
                example_num += 1
        return formatted.strip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(padding_value, batch):
    # Pad right
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
    attention_mask = (padded_batch != padding_value).long()

    # # Pad left
    # max_len = max(len(x) for x in batch)
    # padded_batch = []
    # for seq in batch:
    #     padding = (max_len - len(seq)) * [padding_value]
    #     padded_seq = torch.cat((torch.tensor(padding), seq))
    #     padded_batch.append(padded_seq)
    # padded_batch = torch.stack(padded_batch)

    return padded_batch, attention_mask

def create_dataloaders(train_dir, eval_dir, tokenizer, batch_size):
    eos = tokenizer.eos_token_id
    train_dataset = TokenizedJsonDataset(train_dir, tokenizer)
    eval_dataset = TokenizedJsonDataset(eval_dir, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, eos))
    val_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, eos))

    return train_dataloader, val_dataloader

def plot_data_statistics(loader, title):
    lengths = [len(item) for item in loader.dataset.data]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title(f'{title} - Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot(lengths)
    plt.title(f'{title} - Sequence Length Box Plot')
    plt.ylabel('Sequence Length')
    plt.show()

    print(f"{title} Statistics:")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    print(f"Standard deviation: {np.std(lengths):.2f}")

def plot_batch_statistics(loader, title):
    batch_lengths = []
    for batch, _ in loader:
        batch_lengths.append(batch.shape[1])
        if len(batch_lengths) == 10:  # Plot stats for first 10 batches
            break

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(batch_lengths) + 1), batch_lengths)
    plt.title(f'{title} - Batch Lengths (First 10 Batches)')
    plt.xlabel('Batch Number')
    plt.ylabel('Sequence Length')
    plt.show()

    print(f"{title} Batch Statistics:")
    print(f"Min batch length: {min(batch_lengths)}")
    print(f"Max batch length: {max(batch_lengths)}")
    print(f"Mean batch length: {np.mean(batch_lengths):.2f}")


##########
# Model

model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")

cpu_model = Q.Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu",
    _attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = cpu_model.to('cuda')


optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)



# $$$$$$$$$$
# Experiment truncating weights
dim = 16
with torch.no_grad():
    # embeddings
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight[:, :dim]
    model.model.embed_tokens.embedding_dim = dim

    for layer in model.model.layers:
        # Truncate self-attention projection layers
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:, :dim]
        layer.self_attn.q_proj.in_features = dim

        # k_proj and v_proj have smaller out_features
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[:, :dim]
        layer.self_attn.k_proj.in_features = dim

        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[:, :dim]
        layer.self_attn.v_proj.in_features = dim

        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:dim, :]
        layer.self_attn.o_proj.out_features = dim

        # Truncate MLP layers
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[:, :dim]
        layer.mlp.gate_proj.in_features = dim

        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[:, :dim]
        layer.mlp.up_proj.in_features = dim

        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:dim, :]
        layer.mlp.down_proj.out_features = dim

    # Truncate the final layer norm
    model.model.norm.weight.data = model.model.norm.weight.data[:dim]

    # Truncate the final linear layer (lm_head)
    model.lm_head.weight.data = model.lm_head.weight.data[:, :dim]

# $$$$$$$$$$
# Experiment with input sizes
x = torch.tensor([20] * 3000).long().cuda().unsqueeze(0)
for _ in range(10):
    y = model(x, labels=x)
    y.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

BRK


# # $$$$$$$$$$
# # Experiment with deleting layers
# start_ix = 4
# stop_ix = 20
# for i in reversed(range(start_ix, stop_ix)):
#     del model.model.layers[i]
# d = stop_ix - start_ix
# for i in range(start_ix, len(model.model.layers)):
#     model.model.layers[i].self_attn.layer_idx -= d

tokenizer = AutoTokenizer.from_pretrained(model_name)


##########
# Data
train_dir = os.path.expanduser('~/_/data/ARC-AGI/data/training')
eval_dir = os.path.expanduser('~/_/data/ARC-AGI/data/evaluation')
train_dataloader, val_dataloader = create_dataloaders(train_dir, eval_dir, tokenizer, BATCH_SIZE)

print(f"Number of training samples: {len(train_dataloader.dataset)}")
print(f"Number of evaluation samples: {len(val_dataloader.dataset)}")

# Example of iterating through the data
for batch, attention_mask in train_dataloader:
    print("Batch shape:", batch.shape)
    print("Attention mask shape:", attention_mask.shape)
    for i in range(min(3, len(batch))):
        print("----------")
        print(f"Sample {i} length:", len(batch[i].nonzero()))
        print("Sample tokens:", batch[i][:10])
        print("Sample text:", tokenizer.decode(batch[i], skip_special_tokens=True))
    break  # Just print the first batch


print(f"Number of training samples: {len(train_dataloader.dataset)}")
print(f"Number of evaluation samples: {len(val_dataloader.dataset)}")

# # Plot data statistics
# plot_data_statistics(train_dataloader, "Training Data")
# plot_data_statistics(val_dataloader, "Evaluation Data")

# # Plot batch statistics
# plot_batch_statistics(train_dataloader, "Training Batches")
# plot_batch_statistics(val_dataloader, "Evaluation Batches")



###########
# Params

print('training ALL params')
params = model.parameters()

# print('training LSTM params only')
# params = [x[1] for x in model.named_parameters() if 'lstm' in x[0]]


##########
# Go

LR = 1e-4
WD = 0.0
NUM_EPOCHS = 5

# Set up optimizer and learning rate scheduler
optimizer = AdamW(params, lr=LR, weight_decay=WD)

num_training_steps = NUM_EPOCHS * len(train_dataloader)

# Training loop
val_losses = []
train_losses = []

for epoch in range(NUM_EPOCHS):

    # TRAINING
    model.train()
    batch_losses = []
    for input_ids, attention_mask in tqdm(train_dataloader):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = input_ids
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(loss.item())
        print(f'step loss: {loss.item():.3f}')

    train_losses.append(sum(batch_losses) / len(batch_losses))

    # VALIDATION
    model.eval()
    with torch.no_grad():
        batch_val_losses = []
        for input_ids, attention_mask in tqdm(train_dataloader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = input_ids
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits
            loss = outputs.loss
            batch_val_losses.append(loss)
        val_losses.append(sum(batch_val_losses) / len(batch_val_losses))

    print(f"Epoch {epoch+1}/{num_epochs} completed. Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")
