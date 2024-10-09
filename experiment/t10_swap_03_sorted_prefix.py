'''

An experiment checking whether a single layer transformer encoder can identify a sorted prefix

RESULTS:
- stock transformer doesn't generalize at all
- fixing position embeddings/encodings helps

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the model
class SortedPrefixTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, num_layers=1, max_seq_length=100):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Embedding(10, d_model)  # 10 possible digits (0-9)
        self.fc = nn.Linear(d_model, 2)  # Binary classification: part of sorted prefix or not
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.max_seq_length = max_seq_length

    def forward(self, src):
        src = self.embedding(src)
        batch_size, seq_length, _ = src.shape

        # Generate random offsets for each sample in the batch
        offsets = torch.randint(0, self.max_seq_length - seq_length + 1, (batch_size,), device=src.device)

        # Apply positional encoding with random offsets
        src = self.pos_encoder(src, offsets)
        output = self.transformer(src)
        return self.fc(output)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, offsets):
        batch_size, seq_length, _ = x.shape
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        positions = positions + offsets.unsqueeze(1)
        return x + self.pe[positions]

def generate_data(num_samples, seq_length):
    data, labels = [], []
    for _ in range(num_samples):
        prefix_len = np.random.randint(1, seq_length)
        prefix = sorted(np.random.randint(1, 10, prefix_len))  # Ensure last element is never 0
        suffix = [np.random.randint(0, prefix[-1])] + list(np.random.randint(0, 10, seq_length - prefix_len - 1))
        seq = np.pad(prefix + suffix, (0, max(0, seq_length - len(prefix) - len(suffix))), 'constant')
        label = [1] * prefix_len + [0] * (seq_length - prefix_len)
        data.append(seq)
        labels.append(label)
    return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch, labels in dataloader:
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output.view(-1, 2), labels.long().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in dataloader:
            batch, labels = batch.to(device), labels.to(device)
            output = model(batch)
            loss = criterion(output.view(-1, 2), labels.long().view(-1))
            total_loss += loss.item()
            pred = output.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return total_loss / len(dataloader), correct / total

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_layers = 1
max_seq_length = 100
model = SortedPrefixTransformer(num_layers=num_layers, max_seq_length=max_seq_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

# Generate data
train_len = 20
val_len = 40
train_data, train_labels = generate_data(10000, train_len)
val_data, val_labels = generate_data(1000, val_len)

# Create dataloaders
train_dataloader = DataLoader(TensorDataset(train_data, train_labels), batch_size=64, shuffle=True)
val_dataloader = DataLoader(TensorDataset(val_data, val_labels), batch_size=64)

# Training loop
num_epochs = 40
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Test on a few examples
test_data, test_labels = generate_data(5, val_len)
model.eval()
with torch.no_grad():
    output = model(test_data.to(device))
    pred = output.argmax(dim=-1)
    for i in range(5):
        print(f"Input : {test_data[i].tolist()}")
        print(f"Predic: {pred[i].tolist()}")
        print(f"Target: {test_labels[i].int().tolist()}")
        print()
