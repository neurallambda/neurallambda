'''

NOT COMPLETED YET

'''
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

NOT COMPLETED YET

##########
# Data

# def generate_sine_wave(seq_length,
#                        num_samples,
#                        freq_range=(0.5, 1.5),
#                        amp_range=(0.8, 1.2),
#                        phase_range=(0, np.pi),
#                        noise_level=0.1,
#                        sample_rate=100):
#     data = []
#     for _ in range(num_samples):
#         freq = np.random.uniform(*freq_range)
#         amp = np.random.uniform(*amp_range)
#         phase = np.random.uniform(*phase_range)

#         x = np.linspace(0, seq_length / sample_rate, seq_length)
#         y = amp * np.sin(2 * np.pi * freq * x + phase)
#         y += np.random.normal(0, noise_level, size=y.shape)  # Add Gaussian noise
#         data.append(y)

#     return np.array(data)


def generate_sine_wave(seq_length,
                               num_samples,
                               num_waves=3,
                               freq_range=(0.5, 1.5),
                               amp_range=(0.8, 1.2),
                               phase_range=(0, np.pi),
                               noise_level=0.1,
                               sample_rate=100,
                               change_points=2):
    data = []
    for _ in range(num_samples):
        x = np.linspace(0, seq_length / sample_rate, seq_length)
        y = np.zeros(seq_length)

        change_indices = np.sort(np.random.choice(seq_length, change_points, replace=False))

        start_idx = 0
        for idx in change_indices:
            for _ in range(num_waves):
                freq = np.random.uniform(*freq_range)
                amp = np.random.uniform(*amp_range)
                phase = np.random.uniform(*phase_range)

                y_segment = amp * np.sin(2 * np.pi * freq * x[start_idx:idx] + phase)
                y[start_idx:idx] += y_segment

            start_idx = idx

        # Final segment
        for _ in range(num_waves):
            freq = np.random.uniform(*freq_range)
            amp = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)

            y_segment = amp * np.sin(2 * np.pi * freq * x[start_idx:] + phase)
            y[start_idx:] += y_segment

        # Add Gaussian noise
        y += np.random.normal(0, noise_level, size=y.shape)
        data.append(y)

    return np.array(data)

# Update the parameters for sequence length and number of samples
seq_length = 500  # Total number of samples in the signal
num_samples = 2000  # Number of sine wave samples to generate

# Generate the new sine wave data
data = generate_sine_wave(seq_length, num_samples)

# Prepare the dataset in the Hugging Face's datasets format
inputs = data[:, :-1]  # All but the last value as input
targets = data[:, 1:]  # All but the first value as target
dataset = Dataset.from_dict({'input': inputs, 'target': targets})

# Split the dataset into training and validation sets
dataset_dict = DatasetDict({
    'train': dataset.shuffle(seed=42).select(range(int(num_samples * 0.5))),
    'validation': dataset.shuffle(seed=42).select(range(int(num_samples * 0.5), num_samples))
})

# Define a function to convert dataset entries to tensors
def collate_fn(batch):
    inputs = [torch.FloatTensor(x['input']).unsqueeze(-1) for x in batch]
    targets = [torch.FloatTensor(x['target']).unsqueeze(-1) for x in batch]
    return torch.stack(inputs), torch.stack(targets)

# Create DataLoaders for training and validation
batch_size = 64
train_loader = DataLoader(dataset_dict['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset_dict['validation'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


##########
# Model

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

# Initialize the model
input_size = 1  # Since the input is a single value at each time step
hidden_size = 20  # Number of features in the hidden state
output_size = 1  # Since the output is a single value at each time step

model = SimpleRNN(input_size, hidden_size, output_size)


##########
# Train

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Number of epochs for training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}')


##########
# Validate

with torch.no_grad():
    model.eval()
    val_loss = 0.0
    for inputs, targets in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')


##########
# Visualization

def plot_sine_waves(data, num_waves=5):
    plt.figure(figsize=(10, 8))
    for i in range(num_waves):
        plt.subplot(num_waves, 1, i+1)
        plt.plot(data[i], label='Sine Wave')
        plt.title(f'Sine Wave Sample {i+1}')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Plot some original sine waves
plot_sine_waves(inputs[:5])

def plot_predictions(model, data_loader, num_waves=5):
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i == num_waves: break
            outputs = model(inputs)
            plt.figure(figsize=(10, 4))
            plt.plot(inputs[i].squeeze().numpy(), label='Original')
            plt.plot(outputs[i].squeeze().numpy(), label='Predicted')
            plt.title(f'Original vs Predicted for Sample {i+1}')
            plt.legend()
            plt.show()

# Plot some predictions
plot_predictions(model, val_loader)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

# Assuming you have saved train and validation losses in lists
# train_losses = [...]
# val_losses = [...]
# plot_loss(train_losses, val_losses)
