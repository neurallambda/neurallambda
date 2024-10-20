'''

Continue the t13 series, and do N-way k-shot learning task, where a transformer backbone outputs the weights. Trained with "train-time-data", so hopefully, at inference we won't even need metalearning!


The N-way k-shot learning task, for testing few-shot learners:

    1. N classes: The task tests classification between N different classes that the model has not seen during training.

    2. k examples per class: For each of the N classes, the model is given only k labeled examples (usually k is a small number like 1 or 5).

    3. Support set: The N*k labeled examples make up the "support set" that the model can use to learn about the new classes.

    4. Query set: The model is then asked to classify new unlabeled examples (the "query set") from the N classes.

    5. Meta-learning: Models are typically trained on many different N-way k-shot tasks so they can learn to adapt quickly to new tasks at test time.

    6. Common settings: Typical configurations are 5-way 1-shot and 5-way 5-shot classification.

    7. Datasets: This task is often evaluated on datasets like Omniglot and Mini-ImageNet.

    8. Evaluation: Performance is measured by classification accuracy on the query set.

    The goal is for models to be able to learn new classes from very few examples, mimicking human-like few-shot learning capabilities. This task highlights a model's ability to rapidly adapt to new concepts, rather than requiring large amounts of training data.

Metalearning with pretraining on Omniglot, Testing on Mini-Imagenet Test

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import random
from t13_metalearning_hypernet_data import omniglot_n_way_k_shot, ALPHABET_DICT

from tqdm import tqdm
import warnings

from datetime import datetime
current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = SummaryWriter(f'log/t12_metalearning_3_optimizers_2/{current_time}')
LOG = True

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'

# Notes:
# F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
#   input: [B, in_channels, iH, iW]
#   weight: [out_channels, in_channels/groups, kH, kW]
#   bias: [out_channels]


class Model(torch.nn.Module):
    def __init__(self, in_channels, num_filters, num_classes, image_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.image_size = image_size

        # Conv1: [out_channels, in_channels, kernel_height, kernel_width]
        self.conv1_weight = torch.nn.Parameter(torch.randn(num_filters, in_channels, 3, 3))
        self.conv1_bias = torch.nn.Parameter(torch.zeros(num_filters))

        # Conv2: [out_channels, in_channels, kernel_height, kernel_width]
        self.conv2_weight = torch.nn.Parameter(torch.randn(num_filters, num_filters, 3, 3))
        self.conv2_bias = torch.nn.Parameter(torch.zeros(num_filters))

        # Calculate the size of the flattened features after convolutions
        self.feature_size = self._get_feature_size()

        # Classifier: [out_features, in_features]
        self.classifier_weight = torch.nn.Parameter(torch.randn(num_classes, self.feature_size))
        self.classifier_bias = torch.nn.Parameter(torch.zeros(num_classes))

    def _get_feature_size(self):
        # After Conv1 and MaxPool: image_size // 2
        # After Conv2 and MaxPool: image_size // 4
        final_spatial_size = self.image_size // 4
        return self.num_filters * final_spatial_size * final_spatial_size

    def forward(self, x):
        # Input x shape: [batch_size, in_channels, height, width]
        batch_size = x.shape[0]

        # First convolutional block
        # x shape: [batch_size, num_filters, height, width]
        x = F.conv2d(x, self.conv1_weight, self.conv1_bias, padding=1)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)
        # x shape: [batch_size, num_filters, height//2, width//2]
        x = F.max_pool2d(x, 2)

        # Second convolutional block
        # x shape: [batch_size, num_filters, height//2, width//2]
        x = F.conv2d(x, self.conv2_weight, self.conv2_bias, padding=1)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)
        # x shape: [batch_size, num_filters, height//4, width//4]
        x = F.max_pool2d(x, 2)

        # Flatten
        # x shape: [batch_size, num_filters * (height//4) * (width//4)]
        x = x.view(batch_size, -1)

        # Classifier
        # x shape: [batch_size, num_classes]
        x = F.linear(x, self.classifier_weight, self.classifier_bias)

        return x

##################################################
# Training

def run_epoch(model, dataloader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):

            # batch (Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]):
            #     A batch containing a single task, where:
            #     - The first element is the support set: a list of N*k tuples, each containing
            #       (batched_image, batched_label) for support examples.
            #     - The second element is the query set: a list of N*q tuples, each containing
            #       (batched_image, batched_label) for query examples.

            # (Pdb) type(batch)
            # <class 'list'>
            # (Pdb) len(batch)
            # 2
            # (Pdb) type(batch[0])
            # <class 'list'>
            # (Pdb) len(batch[0])
            # 5
            # (Pdb) type(batch[0][0])
            # <class 'list'>
            # (Pdb) len(batch[0][0])
            # 2
            # (Pdb) batch[0][0][0].shape
            # torch.Size([32, 28, 28])
            # (Pdb) batch[0][0][1].shape
            # torch.Size([32])


            # supports: N*k tuples
            # queries: N queries (or N*q if multiple queries)
            supports, queries = batch
            support_imgs = [x[0].to(device).unsqueeze(1) for x in supports]  # N*k tensors, shape=[B, 1, IMG_SIZE, IMG_SIZE]
            support_labels = [x[1].to(device) for x in supports]  # N*k tensors, shape=[B]
            query_imgs = [x[0].to(device).unsqueeze(1) for x in queries]
            query_labels = [x[1].to(device) for x in queries]
            B = query_labels[0].shape[0]

            #####
            # Go

            # breakpoint()

            loss = 0

            for img, target_label in zip(support_imgs + query_imgs, support_labels + query_labels):
                output_labels = model(img)
                loss = loss + F.cross_entropy(output_labels, target_label)

            loss = loss / (len(support_imgs) + len(query_imgs))

            if train:
                optimizer.zero_grad()
                loss.backward()


                MAX_GRAD_NORM = 1.0
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

                # # TODO: why are these grads so small?! this is such a hack
                # with torch.no_grad():
                #     lor_models['lor_proj'][LOR_LAYER].w1.weight.grad[:] *= 10_000
                #     lor_models['lor_proj'][LOR_LAYER].w2.weight.grad[:] *= 10_000
                #     lor_models['lor_proj'][LOR_LAYER].w3.weight.grad[:] *= 10_000

                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples

    # Log weight histogram
    if LOG and train:
        try:
            # for name, param in itertools.chain(model.named_parameters(), lor_models.named_parameters()):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'weights/{name}', param.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'grads/{name}', param.grad.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
        except Exception as e:
            warnings.warn(f'Failed to write to tensorboard: {e}')

    return avg_loss


##################################################
# Go

global_epoch = 0

# Parameters
train_alphabets = ["Latin", "Greek"]
test_alphabets = ["Mongolian"]

image_size = 28
num_filters = 64
n_way = 5  # for 5-way classification
k_shot = 1  # for 1-shot learning
q_query = 1  # query examples per class
num_tasks = 100  # number of tasks per epoch

num_epochs = 100
batch_size = 32
lr = 1e-3
wd = 1e-2




train_dl, test_dl = omniglot_n_way_k_shot(
    train_alphabets,
    test_alphabets,
    n_way,
    k_shot,
    q_query,
    num_tasks,
    image_size,
    batch_size,
)


# Initialize model
model = Model(in_channels=1, num_filters=num_filters, num_classes=n_way, image_size=image_size).to(DEVICE)

parameters = [{
    'params': model.parameters(),
    'lr': lr,
    'wd': wd
}]

optimizer = optim.AdamW(parameters)

####################

train_losses = []
test_losses = []
best_loss = float('inf')

model.train()
for epoch in range(num_epochs):
    global_epoch += 1
    train_loss = run_epoch(model, train_dl, optimizer, DEVICE, train=True)
    train_losses.append(train_loss)
    writer.add_scalars('loss', {'train': train_loss}, global_epoch)

    if epoch % 1 == 0:
        test_loss = run_epoch(model, test_dl, optimizer, DEVICE, train=False)
        # test_loss = 0
        test_losses.append(test_loss)
        writer.add_scalars('loss', {'test': test_loss}, global_epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
