# '''

# Metalearning with pretraining on Omniglot, Testing on Mini-Imagenet Test

# '''

# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from datasets import load_dataset
# import random
# from collections import defaultdict

# class SimpleConvClassifier(torch.nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.conv1_weight = torch.nn.Parameter(torch.randn(64, in_channels, 3, 3))
#         self.conv1_bias = torch.nn.Parameter(torch.zeros(64))
#         self.conv2_weight = torch.nn.Parameter(torch.randn(64, 64, 3, 3))
#         self.conv2_bias = torch.nn.Parameter(torch.zeros(64))
#         self.fc_weight = torch.nn.Parameter(torch.randn(num_classes, 64 * 5 * 5))
#         self.fc_bias = torch.nn.Parameter(torch.zeros(num_classes))

#     def forward(self, x):
#         x = F.conv2d(x, self.conv1_weight, self.conv1_bias, padding=1)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = F.conv2d(x, self.conv2_weight, self.conv2_bias, padding=1)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         x = F.linear(x, self.fc_weight.t(), self.fc_bias)
#         return x

# def n_way_classification(support_set, query, model):
#     N = len(support_set)
#     support_features = torch.stack([model(x.unsqueeze(0)) for x in support_set])
#     query_features = model(query.unsqueeze(0))

#     similarities = F.cosine_similarity(query_features.expand(N, -1), support_features, dim=1)
#     return similarities

# def train_epoch(model, support_set, queries, labels, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     loss = 0
#     for query, label in zip(queries, labels):
#         pred = n_way_classification(support_set, query, model)
#         loss += F.cross_entropy(pred.unsqueeze(0), torch.tensor([label]))
#     loss.backward()
#     optimizer.step()
#     return loss.item()

# def evaluate(model, support_set, queries, labels):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for query, label in zip(queries, labels):
#             pred = n_way_classification(support_set, query, model)
#             predicted_class = torch.argmax(pred).item()
#             correct += (predicted_class == label)
#             total += 1
#     return correct / total

# def load_omniglot(train_size=200, test_size=50):
#     dataset = load_dataset("dpdl-benchmark/omniglot")
#     train_ds = dataset["train"]
#     test_ds = dataset["test"]

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((28, 28)),
#         transforms.Grayscale()
#     ])

#     def preprocess(example):
#         example["image"] = transform(example["image"].convert("RGB"))
#         return example

#     train_ds = train_ds.map(preprocess)
#     test_ds = test_ds.map(preprocess)

#     # Group examples by alphabet
#     train_by_alphabet = defaultdict(list)
#     for idx, example in enumerate(train_ds):
#         train_by_alphabet[example['alphabet']].append(idx)

#     test_by_alphabet = defaultdict(list)
#     for idx, example in enumerate(test_ds):
#         test_by_alphabet[example['alphabet']].append(idx)

#     # Sample examples ensuring representation from each class
#     train_indices = []
#     for alphabet_examples in train_by_alphabet.values():
#         train_indices.extend(random.sample(alphabet_examples, min(len(alphabet_examples), train_size // len(train_by_alphabet))))

#     test_indices = []
#     for alphabet_examples in test_by_alphabet.values():
#         test_indices.extend(random.sample(alphabet_examples, min(len(alphabet_examples), test_size // len(test_by_alphabet))))

#     # Trim to exact sizes
#     train_indices = train_indices[:train_size]
#     test_indices = test_indices[:test_size]

#     return train_ds.select(train_indices), test_ds.select(test_indices)

# def sample_task(dataset, n_way, k_shot, q_query):
#     alphabets = random.sample(dataset.unique("alphabet"), n_way)
#     support_set = []
#     query_set = []
#     for class_idx, alphabet in enumerate(alphabets):
#         class_data = dataset.filter(lambda x: x["alphabet"] == alphabet)
#         class_samples = random.sample(range(len(class_data)), k_shot + q_query)
#         support_set.extend([class_data[i]["image"] for i in class_samples[:k_shot]])
#         query_set.extend([class_data[i]["image"] for i in class_samples[k_shot:]])

#     query_labels = torch.tensor([alphabets.index(dataset[i]["alphabet"]) for i in range(len(query_set))])
#     return support_set, query_set, query_labels

# def main():
#     n_way = 5
#     k_shot = 1
#     q_query = 5
#     epochs = 1000

#     train_ds, test_ds = load_omniglot(train_size=200, test_size=50)

#     model = SimpleConvClassifier(in_channels=1, num_classes=n_way)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(epochs):
#         support_set, queries, labels = sample_task(train_ds, n_way, k_shot, q_query)
#         loss = train_epoch(model, support_set, queries, labels, optimizer)

#         if epoch % 100 == 0:
#             train_acc = evaluate(model, support_set, queries, labels)
#             test_support, test_queries, test_labels = sample_task(test_ds, n_way, k_shot, q_query)
#             test_acc = evaluate(model, test_support, test_queries, test_labels)
#             print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


# main()
