'''

OMNIGLOT

Total number of unique alphabets: 50

Alphabet distribution:

|----+-------------------------------------------+---------------+---------------+-------------------------|
|    |                                           | Total samples | Unique labels | Avg instances per label |
|----+-------------------------------------------+---------------+---------------+-------------------------|
|  0 | Alphabet_of_the_Magi                      |           400 |            20 |                      20 |
|  1 | Angelic                                   |           400 |            20 |                      20 |
|  2 | Anglo-Saxon_Futhorc                       |           580 |            29 |                      20 |
|  3 | Arcadian                                  |           520 |            26 |                      20 |
|  4 | Armenian                                  |           820 |            41 |                      20 |
|  5 | Asomtavruli_(Georgian)                    |           800 |            40 |                      20 |
|  6 | Atemayar_Qelisayer                        |           520 |            26 |                      20 |
|  7 | Atlantean                                 |           520 |            26 |                      20 |
|  8 | Aurek-Besh                                |           520 |            26 |                      20 |
|  9 | Avesta                                    |           520 |            26 |                      20 |
| 10 | Balinese                                  |           960 |            24 |                      40 |
| 11 | Bengali                                   |           920 |            46 |                      20 |
| 12 | Blackfoot_(Canadian_Aboriginal_Syllabics) |           280 |            14 |                      20 |
| 13 | Braille                                   |           520 |            26 |                      20 |
| 14 | Burmese_(Myanmar)                         |           680 |            34 |                      20 |
| 15 | Cyrillic                                  |           660 |            33 |                      20 |
| 16 | Early_Aramaic                             |           880 |            22 |                      40 |
| 17 | Futurama                                  |           520 |            26 |                      20 |
| 18 | Ge_ez                                     |           520 |            26 |                      20 |
| 19 | Glagolitic                                |           900 |            45 |                      20 |
| 20 | Grantha                                   |           860 |            43 |                      20 |
| 21 | Greek                                     |          1440 |            24 |                      60 |
| 22 | Gujarati                                  |           960 |            48 |                      20 |
| 23 | Gurmukhi                                  |           900 |            45 |                      20 |
| 24 | Hebrew                                    |           440 |            22 |                      20 |
| 25 | Inuktitut_(Canadian_Aboriginal_Syllabics) |           320 |            16 |                      20 |
| 26 | Japanese_(hiragana)                       |          1040 |            52 |                      20 |
| 27 | Japanese_(katakana)                       |          1880 |            47 |                      40 |
| 28 | Kannada                                   |           820 |            41 |                      20 |
| 29 | Keble                                     |           520 |            26 |                      20 |
| 30 | Korean                                    |          1600 |            40 |                      40 |
| 31 | Latin                                     |          1560 |            26 |                      60 |
| 32 | Malay_(Jawi_-_Arabic)                     |           800 |            40 |                      20 |
| 33 | Malayalam                                 |           940 |            47 |                      20 |
| 34 | Manipuri                                  |           800 |            40 |                      20 |
| 35 | Mkhedruli_(Georgian)                      |           820 |            41 |                      20 |
| 36 | Mongolian                                 |           600 |            30 |                      20 |
| 37 | N_Ko                                      |           660 |            33 |                      20 |
| 38 | Ojibwe_(Canadian_Aboriginal_Syllabics)    |           280 |            14 |                      20 |
| 39 | Old_Church_Slavonic_(Cyrillic)            |           900 |            45 |                      20 |
| 40 | Oriya                                     |           920 |            46 |                      20 |
| 41 | Sanskrit                                  |          1680 |            42 |                      40 |
| 42 | Sylheti                                   |           560 |            28 |                      20 |
| 43 | Syriac_(Estrangelo)                       |           460 |            23 |                      20 |
| 44 | Syriac_(Serto)                            |           460 |            23 |                      20 |
| 45 | Tagalog                                   |           680 |            17 |                      40 |
| 46 | Tengwar                                   |           500 |            25 |                      20 |
| 47 | Tibetan                                   |           840 |            42 |                      20 |
| 48 | Tifinagh                                  |          1100 |            55 |                      20 |
| 49 | ULOG                                      |           520 |            26 |                      20 |
|----+-------------------------------------------+---------------+---------------+-------------------------|


TEST ALPHABETS

Angelic
Atemayar_Qelisayer
Atlantean
Aurek-Besh
Avesta
Ge_ez
Glagolitic
Gurmukhi
Kannada
Keble
Malayalam
Manipuri
Mongolian
Old_Church_Slavonic_(Cyrillic)
Oriya
Sylheti
Syriac_(Serto)
Tengwar
Tibetan
ULOG


TRAIN ALPHABETS

Alphabet_of_the_Magi
Anglo-Saxon_Futhorc
Arcadian
Armenian
Asomtavruli_(Georgian)
Balinese
Bengali
Blackfoot_(Canadian_Aboriginal_Syllabics)
Braille
Burmese_(Myanmar)
Cyrillic
Early_Aramaic
Futurama
Grantha
Greek
Gujarati
Hebrew
Inuktitut_(Canadian_Aboriginal_Syllabics)
Japanese_(hiragana)
Japanese_(katakana)
Korean
Latin
Malay_(Jawi_-_Arabic)
Mkhedruli_(Georgian)
N_Ko
Ojibwe_(Canadian_Aboriginal_Syllabics)
Sanskrit
Syriac_(Estrangelo)
Tagalog
Tifinagh

'''


import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from torchvision import transforms
import random


##################################################
# Omniglot

ALPHABET_DICT = {
    0: "Alphabet_of_the_Magi",
    1: "Angelic",
    2: "Anglo-Saxon_Futhorc",
    3: "Arcadian",
    4: "Armenian",
    5: "Asomtavruli_(Georgian)",
    6: "Atemayar_Qelisayer",
    7: "Atlantean",
    8: "Aurek-Besh",
    9: "Avesta",
    10: "Balinese",
    11: "Bengali",
    12: "Blackfoot_(Canadian_Aboriginal_Syllabics)",
    13: "Braille",
    14: "Burmese_(Myanmar)",
    15: "Cyrillic",
    16: "Early_Aramaic",
    17: "Futurama",
    18: "Ge_ez",
    19: "Glagolitic",
    20: "Grantha",
    21: "Greek",
    22: "Gujarati",
    23: "Gurmukhi",
    24: "Hebrew",
    25: "Inuktitut_(Canadian_Aboriginal_Syllabics)",
    26: "Japanese_(hiragana)",
    27: "Japanese_(katakana)",
    28: "Kannada",
    29: "Keble",
    30: "Korean",
    31: "Latin",
    32: "Malay_(Jawi_-_Arabic)",
    33: "Malayalam",
    34: "Manipuri",
    35: "Mkhedruli_(Georgian)",
    36: "Mongolian",
    37: "N_Ko",
    38: "Ojibwe_(Canadian_Aboriginal_Syllabics)",
    39: "Old_Church_Slavonic_(Cyrillic)",
    40: "Oriya",
    41: "Sanskrit",
    42: "Sylheti",
    43: "Syriac_(Estrangelo)",
    44: "Syriac_(Serto)",
    45: "Tagalog",
    46: "Tengwar",
    47: "Tibetan",
    48: "Tifinagh",
    49: "ULOG"
}

ALPHABET_NAME_TO_ID = {v: k for k, v in ALPHABET_DICT.items()}


class OmniglotDataset(Dataset):
    def __init__(self, hf_dataset, image_size=28):
        self.dataset = hf_dataset
        self.resize = transforms.Resize((image_size, image_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Convert PIL Image to numpy array, then to PyTorch tensor
        image = np.array(item['image']).astype(np.float32) / 255.0
        image = np.mean(image, axis=-1)  # Convert to grayscale
        image = image * 2 - 1  # normalize to [-1, 1]
        image = torch.from_numpy(image).unsqueeze(0)  # Add single channel dimension
        image = self.resize(image)
        return image.squeeze(0), item['alphabet'], item['label']


def omniglot_datasets(train_alphabets: List[str], test_alphabets: List[str], image_size: int, batch_size: int = 32):
    # Load the dataset
    dataset = load_dataset("dpdl-benchmark/omniglot")

    # Convert alphabet names to IDs
    train_alphabet_ids = [ALPHABET_NAME_TO_ID[name] for name in train_alphabets]
    test_alphabet_ids = [ALPHABET_NAME_TO_ID[name] for name in test_alphabets]

    # Create a filter function
    def filter_alphabets(alphabet_id, allowed_alphabet_ids):
        return alphabet_id in allowed_alphabet_ids

    # Apply the filter to both train and test sets
    train_dataset = dataset['train'].filter(
        filter_alphabets,
        fn_kwargs={"allowed_alphabet_ids": train_alphabet_ids},
        input_columns=['alphabet']
    )
    test_dataset = dataset['test'].filter(
        filter_alphabets,
        fn_kwargs={"allowed_alphabet_ids": test_alphabet_ids},
        input_columns=['alphabet']
    )

    if False:
        # Print diagnostics for train dataset
        print("\nTrain Dataset Diagnostics:")
        print(f"Number of samples: {len(train_dataset)}")
        print(f"Number of batches: {len(train_dataset) // batch_size + (1 if len(train_dataset) % batch_size else 0)}")
        print(f"Size of first batch: {min(batch_size, len(train_dataset))}")
        print(f"Number of unique image labels: {len(set(train_dataset['label']))}")
        print(f"Alphabets included: {', '.join([ALPHABET_DICT[id] for id in train_alphabet_ids])}")

        # Print diagnostics for test dataset
        print("\nTest Dataset Diagnostics:")
        print(f"Number of samples: {len(test_dataset)}")
        print(f"Number of batches: {len(test_dataset) // batch_size + (1 if len(test_dataset) % batch_size else 0)}")
        print(f"Size of first batch: {min(batch_size, len(test_dataset))}")
        print(f"Number of unique image labels: {len(set(test_dataset['label']))}")
        print(f"Alphabets included: {', '.join([ALPHABET_DICT[id] for id in test_alphabet_ids])}")

    # Convert to OmniglotDataset
    train_dataset = OmniglotDataset(train_dataset, image_size)
    test_dataset = OmniglotDataset(test_dataset, image_size)

    return train_dataset, test_dataset


def omniglot_dataloader(train_alphabets: List[str], test_alphabets: List[str], image_size: int, batch_size: int = 32):
    train_dataset, test_dataset = omniglot_datasets(train_alphabets, test_alphabets, image_size, batch_size)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl

def visualize_samples(dataloader, num_samples=5):
    # Get a batch of data
    images, alphabet_indices, labels = next(iter(dataloader))

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle("Sample Images from Omniglot Dataset")

    for i in range(num_samples):
        # Get the image, alphabet, and label
        img = images[i].squeeze().numpy()
        alphabet = ALPHABET_DICT[alphabet_indices[i].item()]
        label = labels[i].item()

        # Display the image
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Alphabet: {alphabet}\nLabel: {label}")

    plt.tight_layout()
    plt.show()


##########
# Usage Example

if False:
    train_alphabets = ["Latin", "Greek"]
    test_alphabets = ["Mongolian"]

    train_loader, test_loader = omniglot_dataloader(train_alphabets, test_alphabets, image_size=32)

    # Verify the contents of a batch
    images, alphabets, labels = next(iter(train_loader))

    # Visualize samples from the train loader
    print("Visualizing samples from the train loader:")
    visualize_samples(train_loader)

    # Visualize samples from the test loader
    print("Visualizing samples from the test loader:")
    visualize_samples(test_loader)


##############################


class NShotTaskSampler(IterableDataset):
    """
    A sampler that creates N-way K-shot tasks for few-shot learning.
    Tasks are pre-generated during initialization for deterministic sampling.

    Args:
        dataset (Dataset): The dataset to sample from.
        n_way (int): Number of classes per task.
        k_shot (int): Number of support examples per class.
        q_query (int): Number of query examples per class.
        num_tasks (int): Number of tasks to generate.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """

    def __init__(self, dataset, n_way, k_shot, q_query, num_tasks, seed):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks
        self.labels = [item['label'] for item in dataset.dataset]

        # Pre-generate all tasks
        random.seed(seed)
        self.tasks = [self._generate_task() for _ in range(self.num_tasks)]

        # Reset random seed to not affect other randomness in the program
        random.seed()

        self._task_index = 0

    def __iter__(self):
        self._task_index = 0
        return self

    def __next__(self):
        if self._task_index >= self.num_tasks:
            raise StopIteration
        task = self.tasks[self._task_index]
        self._task_index += 1
        return task

    def _generate_task(self) -> Tuple[List, List]:
        """
        Generate a single N-way K-shot task.

        Returns:
            Tuple containing support set and query set.
        """
        classes = random.sample(list(set(self.labels)), self.n_way)

        support_set = []
        query_set = []
        class_ixs = list(range(len(classes)))
        random.shuffle(class_ixs)

        for class_ix, class_label in zip(class_ixs, classes):
            class_instances = [i for i, label in enumerate(self.labels) if label == class_label]
            selected_instances = random.sample(class_instances, self.k_shot + self.q_query)

            support_set.extend((self.dataset[i][0], class_ix) for i in selected_instances[:self.k_shot])
            query_set.extend((self.dataset[i][0], class_ix) for i in selected_instances[self.k_shot:])

        random.shuffle(query_set)
        return support_set, query_set


def omniglot_n_way_k_shot(
    train_alphabets: List[str],
    test_alphabets: List[str],
    n_way: int,
    k_shot: int,
    q_query: int,
    num_tasks: int,
    image_size: int,
    batch_size: int,
    seed_train: int,
    seed_test: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for N-way K-shot tasks on the Omniglot dataset.

    Args:
        train_alphabets (List[str]): List of alphabets to use for training.
        test_alphabets (List[str]): List of alphabets to use for testing.
        n_way (int): Number of classes per task.
        k_shot (int): Number of support examples per class.
        q_query (int): Number of query examples per class.
        num_tasks (int): Number of tasks to generate.
        image_size (int): Size to resize images to.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        Tuple of train and test DataLoaders.

        A batch off the train/test DataLoader looks like:

            List[Tuple[torch.Tensor (img),
                       torch.Tensor (label)]]

        IE a list of N*k tuples, each containing (batched_image, batched_label) for support examples.

    """
    train_dataset, test_dataset = omniglot_datasets(train_alphabets, test_alphabets, image_size)

    train_sampler = NShotTaskSampler(train_dataset, n_way, k_shot, q_query, num_tasks, seed=seed_train)
    test_sampler = NShotTaskSampler(test_dataset, n_way, k_shot, q_query, num_tasks, seed=seed_test)

    train_loader = DataLoader(train_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_sampler, batch_size=batch_size)

    return train_loader, test_loader


def visualize_n_way_k_shot_task(
    task_batch: Tuple[List[Tuple[torch.Tensor, torch.Tensor]],  # support: list of N*k tuples of (batched_image, batched_label)
                      List[Tuple[torch.Tensor, torch.Tensor]]   # query: list of N*k tuples of (batched_image, batched_label)
                      ],
    n_way: int,
    k_shot: int,
    q_query: int
):
    """
    Visualize a single N-way K-shot task from a batch.

    Args:
        task_batch (Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]):
            A batch containing a single task, where:
            - The first element is the support set: a list of N*k tuples, each containing
              (batched_image, batched_label) for support examples.
            - The second element is the query set: a list of N*q tuples, each containing
              (batched_image, batched_label) for query examples.
        n_way (int): Number of classes in the task.
        k_shot (int): Number of support examples per class.
        q_query (int): Number of query examples per class.
    """

    support_set, query_set = task_batch

    batch_ix = 0

    total_rows = k_shot + 1
    fig, axes = plt.subplots(total_rows, n_way, figsize=(3 * n_way, 3 * total_rows))
    fig.suptitle(f"{n_way}-way {k_shot}-shot Task with {q_query} Query Examples")

    if total_rows == 2:
        axes = axes.reshape(2, n_way)

    for i in range(n_way):
        for j in range(k_shot):
            idx = i * k_shot + j
            img   = support_set[idx][0][batch_ix]
            label = support_set[idx][1][batch_ix]
            ax = axes[j, i]
            ax.imshow(img.squeeze().numpy(), cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Class {label}")

    query_examples = {}
    for img, label in query_set:
        if label not in query_examples:
            query_examples[label[batch_ix]] = img[batch_ix]

    for i, (label, img) in enumerate(query_examples.items()):
        axes[-1, i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[-1, i].axis('off')
        axes[-1, i].set_title(f"Query (Class {label})")

    plt.tight_layout()
    plt.show()

# Usage example
if False:
    train_alphabets = ["Latin", "Greek"]
    test_alphabets = ["Mongolian"]
    n_way = 5
    k_shot = 4
    q_query = 1
    num_tasks = 100
    image_size = 28
    batch_size = 4

    train_loader, test_loader = omniglot_n_way_k_shot(
        train_alphabets, test_alphabets, n_way, k_shot, q_query, num_tasks, image_size, batch_size
    )

    print("Visualizing a sample task from the train loader:")
    visualize_n_way_k_shot_task(next(iter(train_loader)), n_way, k_shot, q_query)

    print("Visualizing a sample task from the test loader:")
    visualize_n_way_k_shot_task(next(iter(test_loader)), n_way, k_shot, q_query)


##################################################
#

class MiniImageNetDataset(Dataset):
    def __init__(self, hf_dataset, image_size=84):
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item['image'])
        return image, item['label']

def mini_imagenet_dataloader(split: str, image_size: int, batch_size: int = 32):
    # Load the dataset
    dataset = load_dataset("GATE-engine/mini_imagenet", split=split)

    # Print diagnostics
    print(f"\n{split.capitalize()} Dataset Diagnostics:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of batches: {len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)}")
    print(f"Size of first batch: {min(batch_size, len(dataset))}")
    print(f"Number of unique image labels: {len(set(dataset['label']))}")

    # Convert to MiniImageNetDataset
    dataset = MiniImageNetDataset(dataset, image_size)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

    return dataloader

def visualize_samples(dataloader, num_samples=5):
    # Get a batch of data
    images, labels = next(iter(dataloader))

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle("Sample Images from Mini-ImageNet Dataset")

    # Denormalize the images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean

    for i in range(num_samples):
        # Get the image and label
        img = images[i].permute(1, 2, 0).numpy()
        label = labels[i].item()

        # Display the image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Label: {label}")

    plt.tight_layout()
    plt.show()

# Usage Example
if False:
    # Create dataloaders for each split
    train_loader = mini_imagenet_dataloader('train', image_size=84, batch_size=32)
    val_loader = mini_imagenet_dataloader('validation', image_size=84, batch_size=32)
    test_loader = mini_imagenet_dataloader('test', image_size=84, batch_size=32)

    # Visualize samples from the train loader
    print("Visualizing samples from the train loader:")
    visualize_samples(train_loader)

    # Visualize samples from the validation loader
    print("Visualizing samples from the validation loader:")
    visualize_samples(val_loader)

    # Visualize samples from the test loader
    print("Visualizing samples from the test loader:")
    visualize_samples(test_loader)
