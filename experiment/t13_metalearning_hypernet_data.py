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
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from torchvision.transforms import Resize

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
        self.resize = Resize((image_size, image_size))

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


def omniglot_dataloader(train_alphabets: List[str], test_alphabets: List[str], image_size: int, batch_size: int = 32):
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

    # Create DataLoaders
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
