'''.

RESULTS:

- Don't use softmax in dot product attention, it disallows learning patterns
  other than "high match equals interesting". For instance, it cannot learn
  about 2 anticorrelated vectors. Also, without linear projections (which can
  cause overfitting), softmaxing similarity scores (0-1) will dilute the
  signal.

- If you use W_Q and W_K, it's likely to overfit training unless you overwhelm
  it with data

- Can learn to generalize with <50 training examples(!)

'''

import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

DEBUG = False


##################################################

def swap(x, swap1, swap2):
    ''' swap1 and swap2 are softmax vectors (think onehot) of rows of x that will
    be swapped. '''
    # Combine swap1 and swap2 into a single matrix
    P = torch.einsum('bx,by->bxy', swap1, swap2)
    P = P + P.transpose(1, 2)  # swap both directions
    # identity matrix to keep non-swapped data
    Id = torch.diag_embed(torch.ones_like(swap1) - swap1 - swap2)
    x_swapped = torch.einsum('bij,bjd->bid', P + Id, x)
    return x_swapped


def generate_data(num_samples, sequence_length, embedding_size):
    """Generate input sequences and target sequences for training a model.

    The function generates random input sequences of a specified length and
    embedding size.  It then generates random swap positions (swap1 and swap2)
    for each sequence and creates target sequences by swapping the elements at
    swap1 and swap2 positions. The elements at swap2 positions in the input
    sequences are replaced with similar-but-noised vectors to the elements at
    swap1 positions.

    Args:
        num_samples (int): The number of samples to generate.
        sequence_length (int): The length of each input sequence.
        embedding_size (int): The size of the embedding dimension.

    Returns:
        tuple: A tuple containing six tensors:
            - x (torch.Tensor): The input sequences of shape (num_samples, sequence_length, embedding_size).
            - y (torch.Tensor): The target sequences of shape (num_samples, sequence_length, embedding_size).
            - swap1 (torch.Tensor): The swap1 positions of shape (num_samples,).
            - swap2 (torch.Tensor): The swap2 positions of shape (num_samples,).
            - swap1_oh (torch.Tensor): The one-hot encoded swap1 positions of shape (num_samples, sequence_length).
            - swap2_oh (torch.Tensor): The one-hot encoded swap2 positions of shape (num_samples, sequence_length).

    Example:
        >>> num_samples = 100
        >>> sequence_length = 10
        >>> embedding_size = 5
        >>> x, y, swap1, swap2, swap1_oh, swap2_oh = generate_data(num_samples, sequence_length, embedding_size)
        >>> x.shape
        torch.Size([100, 10, 5])
        >>> y.shape
        torch.Size([100, 10, 5])
        >>> swap1_oh.shape
        torch.Size([100, 10])
        >>> swap2_oh.shape
        torch.Size([100, 10])

    """
    # Generate random input sequences
    # x shape: (num_samples, sequence_length, embedding_size)
    x = torch.randn(num_samples, sequence_length, embedding_size)

    # Generate random swap positions
    swap1 = torch.randint(0, sequence_length, (num_samples,))
    swap2 = torch.randint(0, sequence_length, (num_samples,))

    # Build x with 2 orthogonal vecs in it
    for i in range(num_samples):
        # Get the vector at swap1 position
        v1 = x[i, swap1[i]]

        # Generate a random noised vector of v1
        v2 = -v1 + torch.randn(embedding_size) * 1e-2

        # Set the orthogonal vector at swap2 position in the input sequence
        x[i, swap2[i]] = v2

    # Create target sequences by swapping elements at swap1 and swap2 positions
    y = x.clone()

    for i in range(num_samples):
        y[i, swap1[i]] = x[i, swap2[i]]
        y[i, swap2[i]] = x[i, swap1[i]]

    # Create one-hot encoded swap positions
    swap1_oh = torch.nn.functional.one_hot(swap1, num_classes=sequence_length)
    swap2_oh = torch.nn.functional.one_hot(swap2, num_classes=sequence_length)

    return x, y, swap1, swap2, swap1_oh, swap2_oh

def visualize_swap(x, y):
    """
    Visualize the indices of the input sequence and the target sequence.

    This function takes an input sequence `x` and its corresponding target sequence `y`,
    and prints the indices of the elements in `x` and `y`. The indices are printed in a
    format that allows for visual verification of the swap operation.

    Args:
        x (torch.Tensor): The input sequence of shape (sequence_length, embedding_size).
        y (torch.Tenso
    Returns:
        None

    Example:
        >>> x, y = generate_data(num_samples=1, sequence_length=5, embedding_size=3)
        >>> x_sample = x[0]
        >>> y_sample = y[0]
        >>> visualize_swap(x_sample, y_sample)
        Input sequence indices: [0, 1, 2, 3, 4]
        Target sequence indices: [0, 4, 2, 3, 1]
    """
    sequence_length, embedding_size = x.shape

    # Create an array of indices for the input sequence
    x_indices = torch.arange(sequence_length)

    # Compute cosine similarity between each element of y and all elements of x
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    cosine_similarity = torch.matmul(y_norm, x_norm.t())

    # Find the indices of the most similar elements in x for each element in y
    _, y_indices = torch.max(cosine_similarity, dim=1)

    # Print the indices of the input sequence and the target sequence
    print(x_indices.tolist())
    print(y_indices.tolist())

# # @@@@@@@@@@
# # Hand check generated data
# for i in range(10):
#     print('----------')
#     visualize_swap(x_test[i], y_test[i])
# print(x_train[0, :, :5])
# print(y_train[0, :, :5])

# # Test that the expected swaps took place
# for i in range(num_test_samples):
#     assert torch.allclose(x_test[i, swap1_test[i]], y_test[i, swap2_test[i]]), f"Swap mismatch at sample {i}"
#     assert torch.allclose(x_test[i, swap2_test[i]], y_test[i, swap1_test[i]]), f"Swap mismatch at sample {i}"
#     cs1 = torch.cosine_similarity(x_test[i, swap1_test[i]], y_test[i, swap2_test[i]], dim=0) # orthogonal
#     cs2 = torch.cosine_similarity(x_test[i, 0], y_test[i, 1], dim=0) # random similarity

# # Test that swap1_oh_test and swap2_oh_test are one-hot encoded correctly
# assert torch.all(swap1_oh_test.sum(dim=1) == 1), "swap1_oh_test is not one-hot encoded"
# assert torch.all(swap2_oh_test.sum(dim=1) == 1), "swap2_oh_test is not one-hot encoded"

# # Test that swap1_oh_test and swap2_oh_test match swap1_test and swap2_test
# assert torch.all(swap1_oh_test.argmax(dim=1) == swap1_test), "swap1_oh_test does not match swap1_test"
# assert torch.all(swap2_oh_test.argmax(dim=1) == swap2_test), "swap2_oh_test does not match swap2_test"

# print("All assertions passed!")

# Q = x_train[0]
# K = x_train[0]
# Id = torch.eye(10, 10)
# swap1 = swap1_train[0]
# swap2 = swap2_train[0]
# # cs = torch.einsum('sd, td -> st', F.normalize(Q, dim=1), F.normalize(K, dim=1))
# cs = torch.einsum('sd, td -> st', Q, K)
# # plt.imshow(1 - (cs).abs())
# plt.imshow(cs)
# plt.show()

# BRK
# # @@@@@@@@@@


##################################################



class SwapModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads=2):
        super(SwapModel, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Dimensions for multi-head attention
        self.head_dim = embedding_size // num_heads
        assert embedding_size % num_heads == 0, "embedding_size must be divisible by num_heads"

        # Linear layers for query, key, value
        self.query_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.key_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value_linear = nn.Linear(embedding_size, embedding_size, bias=False)

        # Learnable parameters for value (V) vectors in cross-attention
        MAX_VOCAB = 100
        self.V_DIM = embedding_size
        # self.value_embeddings = nn.Parameter(torch.randn(self.N_EMBEDDINGS, self.V_DIM))
        self.value_embeddings = torch.randn(MAX_VOCAB, self.V_DIM)

        # # Output linear layer
        # self.out = nn.Linear(self.V_DIM, 2)  # 2, for swap1 and swap2

        H = 32
        self.out = nn.Sequential(
            nn.Linear(self.V_DIM, H),
            nn.GELU(),
            nn.Linear(H, 2)  # 2, for swap1 and swap2
        )

    # def scaled_dot_product_attention(self, Q, K, V):
    #     scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
    #     attn_weights = torch.softmax(scores, dim=-1)  # [B, HEAD, S, S]
    #     # attn_weights = scores
    #     attn_output = torch.matmul(attn_weights, V)
    #     return attn_output

    def scaled_dot_product_attention(self, Q, K, V, attn_nonlin):
        scores = torch.einsum('bhsd, bhtd -> bhst', Q, K) / (self.head_dim ** 0.5)

        # # remove self-similarity
        # B, NH, S, HD = Q.size()
        # Id = torch.eye(S, S).unsqueeze(0).unsqueeze(0).expand(B, NH, S, S)
        # scores = scores * (1 - Id)
        # # breakpoint()

        match attn_nonlin:
            case 'softmax':
                attn_weights = torch.softmax(scores, dim=3)  # [B, HEAD, S, S]
            case 'none':
                attn_weights = scores
            case 'sigmoid':
                attn_weights = scores.sigmoid()
            case 'tanh':
                attn_weights = scores.tanh()

        attn_output = torch.einsum('bhst, bhtd -> bhsd', attn_weights, V)

        if DEBUG:
            breakpoint()

        return attn_output

    def forward(self, x, attn_nonlin, use_wq, use_wk, use_wv, use_symbols):
        B, S, D = x.size()
        Q = self.query_linear(x) if use_wq else x
        K = self.key_linear(x) if use_wk else x


        # Reshape Q, K, V for multi-head attention
        Q = torch.einsum('bshd -> bhsd', Q.view(B, S, self.num_heads, self.head_dim))
        K = torch.einsum('bshd -> bhsd', K.view(B, S, self.num_heads, self.head_dim))

        # Expand the value embeddings to match the batch size and apply linear projection
        if use_symbols:
            V = self.value_embeddings[:S].unsqueeze(0).expand(B, -1, -1)  # (B, S, D)
        else:
            V = self.value_linear(x)
            # V = torch.einsum('bshd -> bhsd', K.view(B, S, self.num_heads, self.head_dim))
        V = torch.einsum('bshd -> bhsd', V.view(B, S, self.num_heads, self.V_DIM // self.num_heads))

        attn_output = self.scaled_dot_product_attention(Q, K, V, attn_nonlin)  # (B, num_heads, S, head_dim)

        # Concatenate heads
        attn_output = torch.einsum('bhsd -> bsdh', attn_output).contiguous().view(B, S, self.V_DIM)  # (B, S, V_DIM)
        o = self.out(attn_output)

        swap1 = torch.softmax(o[:, :, 0], dim=1)
        swap2 = torch.softmax(o[:, :, 1], dim=1)
        y_pred = swap(x, swap1, swap2)

        return y_pred

def show(attn_weights, batch_ix):
    head = 0
    with torch.no_grad():
        plt.imshow(attn_weights[batch_ix, head])
        plt.show()

def accuracy(y_pred, y, threshold=0.7):
    B, S, D = y_pred.size()

    # Normalize predictions and targets
    y_pred_norm = F.normalize(y_pred, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    # Compute cosine similarity between predictions and targets
    cosine_sim = torch.einsum('bsd, bsd -> bs', y_pred_norm, y_norm)

    # Check if cosine similarity is above the threshold
    correct = (cosine_sim > threshold).float()

    # Compute accuracy
    acc = torch.mean(correct)

    return acc


def run(model, optimizer,
        x_train, y_train, swap1_train, swap2_train,
        x_test, y_test, swap1_test, swap2_test,
        num_epochs, fwd_params,
        test_mode=False):
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []

    if not test_mode:
        for epoch in range(num_epochs):
            y_pred_train = model(x_train, **fwd_params)

            # Loss: Cosine Distance
            train_loss = torch.einsum('bsd, bsd -> bs',
                                      F.normalize(y_pred_train, dim=2),
                                      F.normalize(y_train, dim=2))
            train_loss = 1 - train_loss.mean()

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                # test_acc and test_loss
                model.eval()
                with torch.no_grad():
                    y_pred_test = model(x_test, **fwd_params)

                    # train check
                    i = torch.arange(x_train.size(0))
                    ii = torch.stack([i, i], dim=1)
                    ss = torch.stack([swap1_train, swap1_train], dim=1)
                    y_pred_train_check = y_pred_train[ii, ss]
                    y_train_check = y_train[ii, ss]

                    # test check
                    i = torch.arange(x_test.size(0))
                    ii = torch.stack([i, i], dim=1)
                    ss = torch.stack([swap1_test, swap1_test], dim=1)
                    y_pred_test_check = y_pred_test[ii, ss]
                    y_test_check = y_test[ii, ss]

                    # train_acc
                    train_acc = accuracy(y_pred_train_check, y_train_check)

                    # train_ix = torch.arange
                    test_acc = accuracy(y_pred_test_check, y_test_check)
                    test_loss = torch.einsum('bsd, bsd -> bs',
                                             F.normalize(y_pred_test, dim=2),
                                             F.normalize(y_test, dim=2))
                    test_loss = 1 - test_loss.mean()

                print(f"Epoch [{epoch+1:>3d}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Acc: {test_acc.item():.4f}")

                train_accs.append(train_acc.item())
                test_accs.append(test_acc.item())
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())

    else:
        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test, **fwd_params)
            test_acc = accuracy(y_pred_test, y_test)
            test_loss = torch.einsum('bsd, bsd -> bs',
                                     F.normalize(y_pred_test, dim=2),
                                     F.normalize(y_test, dim=2))
            test_loss = 1 - test_loss.mean()
            print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_acc.item():.4f}")

    return train_accs, test_accs, train_losses, test_losses


num_repetitions = 10  # number of times to repeat and average a result

num_test_samples = 200
sequence_length = 10

embedding_size = 256
hidden_size = 128
num_heads = 2

learning_rate = 1e-3
num_epochs = 100

train_sizes = [20, 50, 100, 500, 1000, 2000]
architectures = [
    {"name": "Transformer", "fwd_params": {"use_symbols": False, "attn_nonlin": "softmax", "use_wq": True, "use_wk": True, "use_wv": True}},
    # {"name": "Neurallambda (softmax)", "fwd_params": {"use_symbols": True, "attn_nonlin": "softmax", "use_wq": False, "use_wk": False, "use_wv": False}},
    {"name": "Neurallambda", "fwd_params": {"use_symbols": True, "attn_nonlin": "none", "use_wq": False, "use_wk": False, "use_wv": False}},
]

# Collect results for each training size and architecture
results = {}
for train_size in train_sizes:
    for arch in architectures:
        name = arch['name']
        fwd_params = arch['fwd_params']
        print(f"Running test for {name} with {train_size} training samples")

        # Initialize lists to store results for each repetition
        train_accs_list = []
        test_accs_list = []
        train_losses_list = []
        test_losses_list = []

        for _ in range(num_repetitions):
            # Generate train and test data
            x_train, y_train, swap1_train, swap2_train, swap1_oh_train, swap2_oh_train = generate_data(train_size, sequence_length, embedding_size)
            x_test, y_test, swap1_test, swap2_test, swap1_oh_test, swap2_oh_test = generate_data(num_test_samples, sequence_length, embedding_size)

            # Create model and optimizer
            model = SwapModel(
                embedding_size,
                hidden_size,
                num_heads
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Run the model
            train_accs, test_accs, train_losses, test_losses = run(model, optimizer,
                                                                   x_train, y_train, swap1_train, swap2_train,
                                                                   x_test, y_test, swap1_test, swap2_test,
                                                                   num_epochs, fwd_params)

            # Append the results for this repetition
            train_accs_list.append(train_accs[-1])
            test_accs_list.append(test_accs[-1])
            train_losses_list.append(train_losses[-1])
            test_losses_list.append(test_losses[-1])

        # Calculate the average results over all repetitions
        avg_train_acc = sum(train_accs_list) / num_repetitions
        avg_test_acc = sum(test_accs_list) / num_repetitions
        avg_train_loss = sum(train_losses_list) / num_repetitions
        avg_test_loss = sum(test_losses_list) / num_repetitions

        # Store the averaged results
        if name not in results:
            results[name] = {"train_sizes": [], "train_accs": [], "test_accs": [], "train_losses": [], "test_losses": []}
        results[name]["train_sizes"].append(train_size)
        results[name]["train_accs"].append(avg_train_acc)
        results[name]["test_accs"].append(avg_test_acc)
        results[name]["train_losses"].append(avg_train_loss)
        results[name]["test_losses"].append(avg_test_loss)



##########



# Define nicer colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot accuracy comparison
for i, name in enumerate(results):
    color = colors[i % len(colors)]
    ax1.plot(results[name]["train_sizes"], results[name]["train_accs"], marker='o', linestyle='--', color=color)
    ax1.plot(results[name]["train_sizes"], results[name]["test_accs"], marker='o', linestyle='-', color=color, label=name)
ax1.set_xlabel("Total Training Samples")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.set_title("Accuracy Comparison")

# Plot loss comparison
for i, name in enumerate(results):
    color = colors[i % len(colors)]
    ax2.plot(results[name]["train_sizes"], results[name]["train_losses"], marker='o', linestyle='--', color=color)
    ax2.plot(results[name]["train_sizes"], results[name]["test_losses"], marker='o', linestyle='-', color=color, label=name)
ax2.set_xlabel("Total Training Samples")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.set_title("Loss Comparison")

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()



##################################################
##################################################
# GRAVEYARD
##################################################
##################################################

##########
# tests

if False:
    import torch
    import torch.nn.functional as F

    def swap(x, swap1, swap2, variant=3, sk_iterations=0):
        """
        Swap rows in a tensor according to the specified swap vectors.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq, dim].
            swap1 (torch.Tensor): First swap vector of shape [batch, seq].
            swap2 (torch.Tensor): Second swap vector of shape [batch, seq].
            variant (int): Variant of the swapping operation (0, 1, 2, or 3).

        Returns:
            torch.Tensor: Swapped tensor of shape [batch, seq, dim].
        """
        B, S, D = x.shape
        if variant == 0:
            # handle non-swaps by zeroing out the swap vectors if they overlap, by
            # subtraction then relu
            swap1, swap2 = (swap1 - swap2).relu(), (swap2 - swap1).relu()
            overlap = swap1 * swap2
            swap1 = swap1 * (1 - overlap)
            swap2 = swap2 * (1 - overlap)
            # Compute the swapped tensor
            x_swapped = (
                x * (1 - swap1).unsqueeze(2) * (1 - swap2).unsqueeze(2) +  # unswapped data
                torch.einsum('ba, bd -> bad', swap1, torch.einsum('ba, bad -> bd', swap2, x)) +
                torch.einsum('ba, bd -> bad', swap2, torch.einsum('ba, bad -> bd', swap1, x))
            )
            return x_swapped
        elif variant == 1:
            # handle non-swaps by zeroing out the swap vectors if they overlap by
            # multiplying by 1 except in the case of overlap
            swap1, swap2 = swap1 * (1 - swap2), swap2 * (1 - swap1)
            overlap = swap1 * swap2
            swap1 = swap1 * (1 - overlap)
            swap2 = swap2 * (1 - overlap)
            # Compute the swapped tensor
            x_swapped = (
                x * (1 - swap1).unsqueeze(2) * (1 - swap2).unsqueeze(2) +  # unswapped data
                torch.einsum('ba, bd -> bad', swap1, torch.einsum('ba, bad -> bd', swap2, x)) +
                torch.einsum('ba, bd -> bad', swap2, torch.einsum('ba, bad -> bd', swap1, x))
            )
            return x_swapped
        elif variant == 2:
            # handle non-swaps by zeroing out the swap vectors if they overlap by
            # multiplying by 1 except in the case of overlap
            overlap = swap1 * swap2
            swap1 = swap1 * (1 - overlap)
            swap2 = swap2 * (1 - overlap)
            # Compute the swapped tensor
            x_swapped = (
                x * (1 - swap1).unsqueeze(2) * (1 - swap2).unsqueeze(2) +  # unswapped data
                torch.einsum('ba, bd -> bad', swap1, torch.einsum('ba, bad -> bd', swap2, x)) +
                torch.einsum('ba, bd -> bad', swap2, torch.einsum('ba, bad -> bd', swap1, x))
            )
            return x_swapped
        elif variant == 3:
            # handle non-swaps by zeroing out the swap vectors if they overlap by
            # explicitly subtracting the case where no-swap happened so it doesn't
            # double in magnitude.
            same_swap_prob = torch.einsum('ba, ba -> ba', swap1, swap2)  # [batch]

            # Compute the swapped tensor
            x_swapped = (
                x * (1 - swap1).unsqueeze(2) * (1 - swap2).unsqueeze(2) +  # unswapped data
                torch.einsum('ba, bd -> bad', swap1, torch.einsum('ba, bad -> bd', swap2, x)) +
                torch.einsum('ba, bd -> bad', swap2, torch.einsum('ba, bad -> bd', swap1, x)) -

                # explicitly subtract for the case where it's not actually swapping
                torch.einsum('ba, bd -> bad', same_swap_prob, torch.einsum('ba, bad -> bd', swap1, x))
            )
            return x_swapped
        elif variant == 4:
            # don't handle no-swaps, just normalize so if it doubled, well that's
            # ok. Expects inputs to be normalized.
            x_swapped = (
                x * (1 - swap1).unsqueeze(2) * (1 - swap2).unsqueeze(2) +  # unswapped data
                torch.einsum('ba, bd -> bad', swap1, torch.einsum('ba, bad -> bd', swap2, x)) +
                torch.einsum('ba, bd -> bad', swap2, torch.einsum('ba, bad -> bd', swap1, x))
            )
            return F.normalize(x_swapped, dim=2)

        elif variant == 'sinkhorn-knopp':
            # permutation matrices version

            # Combine swap1 and swap2 into a single matrix
            P = torch.einsum('bx,by->bxy', swap1, swap2)

            # Apply Sinkhorn-Knopp algorithm to make P a doubly stochastic matrix
            for _ in range(sk_iterations):
                P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)  # Normalize rows
                P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)  # Normalize columns

            # Create an identity matrix for the original data
            I = torch.eye(S, device=x.device).unsqueeze(0).repeat(B, 1, 1)

            # Interpolate between the identity matrix and the soft permutation matrix
            # combined_matrix = I + P - 2 * I * P
            combined_matrix = (
                1 -
                swap1.unsqueeze(1).expand(B, S, S) -
                swap2.unsqueeze(2).expand(B, S, S)
            ) * I + P

            add_back_no_swap = swap1.unsqueeze(1).expand(B, S, S) * swap2.unsqueeze(2).expand(B, S, S)
            combined_matrix = combined_matrix + add_back_no_swap

            # Swap elements using the combined matrix
            x_swapped = torch.einsum('bij,bjd->bid', combined_matrix, x)

            return x_swapped

        elif variant == 'sinkhorn-knopp-2':
            # permutation matrices version

            # Combine swap1 and swap2 into a single matrix
            P = torch.einsum('bx,by->bxy', swap1, swap2)
            P = P + P.transpose(1, 2)

            # Apply Sinkhorn-Knopp algorithm to make P a doubly stochastic matrix
            for _ in range(sk_iterations):
                P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)  # Normalize rows
                P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)  # Normalize columns

            # Create an identity matrix to keep non-swapped data
            I = torch.diag_embed((1 - swap1) * (1 - swap2))
            x_swapped = torch.einsum('bij,bjd->bid', P + I, x)
            return F.normalize(x_swapped, dim=2)

        elif variant == 'sinkhorn-knopp-3':
            # permutation matrices version

            # Combine swap1 and swap2 into a single matrix
            P = torch.einsum('bx,by->bxy', swap1, swap2)
            P = P + P.transpose(1, 2)

            # Apply Sinkhorn-Knopp algorithm to make P a doubly stochastic matrix
            for _ in range(sk_iterations):
                P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)  # Normalize rows
                P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)  # Normalize columns

            # Create an identity matrix to keep non-swapped data
            I = torch.diag_embed(torch.ones_like(swap1) - swap1 - swap2)
            x_swapped = torch.einsum('bij,bjd->bid', P + I, x)
            return x_swapped


    def swap(x, swap1, swap2):
        # Combine swap1 and swap2 into a single matrix
        P = torch.einsum('bx,by->bxy', swap1, swap2)
        P = P + P.transpose(1, 2)  # swap both directions
        # identity matrix to keep non-swapped data
        I = torch.diag_embed(torch.ones_like(swap1) - swap1 - swap2)
        x_swapped = torch.einsum('bij,bjd->bid', P + I, x)
        return x_swapped


    torch.manual_seed(42)

    # Create a random tensor
    x = torch.randn(3 * 7 * 128).reshape(3, 7, 128)  # [batch, seq, dim]
    x = F.normalize(x, dim=2)

    # Create swap1 and swap2 as one-hot vectors
    i1 = torch.randint(0, 7, (3,))
    i2 = torch.randint(0, 7, (3,))

    swap1 = torch.eye(7)[i1]
    swap2 = torch.eye(7)[i2]

    # Swap rows using the specified variant
    SWAP_OPTS = [
        1, 2, 3, 4,
        'sinkhorn-knopp',
        'sinkhorn-knopp-2',
        'sinkhorn-knopp-3',
    ]

    for _ in range(100):
        for SWAP_OPT in SWAP_OPTS:
            # x_swapped = swap(x, swap1, swap2, variant=SWAP_OPT)
            x_swapped = swap(x, swap1, swap2)

            # Create the expected output tensor
            expected_output = x.clone()
            for i in range(3):
                swap1_idx = torch.argmax(swap1[i])
                swap2_idx = torch.argmax(swap2[i])
                expected_output[i, [swap1_idx, swap2_idx]] = expected_output[i, [swap2_idx, swap1_idx]]

            # Assert that the swapped tensor matches the expected output
            try:
                assert torch.allclose(x_swapped, expected_output), "Swapped tensor does not match the expected output"
                print(f'checked: {SWAP_OPT}')
            except:
                print(x_swapped - expected_output)
                print(f'{(x_swapped - expected_output).max()=}')
                print(f'error on {SWAP_OPT}')
                break



    # END_BLOCK_3
