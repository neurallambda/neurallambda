import torch
from neurallambda.symbol import *


##################################################
# SymbolMapper

def test_round_trip():
    sym_mapper = SymbolMapper(256)
    for v in sym_mapper.symbols_i2v.values():
        rv = sym_mapper.unproject(sym_mapper.project(v))
        assert v == rv, f'Value {v} round-tripped to {rv}'


##################################################
# IntMapper

def test_binary_conversion():
    for i in range(-128, 128):
        binary_vector = int_to_binary_vector(i, 8)
        recovered_int = binary_vector_to_int(binary_vector)
        assert recovered_int == i, f"Failed to recover original integer. Expected {i}, got {recovered_int}"

def test_projection_and_unprojection():
    int_mapper = IntMapper(8, 20)
    for i in range(-128, 128):
        projected = int_mapper.project(i)
        unprojected = int_mapper.unproject(projected)
        assert unprojected == i, f"Failed to unproject integer. Expected {i}, got {unprojected}"

def test_projection_with_noise():
    int_mapper = IntMapper(8, 20)
    for i in range(-16, 16):
        projected = int_mapper.project(i)
        noise = torch.randn_like(projected) * 1e-2
        noisy_projected = projected + noise

        # Attempt to unproject the noisy vector
        unprojected = int_mapper.unproject(noisy_projected)
        assert i == unprojected, f'{i} != {unprojected}'


##########
# Vis Projections

if False:

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Define the entire library with minor adjustments for testing
    BIT_STRING_SIZE = 8  # Length of the binary representation
    N_PROJECTIONS = 10    # Number of projection vectors

    int_mapper = IntMapper(BIT_STRING_SIZE, N_PROJECTIONS)
    plt.imshow(int_mapper.projection_matrix)
    plt.show()

    nums = []
    for i in range(-32, 32):
        nums.append(int_mapper.project(i))
    nums = np.stack(nums)
    plt.imshow(nums)
    plt.show()


    ##########
    # Vis Unprojections


    # Constants
    NOISE_LEVEL = 2.0
    num_cycles = 100  # Number of cycles to average for each tested number
    int_test_range_start = -64
    int_test_range_end = 64

    int_confusion_range_start = int_test_range_start * 4
    int_confusion_range_end = int_test_range_end * 4

    confusion_matrix = np.zeros((int_test_range_end - int_test_range_start, int_confusion_range_end - int_confusion_range_start), dtype=np.float32)

    for cycle in range(num_cycles):
        for i in range(int_test_range_start, int_test_range_end):
            projected = int_mapper.project(i)
            noise = torch.randn_like(projected) * NOISE_LEVEL
            noisy_projected = projected + noise
            unprojected = int_mapper.unproject(noisy_projected)

            # Mapping integers back to the index range 0 to int_confusion_range-1
            original_index = i - int_test_range_start
            unprojected_index = unprojected - int_confusion_range_start
            unprojected_clamped = np.clip(unprojected_index, 0, int_confusion_range_end - int_confusion_range_start - 1)
            confusion_matrix[original_index, unprojected_clamped] += 1

    # Average the confusion matrix
    confusion_matrix /= num_cycles

    # Visualization of the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.viridis)
    fig.colorbar(cax)

    ax.set_xlabel('Unprojected Integer')
    ax.set_ylabel('Original Integer')
    ax.set_title(f'Averaged Confusion Matrix of Unprojections with Noise Level: {NOISE_LEVEL} over {num_cycles} cycles')
    plt.show()
