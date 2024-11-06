'''

DEPRECATED, use github.com/neurallambda/arc-like instead, at `t01_data_arc_like`

Synthetic Data representing visual puzzles in one dimension, such as blocks moving around, or denoising.

'''

import random
from random import randrange, choice, randint, shuffle, sample
from typing import Callable, Dict

import torch
from torch.utils.data import TensorDataset

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


##################################################
# Random Identity

def gen_random_identity(seq_len, max_digits=10, background=0):
    """
    Generate a sequence of random colors and output the same sequence.

    Args:
    seq_len (int): Length of the sequence to generate.
    max_digits (int): Number of different colors to use (excluding background).
    background (int): Background color value.

    Returns:
    tuple: Two identical lists representing the input and output sequences.
    """
    # Create a list of colors excluding the background color
    colors = [i for i in range(1, max_digits + 1) if i != background]

    # Generate the sequence
    sequence = [background] * seq_len

    for i in range(seq_len):
        if random.random() < 0.5:  # 50% chance to change color
            sequence[i] = random.choice(colors)

    return sequence, sequence.copy()


##################################################
# Identity

def gen_identity_block(seq_len, min_len=2, max_len=5, max_digits=10, background=0):
    """
    Generate a single block for the identity task where input and output are the same
    """
    block_len = random.randint(min_len, max_len)
    block_start = random.randint(0, seq_len - block_len)
    block_end = block_start + block_len

    block_color = random.randint(1, max_digits)
    while block_color == background:
        block_color = random.randint(1, max_digits)

    sequence = [background] * seq_len

    for i in range(block_start, block_end):
        sequence[i] = block_color

    return sequence, sequence.copy()


def divide_into_chunks(n, denom):
    '''Divide `n` into `denom` number of chunks of integers, such that all chunks
    sum to `n`. This is useful when you want to divide a sequence into `denom`
    blocks that are roughly equal size and all add to `n`.'''
    base_chunk = n // denom
    remainder = n % denom
    chunks = [base_chunk] * denom
    for i in range(remainder):
        chunks[i] += 1
    return chunks

def test_divide_into_chunks():
    import random
    for _ in range(100):
        n = random.randint(1, 1000)
        denom = random.randint(1, n)
        chunks = divide_into_chunks(n, denom)
        assert sum(chunks) == n, f"Sum of chunks {sum(chunks)} does not equal {n}"
        assert len(chunks) == denom, f"Number of chunks {len(chunks)} does not equal {denom}"
        assert all(isinstance(chunk, int) for chunk in chunks), "Not all chunks are integers"
# test_divide_into_chunks()


def gen_multi_identity(seq_len, num_blocks=3, min_len=2, max_len=5, max_digits=10, background=0):
    """
    Generate multiple identity blocks
    """
    seq_lens = divide_into_chunks(seq_len, num_blocks)
    input_seq = []
    output_seq = []

    for s in seq_lens:
        ii, oo = gen_identity_block(s, min_len, max_len, max_digits, background)
        input_seq += ii
        output_seq += oo

    return input_seq, output_seq


##################################################
# Denoising

def gen_single_1c_denoising(seq_len, max_digits, background):
    """
    1D same color denoising, single object, noise is outside block
    """
    obj_len = randrange(seq_len // 3, seq_len // 2)
    obj_start = randrange(seq_len - obj_len - 2)
    obj_end = obj_start + obj_len

    filling_color = randrange(1, max_digits)

    input = [background] * seq_len
    output = [background] * seq_len

    for i in range(obj_start, obj_end):
        input[i] = filling_color
        output[i] = filling_color

    pixel_suffix = 1

    # prepending noise before block
    pixel_end = 0
    for i in range(randrange(1, 6)):
        pixel_len = 1
        pixel_prefix = randrange(2, 5)

        pixel_start = pixel_end + pixel_prefix
        pixel_end = pixel_start + pixel_len

        if pixel_end + pixel_suffix > obj_start:
            break

        for i in range(pixel_start, pixel_end):
            input[i] = filling_color

    # appending noises after block
    pixel_end = obj_end
    for i in range(randrange(1, 6)):
        pixel_len = 1
        pixel_prefix = randrange(2, 5)

        pixel_start = pixel_end + pixel_prefix
        pixel_end = pixel_start + pixel_len

        if pixel_end + pixel_suffix > seq_len:
            break

        for i in range(pixel_start, pixel_end):
            input[i] = filling_color

    return input, output


##################################################
# Fill

def gen_basic_fill(seq_len, min_len=8, max_len=32, min_hole_len=1, max_digits=10, background=0):
    """
    Generate a basic fill task: fill a single hole in 1D
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    hole_len = randint(min_hole_len, seq_len - 2 - 3)
    hole_start = randint(0, seq_len - hole_len - 2)
    hole_end = hole_start + hole_len + 1

    pivot_pt = randint(1, max_digits)
    filling_color = pivot_pt

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    input_seq[hole_start] = pivot_pt
    input_seq[hole_end] = pivot_pt

    output_seq[hole_start] = pivot_pt
    for i in range(hole_start + 1, hole_end):
        output_seq[i] = filling_color
    output_seq[hole_end] = pivot_pt

    return input_seq, output_seq

def gen_multi_fill(seq_len, min_len=8, max_len=32, min_hole_len=1, max_digits=10, num_blocks=3, background=0):
    """
    3 basic fills of different colors simultaneously
    """

    input_seq = []
    output_seq = []
    seq_lens = divide_into_chunks(seq_len, num_blocks)
    for s in seq_lens:
        ii, oo = gen_basic_fill(s, min_len, max_len, min_hole_len, max_digits, background)
        input_seq += ii
        output_seq += oo

    return input_seq, output_seq

def gen_hollow(seq_len, min_len=8, max_len=32, min_hole_len=1, max_digits=10, background=0):
    """
    Opposite of fill task, 1 block must be hollowed.
    """
    output_seq, input_seq = gen_basic_fill(seq_len, min_len, max_len, min_hole_len, max_digits, background)
    return input_seq, output_seq


##################################################
# Flip

def gen_single_flip(seq_len, min_len=8, max_len=32, max_digits=10, background=0):
    """
    A block with an indicator at start that must be flipped to the end.
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    obj_len = randint(seq_len // 4, seq_len // 2)
    obj_start = randint(0, seq_len - obj_len - 1)
    obj_end = obj_start + obj_len + 1

    pivot_pt = randint(1, max_digits)
    filling_color = randint(1, max_digits)
    if filling_color == pivot_pt:
        filling_color = pivot_pt + 1 if pivot_pt < max_digits else pivot_pt - 1

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    for i in range(obj_start, obj_end):
        if i == obj_start:
            input_seq[i] = pivot_pt
        else:
            input_seq[i] = filling_color

        if i == obj_end - 1:
            output_seq[i] = pivot_pt
        else:
            output_seq[i] = filling_color

    return input_seq, output_seq


##################################################
# Multicolor denoising

def gen_single_mc_denoising(seq_len, n_noise_pixels=4, min_len=32, max_len=33, max_digits=10, background=0):
    """
    A single block with noise pixels within it
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    obj_len = randint(seq_len - 12, seq_len - 6)
    obj_start = randint(0, seq_len - obj_len - 2)
    obj_end = obj_start + obj_len

    filling_color = randint(1, max_digits)
    noise_colors = list(range(1, max_digits + 1))
    noise_colors.remove(filling_color)

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    for i in range(obj_start, obj_end):
        input_seq[i] = filling_color
        output_seq[i] = filling_color

    for _ in range(randint(1, n_noise_pixels)):
        noise_color = choice(noise_colors)
        noise_position = randint(obj_start + 2, obj_end - 3)
        input_seq[noise_position] = noise_color

    return input_seq, output_seq


##################################################
# Mirror

def gen_single_mirror(seq_len, min_len=12, max_len=32, max_digits=10, background=0, pivot_pt=9):
    """
    A block starts left of a "mirror" pixel, and must be reflected across it
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    obj_len = randint(seq_len // 4, seq_len // 3)
    obj_to_pivot = randint(1, max(2, ((seq_len - obj_len * 2 - 3) // 2)))

    obj1_start = randint(0, seq_len - obj_len * 2 - obj_to_pivot * 2 - 1)
    obj1_end = obj1_start + obj_len

    obj2_start = obj1_end + obj_to_pivot * 2 + 1
    obj2_end = obj2_start + obj_len

    filling_color = randint(1, max_digits)

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    for i in range(obj1_start, obj1_end):
        input_seq[i] = filling_color

    for i in range(obj2_start, obj2_end):
        output_seq[i] = filling_color

    pivot_position = obj1_end + obj_to_pivot
    input_seq[pivot_position] = pivot_pt
    output_seq[pivot_position] = pivot_pt

    return input_seq, output_seq


##################################################
# Movement

def gen_move_single_bar(seq_len, min_len=8, max_len=32, move_len=3, min_bar_len=3, max_digits=10, background=0):
    """
    Move a single bar rightward for a fixed move_len
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    bar_len = randint(min_bar_len, seq_len - move_len)
    bar_start = randint(0, seq_len - bar_len - move_len)

    filling_color = randint(1, max_digits)

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    for i in range(bar_start, bar_start + bar_len):
        input_seq[i] = filling_color

    for j in range(bar_start + move_len, bar_start + bar_len + move_len):
        output_seq[j] = filling_color

    return input_seq, output_seq

def gen_move_to_pixel(seq_len, min_len=8, max_len=32, min_bar_len=3, max_digits=10, background=0):
    """
    Move a single bar rightward to touch a dynamically located indicator
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    bar_len = randint(min_bar_len, seq_len // 2)
    bar_start = randint(0, seq_len - bar_len - 3)
    bar_end = bar_start + bar_len

    filling_color = randint(1, max_digits)
    pivot_color = choice([c for c in range(1, max_digits + 1) if c != filling_color])

    pivot_pos = randint(bar_end + 2, seq_len - 1)

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    for i in range(bar_start, bar_end):
        input_seq[i] = filling_color

    input_seq[pivot_pos] = pivot_color
    output_seq[pivot_pos] = pivot_color

    for i in range(pivot_pos - bar_len, pivot_pos):
        output_seq[i] = filling_color

    return input_seq, output_seq


def gen_move_towards_indicator(seq_len, min_len=8, max_len=32, max_move=2, min_bar_len=3, max_digits=10, background=0):
    """
    Move a block towards an indicator by up to 2 positions, possibly covering the indicator.
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    # Generate the block
    bar_len = randint(min_bar_len, seq_len // 3)
    bar_start = randint(0, seq_len - bar_len - 3)
    bar_end = bar_start + bar_len

    # Generate the indicator
    indicator_pos = randint(bar_end + 2, seq_len - 1)

    # Choose colors
    block_color = randint(1, max_digits)
    indicator_color = choice([c for c in range(1, max_digits + 1) if c != block_color])

    # Create input sequence
    input_seq = [background] * seq_len
    for i in range(bar_start, bar_end):
        input_seq[i] = block_color
    input_seq[indicator_pos] = indicator_color

    # Create output sequence
    output_seq = input_seq.copy()

    # Calculate move distance
    distance = indicator_pos - bar_end
    move_distance = min(distance, max_move)

    # Move the block
    for i in range(bar_start, bar_end):
        output_seq[i] = background
    for i in range(bar_start + move_distance, bar_end + move_distance):
        output_seq[i] = block_color

    return input_seq, output_seq


def gen_extend_to_pixel(seq_len, min_len=8, max_len=32, min_bar_len=3, max_digits=10, background=0):
    """
    Extend a block all the way rightward to a dynamically located pixel.
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    bar_len = randint(min_bar_len, seq_len // 2)
    bar_start = randint(0, seq_len - bar_len - 3)
    bar_end = bar_start + bar_len

    filling_color = randint(1, max_digits)
    pivot_color = choice([c for c in range(1, max_digits + 1) if c != filling_color])

    pivot_pos = randint(bar_end + 2, seq_len - 1)

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    for i in range(bar_start, bar_end):
        input_seq[i] = filling_color

    input_seq[pivot_pos] = pivot_color
    output_seq[pivot_pos] = pivot_color

    for i in range(bar_start, pivot_pos):
        output_seq[i] = filling_color

    return input_seq, output_seq


##################################################
# Copy pattern

def gen_grow(seq_len, min_len=32, max_len=33, max_digits=10, background=0):
    """
    Generate randomly spaced seed pixels and grow each to 3 pixels of the same color,
    ensuring a minimum spacing of 5 between seeds
    """
    seq_len = max(min(seq_len, max_len), min_len)
    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    seed_positions = []
    possible_positions = list(range(1, seq_len - 1))

    while possible_positions:
        pos = choice(possible_positions)
        if all(abs(pos - existing_pos) >= 5 for existing_pos in seed_positions):
            seed_positions.append(pos)
            filling_color = random.randint(1, max_digits)
            input_seq[pos] = filling_color
            for j in range(3):
                output_seq[pos + j - 1] = filling_color

            # rm if too close to the seed
            possible_positions = [p for p in possible_positions if abs(p - pos) >= 5]
        else:
            possible_positions.remove(pos)

    return input_seq, output_seq

def gen_grow_copy_color(seq_len, min_len=32, max_len=33, max_digits=10, background=0):
    """
    Generate randomly spaced seed pixels, copy each pixel 3 times, and recolor to the color of the leftmost bar
    """
    seq_len = max(min(seq_len, max_len), min_len)
    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    filling_color = random.randint(1, max_digits)
    init_color = choice([c for c in range(1, max_digits + 1) if c != filling_color and c != background])

    # Color indicator bar
    offset = randint(1, 2)
    for j in range(offset, 3 + offset):
        input_seq[j] = filling_color
        output_seq[j] = filling_color

    seed_positions = []
    possible_positions = list(range(4 + offset, seq_len - 1))

    while possible_positions:
        pos = choice(possible_positions)
        if all(abs(pos - existing_pos) >= 5 for existing_pos in seed_positions):
            seed_positions.append(pos)
            input_seq[pos] = init_color
            for j in range(3):
                output_seq[pos + j - 1] = filling_color

            # rm if too close to this seed
            possible_positions = [p for p in possible_positions if abs(p - pos) >= 5]
        else:
            possible_positions.remove(pos)

    return input_seq, output_seq


##################################################
# Recolor

def gen_recolor_odd_even(seq_len, recolor_map, min_len=12, max_len=33, max_digits=10, background=0):
    """
    Generate a 1D recolor task based on odd-even object size
    :param recolor_map: Dictionary with 'odd' and 'even' keys specifying recoloring

    """
    seq_len = max(min(seq_len, max_len), min_len)

    ori_color = random.randint(1, max_digits)
    while ori_color == background or ori_color in recolor_map.values():
        ori_color = random.randint(1, max_digits)

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    start = 0
    has_odd, has_even = 0, 0

    while start < seq_len:
        bwt_space = random.randint(1, 3)
        start += bwt_space
        bar_len = random.randint(1, 5)

        if has_odd == 0 and bar_len % 2 == 0:
            bar_len += 1
            has_odd += 1
        elif has_even == 0 and bar_len % 2 == 1:
            bar_len += 1
            has_even += 1

        if start + bar_len > seq_len:
            break

        for j in range(bar_len):
            input_seq[start + j] = ori_color
            output_seq[start + j] = recolor_map['even'] if bar_len % 2 == 0 else recolor_map['odd']

        start += bar_len

    return input_seq, output_seq

def gen_recolor_size_cnt(seq_len, recolor_map, min_len=12, max_len=33, max_digits=10, background=0):
    """
    Generate a 1D recolor task based on object size
    :param recolor_map: Dictionary mapping {length : color}
    """
    seq_len = max(min(seq_len, max_len), min_len)

    ori_color = randint(1, max_digits)
    while ori_color == background or ori_color in recolor_map.values():
        ori_color = randint(1, max_digits)

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    start = 0
    bar_lens = [1, 2, 3]

    while start < seq_len:
        bwt_space = randint(1, 3)
        start += bwt_space

        if bar_lens:
            bar_len = choice(bar_lens)
            bar_lens.remove(bar_len)
        else:
            bar_len = randint(1, 3)

        if start + bar_len > seq_len:
            break

        for j in range(bar_len):
            input_seq[start + j] = ori_color
            output_seq[start + j] = recolor_map[bar_len]

        start += bar_len

    return input_seq, output_seq

def gen_recolor_max(seq_len, max_color, min_len=17, max_len=33, max_digits=10, background=0):
    """
    Generate a 1D recolor task based on maximum object size
    """
    seq_len = max(seq_len, min_len)
    seq_len = min(seq_len, max_len)

    ori_color = choice(list(set(range(1, max_digits)) - {max_color}))

    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    max_lens = [4, 5, 7]
    shuffle(max_lens)
    max_bar_len = choice(max_lens)

    space_l = []
    bar_l = []

    # Insert the max bar
    bwt_space = randint(1, 3)
    space_l.append(bwt_space)
    bar_l.append(max_bar_len)

    start = bwt_space + max_bar_len
    while start < seq_len:
        bwt_space = randint(1, 3)
        bar_len = randint(1, max_bar_len)

        if start + bwt_space + bar_len > seq_len:
            break

        space_l.append(bwt_space)
        bar_l.append(bar_len)
        start += bwt_space + bar_len

    order_l = list(range(len(bar_l)))
    shuffle(order_l)

    start = 0
    for i in order_l:
        bwt_space = space_l[i]
        bar_len = bar_l[i]
        start += bwt_space

        if start + bar_len > seq_len:
            break

        for j in range(bar_len):
            input_seq[start + j] = ori_color
            if bar_len == max_bar_len:
                output_seq[start + j] = max_color
            else:
                output_seq[start + j] = ori_color

        start += bar_len

    return input_seq, output_seq


##################################################

##################################################
# Sort

def gen_scattered_pixels_sorted(seq_len, num_pixels, min_len=32, max_len=33, max_digits=10, background=0):
    """
    A list of pixel locations, the output locations are identical but the colors are sorted.
    """
    seq_len = max(min(seq_len, max_len), min_len)

    # empty sequences
    input_seq = [background] * seq_len
    output_seq = [background] * seq_len

    # pixel positions
    positions = sorted(sample(range(seq_len), num_pixels))

    # unique colors
    colors = sample(range(1, max_digits + 1), num_pixels)
    sorted_colors = sorted(colors)

    for pos, c, sc in zip(positions, colors, sorted_colors):
        input_seq[pos] = c
        output_seq[pos] = sc

    return input_seq, output_seq


##################################################
# Stack

def gen_stack(seq_len, max_pixels, min_len=32, max_len=33, max_digits=10, background=0):
    """
    Generate a 1D task with scattered pixels in the input, and stack them to the right in the output.

    :param seq_len: Length of the sequence
    :param num_pixels: Number of colored pixels to include
    :param min_len: Minimum sequence length
    :param max_len: Maximum sequence length
    :param max_digits: Maximum number of colors (excluding background)
    :param background: Background color index
    :return: Tuple of input and output sequences
    """
    seq_len = max(min(seq_len, max_len), min_len)

    num_pixels = randint(2, max_pixels)

    # Generate input sequence
    input_seq = [background] * seq_len

    # Generate unique positions for the colored pixels
    positions = sorted(sample(range(seq_len), num_pixels))

    # Generate unique colors for the pixels
    colors = sample(range(1, max_digits + 1), num_pixels)

    # Place the colored pixels in the input sequence
    for pos, color in zip(positions, colors):
        input_seq[pos] = color

    # Create the output sequence
    output_seq = [background] * seq_len

    # Stack the pixels to the right in the output sequence
    for i, color in enumerate(colors):
        output_seq[seq_len - num_pixels + i] = color

    return input_seq, output_seq


##################################################
# Overlap

def gen_overlap_spread(seq_len, max_blocks=3, block_len=5, min_len=32, max_len=33, max_digits=10, background=0):
    """Generate a 1D task with overlapping blocks in the input, and disambiguate
    them in the output. A starter block will be on top of the stack, and then
    more blocks can be underneath it to the left or right. All block's indices
    are referenced from the left edge of the block.

    :param seq_len: Length of the sequence
    :param num_blocks: Number of blocks to generate (2 or 3)
    :param block_len: Length of each block
    :param min_len: Minimum sequence length
    :param max_len: Maximum sequence length
    :param max_digits: Maximum number of colors (excluding background)
    :param background: Background color index
    :return: Tuple of input and output sequences
    """
    seq_len = max(min(seq_len, max_len), min_len)
    num_blocks = min(max(max_blocks, 2), max_blocks)

    # Generate block colors
    colors = sample(range(1, max_digits + 1), num_blocks)

    # starter (topmost) block must be central so that submerged blocks are within seq bounds
    starter_pos = randint(block_len * num_blocks, seq_len - block_len * num_blocks)

    left_blocks = []
    left_bound = starter_pos - block_len

    right_blocks = []
    right_bound = starter_pos + block_len

    # submerged blocks
    for i in range(1, num_blocks):
        if choice([True, False]):  # go left
            pos = randint(left_bound + 1, left_bound + block_len - 1)
            left_blocks.append((pos, colors[i]))
            left_bound = pos - block_len
        else:  # go right
            pos = randint(right_bound - block_len + 1, right_bound - 1)
            right_blocks.append((pos, colors[i]))
            right_bound = pos + block_len

    ##########
    # Inputs

    # Create input sequence
    input_seq = [background] * seq_len

    # Add left blocks
    for pos, color in reversed(left_blocks):
        for i in range(block_len):
            input_seq[pos + i] = color

    # Add right blocks
    for pos, color in reversed(right_blocks):
        for i in range(block_len):
            input_seq[pos + i] = color

    # Add starter block (on top)
    for i in range(block_len):
        input_seq[starter_pos + i] = colors[0]


    ##########
    # Outputs

    # Create output sequence
    output_seq = [background] * seq_len

    # starter/top block
    for i in range(starter_pos, starter_pos + block_len):
        output_seq[i] = colors[0]

    # left blocks
    for l, (_, c) in enumerate(left_blocks):
        for i in range(starter_pos - (l + 1) * block_len, starter_pos - l * block_len):
            output_seq[i] = c

    # left blocks
    for r, (_, c) in enumerate(right_blocks):
        for i in range(starter_pos + (r + 1) * block_len, starter_pos + (r + 2) * block_len):
            output_seq[i] = c

    return input_seq, output_seq


##################################################
# Random Rotation

def gen_color_rotation(seq_len, max_rotation=1, min_len=8, max_len=32, max_digits=10, background=0):
    """
    Generate a sequence of colors and rotate them by a random amount.

    Args:
    seq_len (int): Length of the sequence to generate.
    min_len (int): Minimum length of the color sequence.
    max_len (int): Maximum length of the color sequence.
    max_digits (int): Maximum number of different colors to use.
    background (int): Background color value.

    Returns:
    tuple: Two lists representing the input sequence and the rotated output sequence.
    """
    # Determine the length of the color sequence
    color_seq_len = random.randint(min_len, min(max_len, seq_len))

    # Generate a list of random colors
    colors = random.choices(range(1, max_digits + 1), k=color_seq_len)

    # Create the input sequence
    input_seq = [background] * seq_len
    start_pos = random.randint(0, seq_len - color_seq_len)
    input_seq[start_pos:start_pos + color_seq_len] = colors

    # Determine the rotation amount
    rotation = random.randint(1, max_rotation)

    # Create the output sequence with rotated colors
    output_seq = input_seq.copy()
    rotated_colors = colors[rotation:] + colors[:rotation]
    output_seq[start_pos:start_pos + color_seq_len] = rotated_colors

    return input_seq, output_seq


##################################################
# Magnets

def gen_magnets(seq_len, min_block_size=2, max_block_size=5, max_digits=10, background=0):
    """
    Generate two blocks of different sizes separated by at least one space.
    The smaller block moves one step towards the larger block in the output.

    Args:
    seq_len (int): Length of the sequence to generate.
    min_block_size (int): Minimum size of a block.
    max_block_size (int): Maximum size of a block.
    max_digits (int): Maximum number for color representation.
    background (int): Background color value.

    Returns:
    tuple: Two lists representing the input sequence and the output sequence with the smaller block moved.
    """
    # Ensure the sequence is long enough to accommodate two blocks and a space
    if seq_len < 2 * min_block_size + 1:
        raise ValueError("Sequence length is too short to accommodate two blocks and a space")

    # Generate two blocks of different sizes
    block1_size = random.randint(min_block_size, max_block_size)
    block2_size = random.randint(min_block_size, max_block_size)
    while block2_size == block1_size:
        block2_size = random.randint(min_block_size, max_block_size)

    # Determine the colors for the blocks
    color1 = random.randint(1, max_digits)
    color2 = random.randint(1, max_digits)
    while color2 == color1:
        color2 = random.randint(1, max_digits)

    # Calculate the maximum possible space between blocks
    max_space = seq_len - block1_size - block2_size

    # Ensure there's at least one space between blocks
    space = random.randint(1, max(1, max_space))

    # Create the input sequence
    input_seq = [background] * seq_len
    start1 = random.randint(0, seq_len - block1_size - space - block2_size)
    start2 = start1 + block1_size + space

    for i in range(start1, start1 + block1_size):
        input_seq[i] = color1
    for i in range(start2, start2 + block2_size):
        input_seq[i] = color2

    # Create the output sequence
    output_seq = input_seq.copy()

    # Move the smaller block towards the larger block
    if block1_size < block2_size:
        # Move block1 right
        for i in range(start1 + block1_size, start1, -1):
            output_seq[i] = output_seq[i-1]
        output_seq[start1] = background
    else:
        # Move block2 left
        for i in range(start2 - 1, start2 + block2_size - 1):
            output_seq[i] = output_seq[i+1]
        output_seq[start2 + block2_size - 1] = background

    return input_seq, output_seq


##################################################
# Helpers

# custom color map
custom_colors = {
    -1: '#222222',  # blank lines
    0: '#000000',   # Black
    1: '#e122a1',   # Pink
    2: '#22e1a1',   # Mint
    3: '#2261e1',   # Blue
    4: '#e16122',   # Orange
    5: '#61e122',   # Lime
    6: '#a122e1',   # Purple
    7: '#e1a122',   # Gold
    8: '#22a1e1',   # Sky Blue
    9: '#c12261',    # Red
    10: '#22c161',
}

custom_cmap = mcolors.ListedColormap([custom_colors[i] for i in sorted(custom_colors.keys())])
norm = mcolors.Normalize(vmin=min(custom_colors.keys()), vmax=max(custom_colors.keys()))

def visualize_datasets(datasets: Dict[str, TensorDataset], grid_width: int, grid_height: int, num_samples: int = 10):
    num_datasets = len(datasets)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(5 * grid_width, 5 * grid_height))
    fig.tight_layout(pad=3.0)

    for i, (dataset_name, dataset) in enumerate(datasets.items()):
        if i >= grid_width * grid_height:
            print("Warning: Not all datasets are displayed. Increase grid size to show all.")
            break

        row = i // grid_width
        col = i % grid_width
        ax = axs[row, col] if grid_height > 1 else axs[col]

        samples = [dataset[j] for j in range(min(num_samples, len(dataset)))]

        # interleave input output pairs for display
        interleaved_data = []
        for d in samples:
            input_seq = torch.tensor(d['inputs'])
            output_seq = torch.tensor(d['outputs'])
            blank_line = torch.full_like(input_seq, -1)
            interleaved_data.extend([input_seq, output_seq, blank_line])

        # rm last blank line
        interleaved_data = interleaved_data[:-1]
        interleaved_data = torch.stack(interleaved_data).numpy()

        im = ax.imshow(interleaved_data, cmap=custom_cmap, aspect='auto', norm=norm, interpolation='nearest')
        ax.set_title(f'{dataset_name}', fontsize=10)
        ax.set_yticks(range(1, len(interleaved_data), 3))
        ax.set_yticklabels([f'S{j+1}' for j in range(num_samples)], fontsize=8)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Remove any unused subplots
    for i in range(num_datasets, grid_width * grid_height):
        row = i // grid_width
        col = i % grid_width
        fig.delaxes(axs[row, col] if grid_height > 1 else axs[col])

    plt.colorbar(im, ax=axs, label='Digit Value', aspect=30, ticks=range(-1, 10))
    plt.show()


def generate_datasets(num_samples: int, seq_len: int, generators: Dict[str, Callable]) -> Dict[str, TensorDataset]:
    datasets = {}
    for dataset_name, generator in generators.items():
        inputs, outputs = [], []
        for _ in range(num_samples):
            input_seq, output_seq = generator(seq_len)
            inputs.append(input_seq)
            outputs.append(output_seq)

        inputs_tensor = torch.tensor(inputs)
        outputs_tensor = torch.tensor(outputs)
        datasets[dataset_name] = TensorDataset(inputs_tensor, outputs_tensor)

    return datasets




##################################################
##################################################
# Example usage


if False:

    seq_len = 32
    max_digits = 10
    num_samples = 20

    recolor_odd_even_map = {'odd': 1, 'even': 2}
    recolor_size_map = {x: x for x in range(max_digits)}


    generators = {
        "Random Identity": lambda s: gen_random_identity(seq_len, max_digits=10, background=0),
        "Identity": lambda s: gen_identity_block(s, min_len=2, max_len=5, max_digits=10, background=0),
        "Identity Multi": lambda s: gen_multi_identity(s, num_blocks=3, min_len=2, max_len=5, max_digits=10, background=0),
        "Single Color Denoising": lambda s: gen_single_1c_denoising(s, max_digits=10, background=0),
        "Multi-Color Denoising": lambda s: gen_single_mc_denoising(s, min_len=32, max_len=33, max_digits=10, background=0),
        "Basic Fill": lambda s: gen_basic_fill(s, min_len=8, max_len=32, min_hole_len=1, max_digits=10, background=0),
        "Multi Fill": lambda s: gen_multi_fill(s, min_len=8, max_len=32, min_hole_len=1, max_digits=10, background=0),
        "Hollow": lambda s: gen_hollow(s, min_len=8, max_len=32, min_hole_len=1, max_digits=10, background=0),
        "Single Flip": lambda s: gen_single_flip(s, min_len=8, max_len=32, max_digits=10, background=0),
        "Mirror": lambda s: gen_single_mirror(s, min_len=12, max_len=32, max_digits=10, background=0, pivot_pt=9),
        "Move Single Bar": lambda s: gen_move_single_bar(s, min_len=8, max_len=32, move_len=3, min_bar_len=3, max_digits=10, background=0),
        "Move to Pixel": lambda s: gen_move_to_pixel(s, min_len=8, max_len=32, min_bar_len=3, max_digits=10, background=0),
        "Move Towards Indicator": lambda s: gen_move_towards_indicator(s, min_len=8, max_len=32, max_move=2, min_bar_len=3, max_digits=10, background=0),
        "Extend to Pixel": lambda s: gen_extend_to_pixel(s, min_len=8, max_len=32, min_bar_len=3, max_digits=10, background=0),
        "Pattern Copy (Single Color)": lambda s: gen_grow_copy_color(s, min_len=32, max_len=33, max_digits=10, background=0),
        "Pattern Copy (Multi-Color)": lambda s: gen_grow(s, min_len=32, max_len=33, max_digits=10, background=0),
        "Recolor Odd-Even": lambda s: gen_recolor_odd_even(s, recolor_odd_even_map, min_len=12, max_len=33, max_digits=10, background=0),
        "Recolor by Size Count": lambda s: gen_recolor_size_cnt(s, recolor_size_map, min_len=12, max_len=33, max_digits=10, background=0),
        "Recolor by Max": lambda s: gen_recolor_max(s, max_color=1, min_len=17, max_len=33, max_digits=10, background=0),

        "Sort Colors": lambda s: gen_scattered_pixels_sorted(s, num_pixels=3, min_len=32, max_len=33, max_digits=3, background=0),
        "Stack Pixels": lambda s: gen_stack(s, max_pixels=5, min_len=32, max_len=33, max_digits=10, background=0),
        "Overlap Spread": lambda s: gen_overlap_spread(s, max_blocks=3, block_len=5, min_len=32, max_len=33, max_digits=10, background=0),
        "Color Rotation": lambda s: gen_color_rotation(s, min_len=8, max_len=32, max_digits=10, background=0),
        "Magnets": lambda s: gen_magnets(s, min_block_size=2, max_block_size=5, max_digits=10, background=0),
    }

    datasets = generate_datasets(num_samples, seq_len, generators)

    visualize_datasets(datasets, grid_width=7, grid_height=4, num_samples=10)
