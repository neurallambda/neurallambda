'''

RESULTS: prelim results show naive version can beat optimized version, or vice versa, depending on batch_size, and especially out_channels

'''

import torch
import torch.nn.functional as F
import time

def batch_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    batch_size, in_channels, height, width = input.shape
    out_channels, in_channels, kernel_height, kernel_width = weight.shape[1:]

    # Reshape input and weight
    input = input.view(1, batch_size * in_channels, height, width)
    weight = weight.view(batch_size * out_channels, in_channels, kernel_height, kernel_width)

    # Perform convolution
    output = F.conv2d(input,
                      weight,
                      bias=None,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=batch_size)

    # Reshape output
    output = output.view(batch_size, out_channels, output.shape[2], output.shape[3])

    # Add bias if provided
    if bias is not None:
        output += bias.view(batch_size, out_channels, 1, 1)

    return output

def naive_batch_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    batch_size, in_channels, height, width = input.shape
    out_channels = weight.shape[1]

    # Initialize output tensor
    output_height = (height + 2 * padding - weight.shape[3]) // stride + 1
    output_width = (width + 2 * padding - weight.shape[4]) // stride + 1
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input.device)

    # Iterate over batch
    for i in range(batch_size):
        output[i] = F.conv2d(input[i].unsqueeze(0), weight[i], bias=None if bias is None else bias[i],
                             stride=stride, padding=padding, dilation=dilation)

    return output

# Test function
def test_equivalence(batch_size, in_channels, out_channels, height, width, kernel_size):
    input = torch.randn(batch_size, in_channels, height, width)
    weight = torch.randn(batch_size, out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.randn(batch_size, out_channels)

    # Run both implementations
    start = time.time()
    output1 = batch_conv2d(input, weight, bias, padding=1)
    time1 = time.time() - start

    start = time.time()
    output2 = naive_batch_conv2d(input, weight, bias, padding=1)
    time2 = time.time() - start

    # Check equivalence
    is_equal = torch.allclose(output1, output2, atol=1e-6)

    print(f"Outputs are equivalent: {is_equal}")
    print(f"Optimized version time: {time1:.6f} seconds")
    print(f"Naive version time: {time2:.6f} seconds")
    print(f"Speed-up factor: {time2/time1:.2f}x")

# Run the test
test_equivalence(batch_size=32, in_channels=3, out_channels=8, height=32, width=32, kernel_size=3)
