'''

Ensure we can do 1d interpolation correctly

'''

import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def upsample_pytorch(x, target_size):
    return F.interpolate(x, size=target_size, mode='linear', align_corners=True)

def upsample_scipy(x, target_size):
    batch, time, dim = x.shape
    x_np = x.cpu().numpy()

    result = np.zeros((batch, time, target_size))

    for b in range(batch):
        for t in range(time):
            original_indices = np.linspace(0, 1, dim)
            new_indices = np.linspace(0, 1, target_size)
            interpolator = interp1d(original_indices, x_np[b, t], kind='linear')
            result[b, t] = interpolator(new_indices)

    return torch.from_numpy(result).to(x.device)

# batch=16, time=145, dim=1024
x = torch.randn(16, 145, 1024, dtype=torch.float64)
target_size = 4000

x_upsampled_pytorch = upsample_pytorch(x, target_size)
x_upsampled_scipy = upsample_scipy(x, target_size)
is_close = torch.allclose(x_upsampled_pytorch, x_upsampled_scipy, atol=1e-8)
max_diff = torch.max(torch.abs(x_upsampled_pytorch - x_upsampled_scipy))

print(f"Outputs are close: {is_close}")
print(f"Maximum difference: {max_diff:.6e}")

# Visualize a slice of the results
b, t = 0, 0  # Choose a batch and time index to visualize
plt.figure(figsize=(12, 6))
plt.plot(x_upsampled_pytorch[b, t, :100].cpu().numpy(), label='PyTorch')
plt.plot(x_upsampled_scipy[b, t, :100].cpu().numpy(), label='SciPy', linestyle='--')
plt.legend()
plt.title("Comparison of PyTorch and SciPy Upsampling (first 100 values)")
plt.show()
