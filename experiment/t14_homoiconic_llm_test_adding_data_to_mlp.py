'''

WRITEUP: https://x.com/neurallambda/status/1842814175501766757

TAKEAWAY: scale LoR values by sqrt(2 / dim)


MLP Low-Rank Knowledge Injection

This analyzes a low-rank update technique for a 2 layer MLP, via inserting a KV pair into the 2 layers.

Key components:

* Vanilla MLP
* Low-rank update mechanism, calculated from outer products of a K, V, with an intermediate value
* Compare vanilla version with 2 versions: separate low-rank weights & low-rank weights added into original weights

The experiment explores three types of key vectors:

Random: randn keys
Orthogonal: keys made orthogonal to the input
Identical: keys identical to the input

For each key type, the module computes and compares:

Similarity between the original MLP output and the low-rank updated output - the output should still be similar to the original output after the low rank step
Similarity between the low-rank updated output and the target 'v' vector - the output should also include the new v value

'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

def setup_mlp(idim, hdim):
    return {
        'U': torch.randn(hdim, idim),
        'D': torch.randn(idim, hdim),
        # 'norm': lambda x: x,
        # 'norm': nn.LayerNorm(idim),
        'norm': nn.RMSNorm(idim),
    }

def original_forward(mlp, x):
    """ forward pass through the original MLP """
    h = torch.einsum('ji, bi -> bj', mlp['U'], x).relu()
    return mlp['norm'](torch.einsum('ij, bj -> bi', mlp['D'], h))

def low_rank_forward(mlp, x, loru, lord):
    """ Perform forward pass with separate low-rank update """
    h = torch.einsum('ji, bi -> bj', mlp['U'], x).relu()
    lh = torch.einsum('bji, bi -> bj', loru, x).relu()
    y1 = torch.einsum('ij, bj -> bi', mlp['D'], h)
    y2 = torch.einsum('bij, bj -> bi', lord, lh)
    return mlp['norm'](y1 + y2)

def integrated_low_rank_forward(mlp, x, loru, lord):
    """ Perform forward pass with low-rank update integrated into original weights """
    B = loru.shape[0]
    U_updated = mlp['U'].unsqueeze(0).repeat(B, 1, 1) + loru
    D_updated = mlp['D'].unsqueeze(0).repeat(B, 1, 1) + lord
    h = torch.einsum('bji, bi -> bj', U_updated, x).relu()
    return mlp['norm'](torch.einsum('bij, bj -> bi', D_updated, h))

def build_lor_weights(k, v, inter_dim, lor_scale):
    """ Generate low-rank update matrices by outer producting keys and values, and introducing a random intermediate vector to tie them together """
    inter = torch.randn(k.shape[0], inter_dim)
    loru = torch.einsum('bi, bj -> bji', k, inter)
    lord = torch.einsum('bj, bi -> bij', inter, v)
    return (
        loru * lor_scale,
        lord * lor_scale
    )

def run_experiment(B, idim, hdim, lor_scale, num_trials):
    k_types = ['random', 'orthogonal', 'identical']
    mlp = setup_mlp(idim, hdim)
    results = {k_type: {'ly_y_sim': [], 'ly_v_sim': [],  # separated lor computations
                        'ily_y_sim': [], 'ily_v_sim': []}  # integrated lor computations
               for k_type in k_types}

    for k_type in k_types:
        for _ in range(num_trials):
            x = torch.randn(B, idim)

            if k_type == 'random':
                k = torch.randn(B, idim)
            elif k_type == 'orthogonal':
                k = torch.randn(B, idim)
                projection = (torch.sum(k * x, dim=1, keepdim=True) / torch.sum(x * x, dim=1, keepdim=True)) * x
                k = k - projection
            elif k_type == 'identical':
                k = x.clone()

            v = torch.randn(B, idim)

            loru, lord = build_lor_weights(k, v, hdim, lor_scale)

            y = original_forward(mlp, x)
            ly = low_rank_forward(mlp, x, loru, lord)
            ily = integrated_low_rank_forward(mlp, x, loru, lord)

            ly_y_sim = torch.cosine_similarity(y, ly, dim=1).mean().item()
            ly_v_sim = torch.cosine_similarity(ly, v, dim=1).mean().item()
            ily_y_sim = torch.cosine_similarity(y, ily, dim=1).mean().item()
            ily_v_sim = torch.cosine_similarity(ily, v, dim=1).mean().item()

            results[k_type]['ly_y_sim'].append(ly_y_sim)
            results[k_type]['ly_v_sim'].append(ly_v_sim)
            results[k_type]['ily_y_sim'].append(ily_y_sim)
            results[k_type]['ily_v_sim'].append(ily_v_sim)

    return results


##################################################
# 2D Plots

def plot_results(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

    for k_type, data in results.items():
        trials = range(1, len(data['ly_y_sim']) + 1)
        ax1.plot(trials, data['ly_y_sim'], label=f'{k_type} k')
        ax2.plot(trials, data['ly_v_sim'], label=f'{k_type} k')
        ax3.plot(trials, data['ily_y_sim'], label=f'{k_type} k')
        ax4.plot(trials, data['ily_v_sim'], label=f'{k_type} k')

    ax1.set_title('Separate LoRA: Cosine Similarity between ly and y')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Cosine Similarity')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Separate LoRA: Cosine Similarity between ly and v')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Cosine Similarity')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Integrated LoRA: Cosine Similarity between ily and y')
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Cosine Similarity')
    ax3.legend()
    ax3.grid(True)

    ax4.set_title('Integrated LoRA: Cosine Similarity between ily and v')
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Cosine Similarity')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

if True:
    print('running 2D plot')
    idim = 1024
    hdim = 4096

    # lor_scale = (1 / idim) ** 0.5  #
    lor_scale = (2 / idim) ** 0.5  #

    results = run_experiment(1, idim, hdim, lor_scale, num_trials=50)
    plot_results(results)


##################################################
# 3D plots, scanning over lor_scale

def run_experiment(B, idim, hdim, lor_scales, num_trials):
    k_types = ['random', 'orthogonal', 'identical']
    mlp = setup_mlp(idim, hdim)
    results = {k_type: {
        'ly_y_sim': np.zeros((len(lor_scales), num_trials)),
        'ly_v_sim': np.zeros((len(lor_scales), num_trials)),
        'ily_y_sim': np.zeros((len(lor_scales), num_trials)),
        'ily_v_sim': np.zeros((len(lor_scales), num_trials))
    } for k_type in k_types}

    for i, lor_scale in enumerate(lor_scales):
        for k_type in k_types:
            for j in range(num_trials):
                x = torch.randn(B, idim)

                if k_type == 'random':
                    k = torch.randn(B, idim)
                elif k_type == 'orthogonal':
                    k = torch.randn(B, idim)
                    projection = (torch.sum(k * x, dim=1, keepdim=True) / torch.sum(x * x, dim=1, keepdim=True)) * x
                    k = k - projection
                elif k_type == 'identical':
                    k = x.clone()

                v = torch.randn(B, idim)

                loru, lord = build_lor_weights(k, v, hdim, lor_scale)

                y = original_forward(mlp, x)
                ly = low_rank_forward(mlp, x, loru, lord)
                ily = integrated_low_rank_forward(mlp, x, loru, lord)

                results[k_type]['ly_y_sim'][i, j] = torch.cosine_similarity(y, ly, dim=1).mean().item()
                results[k_type]['ly_v_sim'][i, j] = torch.cosine_similarity(ly, v, dim=1).mean().item()
                results[k_type]['ily_y_sim'][i, j] = torch.cosine_similarity(y, ily, dim=1).mean().item()
                results[k_type]['ily_v_sim'][i, j] = torch.cosine_similarity(ily, v, dim=1).mean().item()

    return results

def plot_3d_results(results, lor_scales, num_trials):
    fig = plt.figure(figsize=(20, 20))
    plot_titles = [
        'Separate LoRA: Cosine Similarity between ly and y',
        'Separate LoRA: Cosine Similarity between ly and v',
        'Integrated LoRA: Cosine Similarity between ily and y',
        'Integrated LoRA: Cosine Similarity between ily and v'
    ]
    plot_keys = ['ly_y_sim', 'ly_v_sim', 'ily_y_sim', 'ily_v_sim']

    for idx, (title, key) in enumerate(zip(plot_titles, plot_keys), 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        for k_type, data in results.items():
            X, Y = np.meshgrid(lor_scales, range(1, num_trials + 1))
            Z = data[key].T  # Transpose to match X and Y shapes

            ax.plot_surface(X, Y, Z, label=f'{k_type} k', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('LoR Scale')
        ax.set_ylabel('Trial')
        ax.set_zlabel('Cosine Similarity')
        ax.legend()

    plt.tight_layout()
    plt.show()


if False:
    print('running 3D plot')
    idim = 1024
    hdim = 4096

    # s = (1 / idim) ** 0.5
    s = (2 / idim) ** 0.5
    lor_scales = torch.linspace(s / 5, s * 5, 11)  # 11 points from 0 to 1
    num_trials = 20

    results = run_experiment(1, idim, hdim, lor_scales, num_trials)
    plot_3d_results(results, lor_scales, num_trials)
