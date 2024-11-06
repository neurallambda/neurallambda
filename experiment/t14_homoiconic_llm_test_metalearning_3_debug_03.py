'''

A simple homoiconic MLP for solving the N-way k-shot task. Upgrade so that LORs are derived from "Reasoning Embeddings", not linear weights.


>>> nn.Conv2d(in_channels=5, out_channels=13, kernel_size=3, stride=1)(torch.randn((3, 5, 7, 11))).shape
torch.Size([3, 13, 5, 9])

>>> nn.Conv2d(in_channels=5, out_channels=13, kernel_size=3, stride=1, padding=1).weight.shape
torch.Size([3, 13, 7, 11])

>>> nn.Conv2d(in_channels=5, out_channels=13, kernel_size=3, stride=1, padding=1).weight.shape
torch.Size([13, 5, 3, 3])

'''

import os
import math
# import sys
import traceback
import importlib
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
from functools import partial
import itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt

from t13_metalearning_hypernet_data import omniglot_n_way_k_shot, ALPHABET_DICT
from neurallambda.lab.common import print_model_info
from torch.autograd import Function

from einops import rearrange, repeat

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'


##################################################
# Params

img_dim = 28
D = 64
label_dim = D


# data
B = 32
train_alphabets = ["Latin", "Greek"]
test_alphabets = ["Mongolian"]
img_size = 28
n_way = 2  # N-way classification
k_shot = 3  # k-shot learning
q_query = 1  # query examples per class
num_tasks = B * 50  # number of tasks per epoch



##################################################
# Batchified functions

def batch_linear(input, weight, bias=None):
    ''' uses batch-wise different weights and biases '''
    output = torch.bmm(input.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
    if bias is not None:
        output += bias
    return output


##################################################
# Functional Optimizers
#
#   Differentiable variants of: SGD, SGD-Momentum, RMSProp, Adam


##########
# SGD

def sgd_init(params):
    # just used to maintain call-signature parity with other optimizers
    return []

def sgd(params, grads, opt_state: List[Any], lr=0.01):
    # opt_state is ignored. included to maintain signature parity with other optimizers
    return (
        [p - g * lr for p, g in zip(params, grads)],
        []  # empty opt_state
    )


##########
# SGD Momentum

def sgd_momentum_init(params):
    return [torch.zeros_like(p) for p in params]

def sgd_momentum(params, grads, opt_state: List[Any], lr=0.01, momentum=0.9):
    velocity = opt_state[0]
    updated_velocity = [momentum * v + lr * g for v, g in zip(velocity, grads)]
    updated_params = [p - v for p, v in zip(params, updated_velocity)]
    return updated_params, updated_velocity


##########
# RMSProp

def rmsprop_init(params):
    return [torch.zeros_like(p) for p in params]  # square averages

def rmsprop(params, grads, opt_state: List[Any], lr=0.01, alpha=0.99, eps=1e-8):
    square_avg = opt_state[0]
    updated_square_avg = [alpha * avg + (1 - alpha) * g.pow(2) for avg, g in zip(square_avg, grads)]
    updated_params = [p - lr * g / (avg.sqrt() + eps)
                      for p, g, avg in zip(params, grads, updated_square_avg)]
    return updated_params, updated_square_avg


##########
# ADAM

def adam_init(params):
    return (
        [torch.zeros_like(p) for p in params],  # m
        [torch.zeros_like(p) for p in params],  # v
        0  # t
    )

def adam(params, grads, opt_state, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
    m, v, t = opt_state
    t += 1
    updated_m = [betas[0] * m_i + (1 - betas[0]) * g
                 for m_i, g in zip(m, grads)]
    updated_v = [betas[1] * v_i + (1 - betas[1]) * g.pow(2)
                 for v_i, g in zip(v, grads)]

    m_hat = [m_i / (1 - betas[0]**t) for m_i in updated_m]
    v_hat = [v_i / (1 - betas[1]**t) for v_i in updated_v]

    updated_params = [p - lr * mh / (vh.sqrt() + eps)
                      for p, mh, vh in zip(params, m_hat, v_hat)]

    return updated_params, updated_m, updated_v, t


##########
# Adamax

def adamax_init(params):
    return (
        [torch.zeros_like(p) for p in params],  # m (first moment)
        [torch.zeros_like(p) for p in params],  # u (infinity norm)
        0  # t (timestep)
    )

def adamax(params, grads, opt_state, lr=0.002, betas=(0.9, 0.999), eps=1e-8):
    """
    Implements Adamax optimization algorithm (a variant of Adam based on infinity norm).

    Args:
        params: List of parameters
        grads: List of gradients
        m: List of first moment vectors
        u: List of infinity norm vectors
        t: Integer, timestep counter
        lr: Float, learning rate (default: 0.002)
        betas: Tuple of coefficients for computing running averages
        eps: Term for numerical stability

    Returns:
        (updated_params, updated_m, updated_u, t)
    """
    m, u, t = opt_state
    t += 1

    # Update biased first moment estimate
    updated_m = [betas[0] * m_i + (1 - betas[0]) * g
                 for m_i, g in zip(m, grads)]

    # Update the infinity norm estimate
    # u_t = max(beta_2 * u_{t-1}, |g_t|)
    updated_u = [torch.maximum(betas[1] * u_i, torch.abs(g))
                 for u_i, g in zip(u, grads)]

    # Bias correction for the first moment
    # Note: infinity norm doesn't need bias correction
    m_hat = [m_i / (1 - betas[0]**t) for m_i in updated_m]

    # Update parameters
    # theta_t = theta_{t-1} - (alpha / u_t) * m_hat_t
    updated_params = [p - (lr / (u_i + eps)) * mh
                      for p, mh, u_i in zip(params, m_hat, updated_u)]

    return updated_params, updated_m, updated_u, t


##################################################
# Data

global_epoch = 0


# training
num_epochs = 1000
lr = 1e-4
wd = 1e-2
alpha = 1e-1  # scale of weight_loss loss


##########
# Dataset

try:
    already_loaded
except:
    train_dl, test_dl = omniglot_n_way_k_shot(
        train_alphabets,
        test_alphabets,
        n_way,
        k_shot,
        q_query,
        num_tasks,
        img_size,
        B,
        seed_train=1,
        seed_test=2,
    )
    already_loaded = True

   #  # TODO: remove hack. The omniglot dataset has separate train/test splits,
   #  # but if I want random label pairings, but keep the same alphabet across
   #  # train/test, I need to do this dumb hack rn.
   #  train_dl, _ = omniglot_n_way_k_shot(
   #      train_alphabets,
   #      ["Mongolian"],  # not used
   #      n_way,
   #      k_shot,
   #      q_query,
   #      num_tasks,
   #      img_size,
   #      B,
   #      seed_train=1,
   #      seed_test=2,
   #  )

   #  test_dl, _ = omniglot_n_way_k_shot(
   #      test_alphabets,
   #      ["Mongolian"],  # not used
   #      n_way,
   #      k_shot,
   #      q_query,
   #      num_tasks,
   #      img_size,
   #      B,
   #      seed_train=3,
   #      seed_test=4,
   # )

    already_loaded = True


if False:
    # Explore Data
    for batch in train_dl:
        supports, queries = batch
        # support_imgs.shape   = [32, 2, 784]
        # support_labels.shape = [32, 2]
        # query_imgs.shape     = [32, 2, 784]
        # query_labels.shape   = [32, 2]
        support_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in supports], dim=1)  # N*k tensors [B, IMG, IMG]
        support_labels = torch.stack([x[1].to(DEVICE) for x in supports], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in queries], dim=1)  # N*k tensors [B, IMG, IMG]
        query_labels = torch.stack([x[1].to(DEVICE) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        break


if False:
    # START_BLOCK_4

    # Explore Data
    # for batch in train_dl:
    for batch in test_dl:
        supports, queries = batch
        support_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in supports], dim=1)  # N*k tensors [B, IMG, IMG]
        support_labels = torch.stack([x[1].to(DEVICE) for x in supports], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in queries], dim=1)  # N*k tensors [B, IMG, IMG]
        query_labels = torch.stack([x[1].to(DEVICE) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        break

    # We'll visualize the first batch only
    batch_idx = 6

    # Calculate image size (assuming square images)
    img_size = int(np.sqrt(support_imgs.shape[-1]))

    # Create a figure with a grid: k_shot + 1 rows (support + query) x n_way columns
    fig, axes = plt.subplots(k_shot + 1, n_way, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.5)

    # Plot support images
    for i in range(k_shot):
        for j in range(n_way):
            idx = i * n_way + j
            img = support_imgs[batch_idx, idx].cpu().numpy().reshape(img_size, img_size)
            label = support_labels[batch_idx, idx].cpu().numpy()

            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Class {label}')

    # Plot query images
    for j in range(n_way):
        img = query_imgs[batch_idx, j].cpu().numpy().reshape(img_size, img_size)
        label = query_labels[batch_idx, j].cpu().numpy()

        axes[-1, j].imshow(img, cmap='gray')
        axes[-1, j].axis('off')
        axes[-1, j].set_title(f'Query\nClass {label}')

    # Add row labels
    for i in range(k_shot):
        axes[i, 0].set_ylabel(f'Support\nSet {i+1}', rotation=0, labelpad=40)
    axes[-1, 0].set_ylabel('Query\nSet', rotation=0, labelpad=40)

    plt.suptitle('Omniglot Few-Shot Learning Task Visualization\n'
                 f'{n_way}-way {k_shot}-shot with 1 query example per class',
                 y=1.02)
    plt.tight_layout()

    plt.show()

    # END_BLOCK_4



##################################################

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = torch.mm(input, weight.t()) + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), input)
        grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)



##################################################


# START_BLOCK_1

class RandomScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if not self.training:
            return x
        s = torch.rand_like(x) * self.scale
        return x * s


class NullOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class SRWM(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.w_init = nn.Parameter(torch.randn(
            d_out + d_in * 2 + 4,  # y, k, q, lr=(wy, wk, yq, wlr)  # separate lr for targeting each submatrix
            d_in
        ))
        nn.init.xavier_uniform_(self.w_init)
        # nn.init.orthogonal_(self.w_init)

    def forward(self, x, w):
        # note: for supervised tasks, use x. for unsupervised tasks use softmax(x)
        out = torch.einsum('bed, bd -> be', w, x)
        y, k, q, b1, b2, b3, b4 = torch.split(out, [self.d_out, self.d_in, self.d_in, 1, 1, 1, 1], dim=1)
        kphi = k.softmax(dim=1)
        qphi = q.softmax(dim=1)
        vbar = torch.einsum('bed, bd -> be', w, kphi)
        v    = torch.einsum('bed, bd -> be', w, qphi)
        beta = torch.cat([
            b1.expand(-1, self.d_out),
            b2.expand(-1, self.d_in),
            b3.expand(-1, self.d_in),
            b4.expand(-1, 4)
        ], dim=1)
        dw = beta.sigmoid().unsqueeze(2) * torch.einsum('be, bd -> bed', v - vbar, kphi)
        w_out = w + dw
        return y, w_out


# @@@@@@@@@@
# usage
if False:
    B = 5
    d_in = 7
    d_out = 3
    x = torch.randn(B, d_in)
    model = SRWM(d_in, d_out)
    w = model.w_init.unsqueeze(0).repeat(B, 1, 1)
    y, w_out = model(x, w)
    brk
# @@@@@@@@@@



class Model(nn.Module):
    def __init__(self, dim, img_dim):
        super().__init__()
        self.dim = dim

        # image input
        norm = nn.BatchNorm2d
        # norm = NullOp

        drp_p = 0.5
        drp = nn.Dropout
        # drp = NullOp
        self.img_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), norm(32), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), norm(32), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), norm(32), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1), norm(1), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),

            nn.Flatten(1, -1),
            nn.Linear(8 ** 2, dim, bias=False),
        )

        # label input
        self.emb = nn.Parameter(torch.randn(n_way, label_dim))
        torch.nn.init.orthogonal_(self.emb)

        # backbone
        self.w1 = nn.Parameter(torch.randn(dim, dim))
        torch.nn.init.orthogonal_(self.w1)

        self.w2 = nn.Parameter(torch.randn(dim, dim))
        torch.nn.init.orthogonal_(self.w2)

        # l1, r1, l2, r2: lor tags
        # q: query tag
        # serr, qerr: support/query error tag
        # hs: hidden state
        self.tags = nn.Parameter(torch.randn(8, dim))
        torch.nn.init.orthogonal_(self.tags)

        # EXPERIMENT: generate LORs from attention over "reasoning embs", not linear projections
        n_r_embs = 16
        n_r_heads = 4
        head_dim = dim // n_r_heads
        self.head_dim = head_dim

        # RQ is a projection matrix
        self.r_q = nn.Parameter(torch.randn(n_r_heads, head_dim, dim))
        torch.nn.init.orthogonal_(self.r_q)

        # RK and RV are "embeddings"
        self.r_k = nn.Parameter(torch.randn(n_r_embs, n_r_heads, head_dim))
        torch.nn.init.orthogonal_(self.r_k)

        self.r_v = nn.Parameter(torch.randn(n_r_embs, n_r_heads, head_dim))
        torch.nn.init.orthogonal_(self.r_v)

        self.r_o = nn.Parameter(torch.randn(dim, dim))
        torch.nn.init.orthogonal_(self.r_o)

        self.r_ln = nn.LayerNorm(dim)
        with torch.no_grad():
            self.r_ln.weight[:] = torch.zeros_like(self.r_ln.weight) + 1e-3  # initially gate out lor info

        # EXPERIMENT: DFA for Metalearning
        self.dfa_l1 = (torch.randn(dim) * 1)
        self.dfa_r1 = (torch.randn(dim) * 1)
        self.dfa_l2 = (torch.randn(dim) * 1)
        self.dfa_r2 = (torch.randn(dim) * 1)


    def get_lor(self, x, w1, w2):
        B, D = x.shape
        q, _ = self.net(x, w1, w2)
        q = torch.einsum('nhd, bd -> bnh', self.r_q, q)
        attn_scores = torch.einsum('bnh, rnh -> bnr', q, self.r_k)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        # emb_attn = F.gumbel_softmax(emb_attn, dim=1, hard=True)
        attn_probs = F.softmax(attn_scores, dim=-1)
        lor = torch.einsum('bnr, rnh -> bnh', attn_probs, self.r_v)
        lor = lor.reshape(B, D)
        lor = torch.einsum('de, bd -> be', self.r_o, lor)
        lor = self.r_ln(lor + x)
        return lor


    def net(self, x, w1, w2):
        h = torch.einsum('bed, bd -> be', w1, x)
        ah = F.gelu(h)
        y = torch.einsum('bed, bd -> be', w2, ah)
        # EXPERIMENT: return h for self modelling loss
        return y, h

        # # Unbactched weights
        # h = torch.einsum('bed, bd -> be', w1, x)
        # ah = F.gelu(h)
        # y = torch.einsum('bed, bd -> be', w2, ah)
        # # EXPERIMENT: return h for self modelling loss
        # return y, h


    def forward(self,
                query_w,
                sxs, sys,  # support xs/ys
                qxs, qys,  # query xs/ys
                inner_lr):
        B = query_w.shape[0]
        S = sxs.shape[1]
        Q = qxs.shape[1]

        # embed inputs
        sxs    = self.img_in(sxs.view(-1, 1, img_dim, img_dim)).view(B, n_way * k_shot, -1)  # [B, S, D]
        sys_emb = self.emb[sys]

        # embed outputs
        qxs = self.img_in(qxs.view(-1, 1, img_dim, img_dim)).view(B, n_way * q_query, -1)  # [B, Q, D]
        qys_emb = self.emb[qys]

        (l1e, r1e, l2e, r2e,
         qe,
         serr, qerr,
         hs) = self.tags

        l1s = []
        r1s = []
        l2s = []
        r2s = []

        # Batched weights
        w1 = self.w1.unsqueeze(0).repeat(B, 1, 1)
        w2 = self.w2.unsqueeze(0).repeat(B, 1, 1)

        # # Unbatched weights
        # w1 = self.w1
        # w2 = self.w2

        # Build up LORs based on Supports
        for six in range(S):

            bound = (
                sxs[:, six]  +
                sys_emb[:, six]  #  * torch.rand(B, D, device=DEVICE)  # EXPERIMENT: noise
            )

            l1s.append(self.get_lor(bound + l1e, w1, w2))
            r1s.append(self.get_lor(bound + r1e, w1, w2))
            l2s.append(self.get_lor(bound + l2e, w1, w2))
            r2s.append(self.get_lor(bound + r2e, w1, w2))

            # l1s.append(self.net(bound + torch.randn_like(bound) * 1e-1, w1, w2))
            # r1s.append(self.net(bound + torch.randn_like(bound) * 1e-1, w1, w2))
            # l2s.append(self.net(bound + torch.randn_like(bound) * 1e-1, w1, w2))
            # r2s.append(self.net(bound + torch.randn_like(bound) * 1e-1, w1, w2))


        # #####
        # # Metalearning

        # # Apply LORs
        # l1s = torch.stack(l1s, dim=1).requires_grad_()
        # r1s = torch.stack(r1s, dim=1).requires_grad_()
        # l2s = torch.stack(l2s, dim=1).requires_grad_()
        # r2s = torch.stack(r2s, dim=1).requires_grad_()

        # with torch.enable_grad():
        #     params = [l1s, r1s, l2s, r2s]
        #     opt_state = opt_state_fn(params)

        #     for _ in range(N_LOOPS):
        #         sw1 = w1 + torch.einsum('bsl, bsr -> blr', l1s, r1s)
        #         sw2 = w2 + torch.einsum('bsl, bsr -> blr', l2s, r2s)

        #         # Use LORs on support images
        #         pred_embs = []
        #         for six in range(S):
        #             pred, _ = self.net(sxs[:, six] + qe, sw1, sw2)  # query tag on support images
        #             pred_embs.append(pred)

        #         # task loss
        #         pred_embs = torch.stack(pred_embs, dim=1)  # [B, S, D]
        #         pred_labels = torch.einsum('ld, bsd -> bsl', self.emb, pred_embs);    assert pred_labels.shape == torch.Size([B, sys.shape[1], n_way])
        #         task_loss = F.cross_entropy(pred_labels.view(-1, n_way),
        #                                     sys.view(-1))

        #         grads = torch.autograd.grad(task_loss, params, create_graph=True)
        #         [l1s, r1s, l2s, r2s], opt_state = opt_fn(params, grads, opt_state, lr=inner_lr)


        # #####
        # # DFA Metalearning

        # # Apply LORs
        # l1s = torch.stack(l1s, dim=1)  # [B, S, D]
        # r1s = torch.stack(r1s, dim=1)
        # l2s = torch.stack(l2s, dim=1)
        # r2s = torch.stack(r2s, dim=1)

        # N_DFA_LOOPS = 5
        # DFA_LR = 1e-3
        # for _ in range(N_DFA_LOOPS):
        #     sw1 = w1 + torch.einsum('bsl, bsr -> blr', l1s, r1s)
        #     sw2 = w2 + torch.einsum('bsl, bsr -> blr', l2s, r2s)

        #     # Use LORs on support images
        #     pred_embs = []
        #     for six in range(S):
        #         pred, _ = self.net(sxs[:, six] + qe, sw1, sw2)  # query tag on support images
        #         pred_embs.append(pred)

        #     # task loss
        #     pred_embs = torch.stack(pred_embs, dim=1)  # [B, S, D]
        #     pred_labels = torch.einsum('ld, bsd -> bsl', self.emb, pred_embs);    assert pred_labels.shape == torch.Size([B, sys.shape[1], n_way])
        #     task_loss = F.cross_entropy(pred_labels.view(-1, n_way), sys.view(-1), reduction='none')

        #     task_loss = -task_loss.view(B, S).unsqueeze(2).repeat(1, 1, D) * DFA_LR

        #     l1s = l1s + self.dfa_l1.to(DEVICE).expand(B, S, -1) * task_loss
        #     r1s = r1s + self.dfa_r1.to(DEVICE).expand(B, S, -1) * task_loss
        #     l2s = l2s + self.dfa_l2.to(DEVICE).expand(B, S, -1) * task_loss
        #     r2s = r2s + self.dfa_r2.to(DEVICE).expand(B, S, -1) * task_loss


        #####
        # Test

        l1s = torch.stack(l1s, dim=1)
        r1s = torch.stack(r1s, dim=1)
        l2s = torch.stack(l2s, dim=1)
        r2s = torch.stack(r2s, dim=1)

        # EXPERIMENT: + LORS
        qw1 = w1 + torch.einsum('bsl, bsr -> blr', l1s, r1s)
        qw2 = w2 + torch.einsum('bsl, bsr -> blr', l2s, r2s)

        # # EXPERIMENT: only use lors
        # qw1 = torch.einsum('bsl, bsr -> blr', l1s, r1s)
        # qw2 = torch.einsum('bsl, bsr -> blr', l2s, r2s)

        # EXPERIMENT with *
        # qw1 = w1 * torch.einsum('bsl, bsr -> blr', l1s, r1s)
        # qw2 = w2 * torch.einsum('bsl, bsr -> blr', l2s, r2s)

        pred_embs = []
        for qix in range(Q):
            pred, h = self.net(qxs[:, qix] + qe, qw1, qw2)
            pred_embs.append(pred)

        # task loss
        pred_embs = torch.stack(pred_embs, dim=1)  # [B, Q, D]

        # pred_embs = torch.randn_like(pred_embs)

        pred_labels = torch.einsum('ld, bqd -> bql', self.emb, pred_embs);    assert pred_labels.shape == torch.Size([B, qys.shape[1], n_way])
        task_loss = F.cross_entropy(pred_labels.view(-1, n_way),
                                    qys.view(-1))

        n_correct = (pred_labels.view(-1, n_way).argmax(dim=1) == qys.view(-1)).sum()
        n_q = B * Q


        # #####
        # # error prediction
        # error = pred_embs - qys_emb
        # error_preds = []
        # for qix in range(Q):
        #     bound = qxs[:, qix]  # + qys_emb[:, qix]
        #     error_preds.append(self.net(bound + qerr, qw1, qw2))
        # error_pred = torch.stack(error_preds, dim=1)
        # error_pred_loss = F.mse_loss(error, error_pred)


        # #####
        # # Hidden State Prediction
        # h_err = []
        # for qix in range(Q):
        #     bound = qxs[:, qix]  # + qys_emb[:, qix]
        #     h_pred, h_actual = self.net(bound + hs, qw1, qw2)
        #     h_err.append(F.mse_loss(h_pred, h_actual))
        # h_err_loss = torch.stack(h_err, dim=0).mean()

        fake_loss = torch.tensor(0.).to(DEVICE)
        loss = task_loss

        return loss, task_loss, fake_loss, n_correct, n_q




# class Model(nn.Module):
#     def __init__(self, dim, img_dim):
#         super().__init__()
#         self.dim = dim

#         # image input
#         # norm = nn.BatchNorm2d
#         norm = NullOp

#         drp_p = 0.1
#         drp = nn.Dropout
#         # drp = NullOp
#         self.img_in = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), norm(32), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), norm(32), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), norm(32), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),
#             nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1), norm(1), nn.AdaptiveAvgPool2d((8, 8)), drp(drp_p), nn.GELU(),

#             nn.Flatten(1, -1),
#             nn.Linear(8 ** 2, dim, bias=False),
#         )

#         # label input
#         self.emb = nn.Parameter(torch.randn(n_way, label_dim))
#         torch.nn.init.orthogonal_(self.emb)

#         # backbone
#         self.net1 = SRWM(dim*2, dim)
#         self.net2 = SRWM(dim, dim)

#         # # l1, r1, l2, r2: lor tags
#         # # q: query tag
#         # # serr, qerr: support/query error tag
#         # self.tags = nn.Parameter(torch.randn(7, dim))
#         # torch.nn.init.orthogonal_(self.tags)

#     def net(self, x, w1, w2):
#         x, w1 = self.net1(x, w1)
#         x = F.gelu(x)
#         x, w2 = self.net2(x, w2)
#         return x, w1, w2


#     def forward(self,
#                 query_w,
#                 sxs, sys,  # support xs/ys
#                 qxs, qys,  # query xs/ys
#                 inner_lr):
#         B = query_w.shape[0]
#         S = sxs.shape[1]
#         Q = qxs.shape[1]

#         # embed inputs
#         sxs    = self.img_in(sxs.view(-1, 1, img_dim, img_dim)).view(B, n_way * k_shot, -1)  # [B, S, D]
#         sys_emb = self.emb[sys]

#         # embed outputs
#         qxs = self.img_in(qxs.view(-1, 1, img_dim, img_dim)).view(B, n_way * q_query, -1)  # [B, Q, D]
#         qys_emb = self.emb[qys]

#         w1 = self.net1.w_init.unsqueeze(0).repeat(B, 1, 1)
#         w2 = self.net2.w_init.unsqueeze(0).repeat(B, 1, 1)

#         # Build up LORs based on Supports
#         for six in range(S):
#             # bound = sxs[:, six] + sys_emb[:, six]
#             bound = torch.cat([sxs[:, six], sys_emb[:, six]], dim=1)
#             y, w1, w2 = self.net(bound, w1, w2)

#         # #####
#         # # Metalearning

#         # # Apply LORs
#         # l1s = torch.stack(l1s, dim=1).requires_grad_()
#         # r1s = torch.stack(r1s, dim=1).requires_grad_()
#         # l2s = torch.stack(l2s, dim=1).requires_grad_()
#         # r2s = torch.stack(r2s, dim=1).requires_grad_()

#         # with torch.enable_grad():
#         #     params = [l1s, r1s, l2s, r2s]
#         #     opt_state = opt_state_fn(params)

#         #     for _ in range(N_LOOPS):
#         #         sw1 = w1 + torch.einsum('bsl, bsr -> blr', l1s, r1s)
#         #         sw2 = w2 + torch.einsum('bsl, bsr -> blr', l2s, r2s)

#         #         # Use LORs on support images
#         #         pred_embs = []
#         #         for six in range(S):
#         #             pred = self.net(sxs[:, six] + qe, sw1, sw2)  # query tag on support images
#         #             pred_embs.append(pred)

#         #         # task loss
#         #         pred_embs = torch.stack(pred_embs, dim=1)  # [B, S, D]
#         #         pred_labels = torch.einsum('ld, bsd -> bsl', self.emb, pred_embs);    assert pred_labels.shape == torch.Size([B, sys.shape[1], n_way])
#         #         task_loss = F.cross_entropy(pred_labels.view(-1, n_way),
#         #                                     sys.view(-1))

#         #         grads = torch.autograd.grad(task_loss, params, create_graph=True)

#         #         [l1s, r1s, l2s, r2s], opt_state = opt_fn(params, grads, opt_state, lr=inner_lr)


#         #####
#         # Test

#         pred_embs = []
#         for qix in range(Q):
#             z = torch.zeros(B, D, device=DEVICE)
#             q = torch.cat([qxs[:, qix], z], dim=1)
#             pred, w1, w2 = self.net(q, w1, w2)
#             pred_embs.append(pred)

#         # task loss
#         pred_embs = torch.stack(pred_embs, dim=1)  # [B, Q, D]
#         pred_labels = torch.einsum('ld, bqd -> bql', self.emb, pred_embs);    assert pred_labels.shape == torch.Size([B, qys.shape[1], n_way])
#         task_loss = F.cross_entropy(pred_labels.view(-1, n_way),
#                                     qys.view(-1))

#         n_correct = (pred_labels.view(-1, n_way).argmax(dim=1) == qys.view(-1)).sum()
#         n_q = B * Q

#         # #####
#         # # error prediction
#         # error = pred_embs - qys_emb
#         # error_preds = []
#         # for qix in range(Q):
#         #     bound = qxs[:, qix]  # + qys_emb[:, qix]
#         #     error_preds.append(self.net(bound + qerr, qw1, qw2))
#         # error_pred = torch.stack(error_preds, dim=1)
#         # error_pred_loss = F.mse_loss(error, error_pred)

#         fake_loss = torch.tensor(0.).to(DEVICE)
#         loss = task_loss
#         return loss, task_loss, fake_loss, n_correct, n_q





##################################################

num_epochs = 1000

# INNER_OPT = 'sgd'
INNER_OPT = 'sgd_momentum'
# INNER_OPT = 'rmsprop'
# INNER_OPT = 'adam'
# INNER_OPT = 'adamax'


match INNER_OPT:
    case 'sgd':
        INNER_LR = 1.0  # SGD-momentum needs much higher lr
        N_LOOPS = 20
        opt_state_fn = sgd_init
        opt_fn = sgd
    case 'sgd_momentum':
        INNER_LR = 0.5  # SGD-momentum needs much higher lr
        N_LOOPS = 3
        opt_state_fn = sgd_momentum_init
        opt_fn = sgd_momentum
    case 'rmsprop':
        INNER_LR = 1e-1
        N_LOOPS = 20
        opt_state_fn = rmsprop_init
        opt_fn = rmsprop
    case 'adam':
        INNER_LR = 1e-1
        N_LOOPS = 20
        opt_state_fn = adam_init
        opt_fn = adam
    case 'adamax':
        INNER_LR = 1e-2
        N_LOOPS = 20
        opt_state_fn = adamax_init
        opt_fn = adamax


model = Model(D, img_dim).to(DEVICE)

print_model_info(model)

# parameters = [
#     # model.w_in[0].weight,
#     # model.img_in[1].weight,
#     model.emb,
#     model.w1,
#     # model.w_out.weight,
# ]

parameters = model.parameters()

optimizer = optim.AdamW(parameters, lr=lr, weight_decay=wd)

def run_epoch(model, dataloader, inner_lr, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_task_loss = 0
    total_weight_loss = 0
    total_samples = 0

    total_correct = 0
    total_queries = 0

    with torch.set_grad_enabled(train):
        for batch in dataloader:
            supports, queries = batch
            support_imgs = torch.stack([x[0].to(DEVICE) for x in supports], dim=1)  # [B, NK, IMG**2]
            support_labels = torch.stack([x[1].to(DEVICE) for x in supports], dim=1)  # [B, NK]
            query_imgs = torch.stack([x[0].to(DEVICE) for x in queries], dim=1)  # [B, NK, IMG**2]
            query_labels = torch.stack([x[1].to(DEVICE) for x in queries], dim=1)  # [B, NQ]

            B = support_imgs.shape[0]
            query_w = torch.randn((B, D)).to(DEVICE)

            loss, task_loss, w_loss, n_correct, n_q = model(query_w, support_imgs, support_labels, query_imgs, query_labels, inner_lr)
            total_correct += n_correct
            total_queries += n_q

            if torch.isnan(loss):
                print('NaN encountered:')
                breakpoint()

            if train:
                MAX_GRAD_NORM = 1.0

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()


            total_loss += loss.item() * B
            total_task_loss += task_loss.item() * B
            total_weight_loss += w_loss.item() * B
            total_samples += B

    return (
        total_loss / total_samples,
        total_task_loss / total_samples,
        total_weight_loss / total_samples,
        total_correct / total_queries
    )


train_losses = []
test_losses = []

for epoch in range(num_epochs):
    global_epoch += 1

    # with torch.autograd.detect_anomaly():
    model.train()
    train_loss, train_task_loss, train_weight_loss, train_acc = run_epoch(model, train_dl, INNER_LR, optimizer, DEVICE, train=True)

    train_losses.append((global_epoch, train_loss))

    if epoch % 25 == 0:
        model.eval()
        with torch.no_grad():
            test_loss, test_task_loss, test_weight_loss, test_acc = run_epoch(model, test_dl, INNER_LR, optimizer, DEVICE, train=False)
        test_losses.append((global_epoch, test_loss))
        print(f"Epoch {global_epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:>.2f}, Test Acc: {test_acc:>.2f}")
        print(f'    TRAIN: task_loss: {train_task_loss:>.3f}, w_loss: {train_weight_loss:>.3f}')
        print(f'    TEST:  task_loss: {test_task_loss:>.3f}, w_loss: {test_weight_loss:>.3f}')

    elif epoch % 10 == 0:
        print(f"Epoch {global_epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:>.2f}")


# END_BLOCK_1



weight_1 = model.w1.detach().cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(weight_1)
fig.tight_layout()
plt.show()





#####
# EXPERIMENT: Instead of using a net, or training, just outer pdt imgs and labels, and check cossim of query with target label. Works great!
#
# NOTE: this has to learn a single matrix *across the entire batch* and it still works

if False:

    label_dim = 28 ** 2
    emb = nn.Embedding(n_way, label_dim).to(DEVICE)
    for batch in train_dl:
        supports, queries = batch
        support_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in supports]  # N*k tensors [B, IMG, IMG]
        support_labels = [x[1].to(DEVICE) for x in supports]  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in queries]  # N*k tensors [B, IMG, IMG]
        query_labels = [x[1].to(DEVICE) for x in queries]  # N*k tensors, shape=[B] -> [B, N*k]
        w = torch.zeros(label_dim, img_size**2).to(DEVICE)
        for im, lab in zip(support_imgs, support_labels):
            w += torch.einsum('bd, bl -> ld', im, emb(lab))
        cs = 0
        correct = 0
        for im, lab in zip(query_imgs, query_labels):
            out = torch.einsum('ld, bd -> bl', w, im)
            cs += torch.cosine_similarity(out, emb(lab), dim=1)
            pred = out.argmax(dim=1)
            correct += (pred == lab).sum()
        cs /= len(query_imgs)
        print(cs.mean().item())
        print(correct)

        brk

if False:
    # START_BLOCK_3
    def norm(vectors):
        means = vectors.mean(dim=1, keepdim=True)
        centered = vectors - means
        # stds = torch.sqrt(torch.var(centered, dim=1, keepdim=True) + 1e-8)
        stds = centered.abs().max()
        return centered / stds


    def blur_batch(vectors, kernel):
        # Reshape kernel for batch convolution: [1, 1, kernel_size]
        kernel = kernel.view(1, 1, -1)

        # Add channel dim to input: [batch, 1, vector_length]
        x = vectors.unsqueeze(1)

        # Apply convolution with padding to maintain size
        blurred = F.conv1d(x, kernel, padding=kernel.size(-1) // 2)

        # Remove channel dim
        return blurred.squeeze(1)


    label_dim = 64
    emb = nn.Embedding(n_way, label_dim)
    nn.init.orthogonal_(emb.weight)
    emb = emb.to(DEVICE)
    with torch.no_grad():
        emb.weight[:] = norm(emb.weight)

    for batch in train_dl:
        supports, queries = batch

        # Process support images and labels
        support_imgs = [norm(x[0].to(DEVICE).flatten(start_dim=1)) for x in supports]
        support_labels = [x[1].to(DEVICE) for x in supports]

        # Process query images and labels
        query_imgs = [norm(x[0].to(DEVICE).flatten(start_dim=1)) for x in queries]
        query_labels = [x[1].to(DEVICE) for x in queries]

        B = support_imgs[0].shape[0]

        # Initialize and compute weight matrix
        w = torch.zeros(B, label_dim, img_size**2).to(DEVICE)
        for _ in range(100):
            for im, lab in zip(support_imgs, support_labels):
                lab_emb = emb(lab)
                kernel = torch.rand(3).to(DEVICE)
                w = w + torch.einsum(
                    'bl, bi -> bli',
                    lab_emb,
                    im + blur_batch(im, kernel)
                )

        # Evaluate queries
        correct = 0
        total = 0
        all_similarities = []

        for im, lab in zip(query_imgs, query_labels):
            # Project query through weight matrix
            projected = torch.einsum('bli, bi -> bl', w, im)

            # Calculate similarities with all possible classes
            similarities = torch.einsum('bl, kl -> bk', projected, emb.weight)

            # Get predictions
            pred = similarities.argmax(dim=1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)

            all_similarities.append(similarities)

        # Calculate accuracy
        accuracy = correct / total

        # Calculate loss
        all_similarities = torch.cat(all_similarities, dim=0)
        all_labels = torch.cat(query_labels, dim=0)
        loss = F.cross_entropy(all_similarities, all_labels)

        print(f"Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}")
    # END_BLOCK_3




#####
# Experiment with CONVOLUTION

# START_BLOCK_2

if False:
    emb = nn.Embedding(n_way, label_dim).to(DEVICE)
    for batch in train_dl:
        supports, queries = batch
        # Normalize images first
        support_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in supports]  # N*k tensors [B, IMG*IMG]
        support_labels = [x[1].to(DEVICE) for x in supports]  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in queries]  # N*k tensors [B, IMG*IMG]
        query_labels = [x[1].to(DEVICE) for x in queries]  # N*k tensors, shape=[B] -> [B, N*k]

        B = support_imgs[0].shape[0]
        Q = len(query_labels)

        # Stack all support images and labels
        support_imgs = torch.stack(support_imgs, dim=1)  # [B, N*k, D]
        support_labels = torch.stack(support_labels, dim=1)  # [B, N*k]
        query_imgs = torch.stack(query_imgs, dim=1)  # [B, Q, D]
        query_labels = torch.stack(query_labels, dim=1)  # [B, Q]

        # Get normalized embeddings for all labels
        support_emb = F.normalize(emb(support_labels), dim=-1)  # [B, N*k, D]

        # Compute all support convolutions at once using conjugate for correct correlation

        # fft_support_imgs = torch.fft.fft(support_imgs, dim=-1)  # [B, N*k, D]
        # fft_support_emb = torch.fft.fft(support_emb, dim=-1)  # [B, N*k, D]
        # support_bound = torch.fft.ifft(fft_support_imgs * fft_support_emb, dim=-1).real  # [B, N*k, D]

        # simple mul, instead of convolution
        support_bound = support_imgs * support_emb

        # Normalize the bound vectors
        support_bound = F.normalize(support_bound, dim=-1)

        # Pre-compute query convolutions with all possible class embeddings
        all_classes = torch.arange(n_way).to(DEVICE)
        class_emb = F.normalize(emb(all_classes), dim=-1)  # [N, D]

        # fft_query = torch.fft.fft(query_imgs, dim=-1).unsqueeze(-2)  # [B, Q, 1, D]
        # fft_class_emb = torch.fft.fft(class_emb, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
        # query_bound = torch.fft.ifft(fft_query * fft_class_emb, dim=-1).real  # [B, Q, N, D]

        # simple mul, instead of convolution
        query_bound = (
            query_imgs.unsqueeze(2) *
            class_emb.unsqueeze(0).unsqueeze(0)
        )

        # Normalize the query bound vectors
        query_bound = F.normalize(query_bound, dim=-1)

        # Compute logits for all queries and all possible classes
        logits = torch.zeros((B, Q, n_way)).to(DEVICE)

        # Debug distributions
        class_counts = torch.zeros(n_way)

        # Compute similarities with all support examples
        for class_idx in range(n_way):
            mask = (support_labels == class_idx)  # [B, N*k]
            masked_support = support_bound * mask.unsqueeze(-1)  # [B, N*k, D]
            class_prototype = masked_support.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # [B, D]

            # Normalize the class prototype
            class_prototype = F.normalize(class_prototype, dim=-1)

            # Use the same class index to align with the class embeddings used in query_bound
            sim = torch.einsum(
                'bqnd, bd -> bqn',
                query_bound,  # [B, Q, N, D]
                class_prototype  # [B, D]
            )[:, :, class_idx]  # Take the similarity for the current class

            logits[:, :, class_idx] = sim

        loss = F.cross_entropy(logits.view(-1, n_way), query_labels.view(-1))

        # Compute accuracy
        predictions = logits.argmax(dim=1)  # [B, Q]
        correct = (predictions == query_labels).float()
        accuracy = correct.mean().item()

        # Print per-class prediction distribution
        print()
        for i in range(n_way):
            count = (predictions == i).float().mean().item()
            print(f'Class {i} predicted: {count:.2%}')

        print(f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}')
        break


# END_BLOCK_2














# START_BLOCK_5


# Parameters
batch_size = 2
seq_length = 3
embed_dim = 32
num_heads = 4
head_dim = embed_dim // num_heads  # = 8

# Create example input tensor (batch_size, embedding_dim)
x = torch.randn(batch_size, embed_dim)

# Create weights for Q projection and pre-baked K, V embeddings
W_q = torch.randn(embed_dim, num_heads, head_dim)
# K and V should have dimensions that include the sequence length and head_dim
K = torch.randn(seq_length, num_heads, head_dim)  # Pre-baked K embeddings
V = torch.randn(seq_length, num_heads, head_dim)  # Pre-baked V embeddings

# Project input to Q (batch_size, num_heads, head_dim)
Q = torch.einsum('be,ehd->bhd', x, W_q)

# Expand K and V for batch dimension
# K, V shape: (batch_size, seq_length, num_heads, head_dim)
K = K.unsqueeze(0).expand(batch_size, -1, -1, -1)
V = V.unsqueeze(0).expand(batch_size, -1, -1, -1)

# Scaled dot-product attention
scaling_factor = head_dim ** 0.5
# Q shape: (batch, num_heads, head_dim)
# K shape: (batch, seq_length, num_heads, head_dim)
attention_scores = torch.einsum('bhd,bshd->bhs', Q, K) / scaling_factor
attention_probs = torch.softmax(attention_scores, dim=-1)
# attention_probs shape: (batch, num_heads, seq_length)
# V shape: (batch, seq_length, num_heads, head_dim)
attention_output = torch.einsum('bhs,bshd->bhd', attention_probs, V)

# Reshape from (batch, num_heads, head_dim) to (batch, num_heads * head_dim)
concat_output = attention_output.reshape(batch_size, num_heads * head_dim)

# Create output projection weight
W_o = torch.randn(embed_dim, embed_dim)

# Final projection
final_output = torch.einsum('bc,ce->be', concat_output, W_o)

print("\nShape check:")
print(f"Q shape: {Q.shape}")  # [batch_size, num_heads, head_dim]
print(f"K shape: {K.shape}")  # [batch_size, seq_length, num_heads, head_dim]
print(f"V shape: {V.shape}")  # [batch_size, seq_length, num_heads, head_dim]
print(f"Attention scores shape: {attention_scores.shape}")  # [batch_size, num_heads, seq_length]
print(f"Attention output shape (before concat): {attention_output.shape}")  # [batch_size, num_heads, head_dim]
print(f"Concatenated shape: {concat_output.shape}")  # [batch_size, embed_dim]
print(f"Final output shape: {final_output.shape}")  # [batch_size, embed_dim]

# END_BLOCK_5
