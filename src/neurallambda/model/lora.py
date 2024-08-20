'''

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Linear(nn.Module):
    ''' A Low Rank Linear layer. This is not particularly intended as an adapter. '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super(Linear).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.has_bias = bias

        self.in_proj = nn.Parameter(torch.empty((in_features, rank)))
        self.out_proj = nn.Parameter(torch.empty((rank, out_features)))

        # Initialize params
        nn.init.kaiming_uniform_(self.in_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj, a=math.sqrt(5))
        # nn.init.zeros_(self.out_proj)  # since this gets trained from the get-go, not as an adapter, should it get a random init?

        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}, bias={self.has_bias})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.in_proj
        x = x @ self.out_proj * self.alpha / self.rank
        if self.has_bias:
            x = x + self.bias
        return x

    @property
    def weight(self) -> torch.Tensor:
        return torch.einsum("i r, r o -> o i", self.in_proj, self.out_proj)
