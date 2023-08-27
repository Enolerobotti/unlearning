import torch
from torch import Tensor
from torch.nn import Linear, Module, Sequential, ReLU
from .net import Net


class Controller(Module):
    def __init__(self, n_in: int, n_out: int, n_hidden=300):
        super().__init__()
        self.mlp = Sequential(
            Linear(n_in, n_hidden),
            ReLU(),
            Linear(n_hidden, n_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
