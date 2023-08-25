from torch import Tensor
from torch.nn import Module, Linear


class Net(Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.layer = Linear(n_in, n_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
