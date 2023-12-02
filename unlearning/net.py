from torch import Tensor, flatten, relu
from torch.nn import Module, Linear, Conv2d, MaxPool2d


class Net(Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.layer = Linear(n_in, n_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class CNNEncoder(Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = Conv2d(num_channels, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.out_features = 16 * 5 * 5

    def forward(self, x):
        # samples x 32 x 32 x 3
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        return flatten(x, 1)  # flatten all dimensions except batch


class CNN(Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.encoder = CNNEncoder(num_channels)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        # samples x 32 x 32 x 3
        x = self.encoder(x)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)
