import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from unlearning.net import Net

class Model:
    def __init__(self, epochs: int):
        self.net = None
        self.epochs = epochs

        self.importance = None

    def fit(self, x, y, *args):
        n_in = x.size(1)
        n_out = y.size(1) if y.ndim > 1 else 1
        net = Net(n_in, n_out)
        optimizer = Adam(params=net.parameters())
        loss_fn = BCEWithLogitsLoss(reduction='none')
        for e in range(self.epochs):
            optimizer.zero_grad()
            _sample_grads = []
            y_hat = net(x)
            loss = loss_fn(y_hat, y).mean(dim=1)
            for i in range(len(x)):
                sg = torch.autograd.grad(loss[i], list(net.parameters()), retain_graph=True)
                _sample_grads.append(sg[0])

            sample_grads = torch.concat(_sample_grads)
            external_grad = sample_grads.mean()
            loss.mean().backward(gradient=external_grad)
            optimizer.step()

    def predict(self,*args):
        return

