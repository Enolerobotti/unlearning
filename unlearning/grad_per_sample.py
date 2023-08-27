"""https://pytorch.org/tutorials/intermediate/per_sample_grads.html"""

import torch
from torch import Tensor
from copy import deepcopy
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from unlearning.net import Net
from unlearning.controller import Controller


class Model:
    def __init__(self, epochs: int, controller_epochs: int = 10):
        self.net = None
        self.epochs = epochs
        self.controller_epochs = controller_epochs
        self.controller = None
        self.net = None

    def fit(self, x: Tensor, y: Tensor, x_test: Tensor, y_test: Tensor):
        n_in = x.size(1)
        n_out = y.size(1) if y.ndim > 1 else 1
        net = Net(n_in, n_out)
        controller = Controller(n_in + 1, n_in)
        optimizer = Adam(params=net.parameters())
        c_optimizer = Adam(params=controller.parameters())
        loss_fn = BCEWithLogitsLoss(reduction='none')
        c_loss_fn = MSELoss()
        eval_loss = float('inf')
        for e in range(self.epochs):
            net.train()
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

            net.eval()
            e_y_hat = net(x_test)
            _eval_loss = loss_fn(e_y_hat, y_test)
            if _eval_loss < eval_loss:
                eval_loss = _eval_loss
                self.net = deepcopy(net)

            # train the controller
            c_features = torch.concat([torch.ones(size=(len(x), 1)) * e / self.epochs, x], dim=1)

            c_train, c_test, g_train, g_test = train_test_split(c_features, sample_grads.detach())

            controller.train()
            for ce in range(self.controller_epochs):
                c_optimizer.zero_grad()
                g_hat = controller(c_train)
                c_loss = c_loss_fn(g_hat, g_train)
                c_loss.backward()
                c_optimizer.step()

            controller.eval()
            e_g_hat = controller(c_test)
            c_eval_loss = c_loss_fn(e_g_hat, g_test)
            self.controller = deepcopy(controller)
            # print(c_eval_loss.item())

    def predict(self, x: Tensor):
        self.net.eval()
        return self.net(x).cpu().detach().numpy()

    def predict_grad(self, x: Tensor):
        self.controller.eval()
        return self.controller(x).cpu().detach()

