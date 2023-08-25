from copy import deepcopy

import numpy as np
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from unlearning.net import Net


class Model:
    def __init__(self, epochs: int):
        self.net = None
        self.epochs = epochs
        self.container = []
        self.importance = []
        self.track_nonlineariry_effect = {
            'mean_importance': [],
            'loss': [],
            'abs_diff': []
        }

    def hook(self, grad):
        self.container.append(grad.detach().cpu().numpy()[0])
        return grad

    def fit(self, x: Tensor, y: Tensor, x_test: Tensor, y_test: Tensor):
        n_in = x.size(1)
        n_out = y.size(1) if y.ndim > 1 else 1
        net = Net(n_in, n_out)
        net.layer.weight.register_hook(self.hook)
        # net.layer.bias.register_hook(hook)
        loss_f = BCEWithLogitsLoss(reduction='none')
        optimizer = Adam(params=net.parameters())
        _loss = float('inf')
        stat = {'train_loss': [], 'eval_loss': []}
        for e in range(self.epochs):
            net.train()
            optimizer.zero_grad()
            y_hat = net(x)
            loss_unr = loss_f(y_hat, y)

            for lo in loss_unr:
                lo.backward(retain_graph=True)
                optimizer.zero_grad()

            importance = np.array(self.container).mean(axis=1)
            mean_importance = importance.mean()
            self.track_nonlineariry_effect['mean_importance'].append(mean_importance)
            self.importance.append(importance)

            loss = torch.mean(loss_unr)
            loss.backward()

            mean_loss = self.container[-1].mean()
            self.track_nonlineariry_effect['loss'].append(mean_loss)
            self.track_nonlineariry_effect['abs_diff'].append(np.abs(mean_loss-mean_importance))


            optimizer.step()
            self.container = []
            train_loss = loss.item()
            eval_loss = self.eval(net, x_test, y_test)
            # print(f"epoch {e}, train_loss {train_loss:.3f}, eval loss {eval_loss:.3f}")
            stat['train_loss'].append(train_loss)
            stat['eval_loss'].append(eval_loss)
            if train_loss < _loss:
                _loss = train_loss
                self.net = deepcopy(net)
                # print("weight updated")
        self.importance = np.array(self.importance)
        return stat

    def eval(self, net: Net, x: Tensor, y: Tensor):
        net.eval()
        loss_f = BCEWithLogitsLoss()
        y_hat = net(x)
        loss = loss_f(y_hat, y)
        return loss.item()

    def predict(self, x: Tensor):
        self.net.eval()
        y_hat = torch.sigmoid(self.net(x))
        return y_hat.detach().cpu().numpy()
