"""https://pytorch.org/tutorials/intermediate/per_sample_grads.html"""

import torch
from torch import Tensor
from copy import deepcopy
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as roc_auc
from unlearning.net import Net
from unlearning.controller import Controller


class Model:
    def __init__(self, epochs: int, controller_epochs: int = 100):
        self.net = None
        self.epochs = epochs
        self.controller_epochs = controller_epochs
        self.controller = None
        self.net = None
        self.w0 = None

    def fit(self, x: Tensor, y: Tensor, x_test: Tensor, y_test: Tensor):
        n_in = x.size(1)
        n_out = y.size(1) if y.ndim > 1 else 1
        net = Net(n_in, n_out)
        self.w0 = deepcopy(net.layer.weight.detach())
        controller = Controller(n_in + 1, n_in)
        optimizer = Adam(params=net.parameters())
        c_optimizer = Adam(params=controller.parameters())
        loss_fn = BCEWithLogitsLoss(reduction='none')
        c_loss_fn = MSELoss()
        eval_loss = float('inf')
        stat = {
            'epoch': [],
            'roc_auc': [],
            'train_loss': [],
            'eval_loss': [],
            'controller_epochs': [],
            'controller_train_mse': [],
            'controller_eval_mse': []
        }
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
            external_grad = sample_grads.mean(dim=0)
            train_loss = loss.mean()
            train_loss.backward()
            ref_grads = net.layer.weight.grad
            ref = (external_grad-ref_grads).mean().item()
            assert ref < 1e-7, f"{ref}"
            optimizer.step()

            net.eval()
            e_y_hat = net(x_test)
            _eval_loss = loss_fn(e_y_hat, y_test).mean()
            auc = roc_auc(y_test.cpu().numpy(), e_y_hat.detach().cpu().numpy())
            stat['epoch'].append(e)
            stat['roc_auc'].append(auc)
            stat['train_loss'].append(train_loss.item())
            stat['eval_loss'].append(_eval_loss.item())
            if _eval_loss < eval_loss:
                eval_loss = _eval_loss
                self.net = deepcopy(net)

            # train the controller
            c_features = self.grad_features(x, e)

            c_train, c_test, g_train, g_test = train_test_split(c_features, sample_grads.detach())

            controller_loss_diff = float('inf')
            patience = 10
            report_c_epoch = self.controller_epochs
            for ce in range(self.controller_epochs):
                controller.train()
                c_optimizer.zero_grad()
                g_hat = controller(c_train)
                c_train_loss = c_loss_fn(g_hat, g_train)
                c_train_loss.backward()
                c_optimizer.step()

                controller.eval()
                e_g_hat = controller(c_test)
                c_eval_loss = c_loss_fn(e_g_hat, g_test)
                c_loss_diff = torch.abs(c_eval_loss - c_train_loss).item()
                if c_loss_diff < controller_loss_diff:
                    controller_loss_diff = c_loss_diff
                else:
                    patience -= 1

                if patience == 0:
                    report_c_epoch = ce
                    break
            stat['controller_epochs'] = report_c_epoch
            stat['controller_train_mse'].append(c_train_loss.item())
            stat['controller_eval_mse'].append(c_eval_loss.item())
        self.controller = deepcopy(controller)
        return stat

    def grad_features(self, x: Tensor, epoch: int) -> Tensor:
        return torch.concat([torch.ones(size=(len(x), 1)) * epoch / self.epochs, x], dim=1)

    def predict(self, x: Tensor):
        self.net.eval()
        return self.net(x).cpu().detach().numpy()

    def predict_grad(self, x: Tensor):
        self.controller.eval()
        return self.controller(x).cpu().detach()

