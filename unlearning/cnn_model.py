import torch
from torch import Tensor
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn import CrossEntropyLoss, MSELoss, Module, Sequential
from torch.optim import Adam, Optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as roc_auc
from unlearning.net import CNN as Net, CNNEncoder
from unlearning.controller import Controller


class CNNModel:
    def __init__(self, epochs: int, controller_epochs: int):
        self.epochs = epochs
        self.controller_epochs = controller_epochs
        self.controller = None
        self.net = None
        self.w0 = None  # initial parameter values

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader):
        net = Net()
        self.w0 = deepcopy([param.detach() for param in net.parameters()])
        n_in = len(torch.concat([torch.flatten(w0) for w0 in self.w0]))
        controller = Sequential(
            CNNEncoder(num_channels=4),  # an additional channel for epoch feature
            Controller(net.encoder.out_features, n_in))
        optimizer = Adam(params=net.parameters())
        c_optimizer = Adam(params=controller.parameters())
        loss_fn = CrossEntropyLoss(reduction='none')
        c_loss_fn = MSELoss()
        eval_loss = float('inf')
        stat = {
            'epoch': [],
            'train_loss': [],
            'eval_loss': [],
            'controller_epochs': [],
            'controller_train_mse': [],
            'controller_eval_mse': []
        }
        for e in range(self.epochs):
            net.train()
            train_loss, c_train_loss, c_eval_loss, controller_epochs = self.train_an_epoch(
                optimizer=optimizer, c_optimizer=c_optimizer,
                net=net, controller=controller,
                loss_fn=loss_fn, c_loss_fn=c_loss_fn, train_dataloader=train_dataloader, epoch_number=e)

            net.eval()
            _eval_loss = self.eval_model(net, loss_fn, test_dataloader)
            stat['epoch'].append(e)
            stat['train_loss'].append(train_loss)
            stat['eval_loss'].append(_eval_loss)
            if _eval_loss < eval_loss:
                eval_loss = _eval_loss
                self.net = deepcopy(net)

            stat['controller_epochs'] = controller_epochs
            stat['controller_train_mse'].append(c_train_loss)
            stat['controller_eval_mse'].append(c_eval_loss)
        self.controller = deepcopy(controller)
        return stat

    def train_an_epoch(
            self,
            optimizer: Optimizer,
            c_optimizer: Optimizer,
            net: Module,
            controller: Module,
            loss_fn: Module,
            c_loss_fn: Module,
            train_dataloader: DataLoader,
            epoch_number: int
    ):
        optimizer.zero_grad()
        running_train_loss = 0.0
        running_c_train_loss = 0.0
        running_c_eval_loss = 0.0
        ce = []
        i = 1
        for i, batch in enumerate(train_dataloader):
            features, targets = batch
            y_hat = net(features)
            loss = loss_fn(y_hat, targets)
            # Flatten and concat all the gradients. The first dim should be a batch size
            sample_grads = torch.stack([
                torch.concat([torch.flatten(t) for t in torch.autograd.grad(_loss, list(net.parameters()), retain_graph=True)])
                for _loss in loss])
            train_loss = loss.mean()
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            c_train_loss, c_eval_loss, report_c_epoch = self.train_controller(
                c_optimizer, controller, c_loss_fn, features, sample_grads, epoch_number)
            running_c_train_loss += c_train_loss
            running_c_eval_loss += c_eval_loss
            ce.append(report_c_epoch)
        return running_train_loss / i, running_c_train_loss / i, running_c_eval_loss / i, ce

    def eval_model(self, net: Module, loss_fn: Module, test_dataloader: DataLoader):
        i = 1.0
        eval_loss = 0.0
        for i, batch in enumerate(test_dataloader):
            features, targets = batch
            y_hat = torch.sigmoid(net(features))
            eval_loss += loss_fn(y_hat, targets).mean().item()
        return eval_loss / i

    def train_controller(
            self, c_optimizer: Optimizer, controller: Module, c_loss_fn: Module, features: Tensor,
            sample_grads: Tensor, epoch_number: int):

        c_features = self.grad_features(features, epoch_number)

        c_train, c_test, g_train, g_test = train_test_split(c_features, sample_grads.detach())

        controller_loss_diff = float('inf')
        patience = 10
        c_train_loss, c_eval_loss = None, None
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

        return c_train_loss.item(), c_eval_loss.item(), report_c_epoch

    def grad_features(self, x: Tensor, epoch: int) -> Tensor:
        return torch.concat([torch.ones(size=(len(x), 1, 32, 32)) * epoch / self.epochs, x], dim=1)

    def predict(self, x: Tensor):
        self.net.eval()
        return self.net(x).cpu().detach().numpy()

    def predict_grad(self, x: Tensor):
        self.controller.eval()
        return self.controller(x).cpu().detach()
