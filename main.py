import torch
from torch import Tensor
from copy import deepcopy
from torch.nn import Linear, BCEWithLogitsLoss, Module
from torch.optim import Adam
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class Net(Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.layer = Linear(n_in, n_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class Model:
    def __init__(self, epochs: int):
        self.net = None
        self.epochs = epochs

    def fit(self, x: Tensor, y: Tensor, x_test: Tensor, y_test: Tensor):
        n_in = x.size(1)
        n_out = y.size(1) if y.ndim > 1 else 1
        net = Net(n_in, n_out)
        loss_f = BCEWithLogitsLoss()
        optimizer = Adam(params=net.parameters())
        _loss = float('inf')
        stat = {'train_loss': [], 'eval_loss': []}
        for e in range(self.epochs):
            net.train()
            optimizer.zero_grad()
            y_hat = net(x)
            loss = loss_f(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            eval_loss = self.eval(net, x_test, y_test)
            # print(f"epoch {e}, train_loss {train_loss:.3f}, eval loss {eval_loss:.3f}")
            stat['train_loss'].append(train_loss)
            stat['eval_loss'].append(eval_loss)
            if train_loss < _loss:
                _loss = train_loss
                self.net = deepcopy(net)
                # print("weight updated")
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


def main():
    df = load_iris(as_frame=True)
    data = df.frame
    data = data[data.target != 2]
    train_df, test_df = train_test_split(data)
    scaler = StandardScaler()
    train_f = scaler.fit_transform(train_df.iloc[:, :-1].values)
    train_t = train_df.target.values
    test_f = scaler.transform(test_df.iloc[:, :-1].values)
    test_t = test_df.target.values
    train_features = torch.tensor(train_f, dtype=torch.float32)
    train_targets = torch.tensor(train_t, dtype=torch.float32).unsqueeze(dim=1)
    test_features = torch.tensor(test_f, dtype=torch.float32)
    test_targets = torch.tensor(test_t, dtype=torch.float32).unsqueeze(dim=1)
    model = Model(5000)
    stats = model.fit(train_features, train_targets, test_features, test_targets)
    y_hat = model.predict(test_features)
    roc = roc_auc_score(test_t, y_hat)
    stats['roc'] = roc
    return stats


def plot_learning_curves(stat: dict):
    plt.title(f"AUC {stat['roc']}")
    plt.plot(stat['train_loss'], label='train_loss')
    plt.plot(stat['eval_loss'], label='eval_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    stat = main()
    print(f"AUC {stat['roc']}")
    plot_learning_curves(stat)

