import torch
from torch.nn import Parameter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

from unlearning.data import prepare
from unlearning.grad_per_sample import Model

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)


def train(train_df, test_df, poisson_idx):
    scaler = StandardScaler()
    train_f = scaler.fit_transform(train_df.iloc[:, :-1].values)
    # train_f = train_f[~poisson_idx]
    train_t = train_df.target.values
    # train_t = train_t[~poisson_idx]

    test_f = scaler.transform(test_df.iloc[:, :-1].values)
    test_t = test_df.target.values
    train_features = torch.tensor(train_f, dtype=torch.float32)
    train_targets = torch.tensor(train_t, dtype=torch.float32).unsqueeze(dim=1)
    test_features = torch.tensor(test_f, dtype=torch.float32)
    test_targets = torch.tensor(test_t, dtype=torch.float32).unsqueeze(dim=1)
    n_epochs = 150
    model = Model(n_epochs)

    stats = model.fit(train_features, train_targets, test_features, test_targets)
    y_hat = model.predict(test_features)
    roc = roc_auc_score(test_t, y_hat)
    stats['roc'] = roc
    return stats


def main(train_df, test_df):

    scaler = StandardScaler()
    train_f = scaler.fit_transform(train_df.iloc[:, :-1].values)
    train_t = train_df.target.values

    poisson_idx = np.random.choice(range(len(train_f)), size=(20, ), replace=False)
    train_t[poisson_idx] = 1 - train_t[poisson_idx]

    test_f = scaler.transform(test_df.iloc[:, :-1].values)
    test_t = test_df.target.values
    train_features = torch.tensor(train_f, dtype=torch.float32)
    train_targets = torch.tensor(train_t, dtype=torch.float32).unsqueeze(dim=1)
    test_features = torch.tensor(test_f, dtype=torch.float32)
    test_targets = torch.tensor(test_t, dtype=torch.float32).unsqueeze(dim=1)
    n_epochs = 150
    model = Model(n_epochs)

    stats = model.fit(train_features, train_targets, test_features, test_targets)
    y_hat = model.predict(test_features)
    roc = roc_auc_score(test_t, y_hat)
    stats['roc'] = roc

    # Cure the model / unlearn
    poisson_features = train_features[poisson_idx]
    grad = torch.zeros(size=(1, 4))
    for e in range(n_epochs):
        features = model.grad_features(poisson_features, e)
        grad_per_sample = model.predict_grad(features)
        grad += grad_per_sample.mean(dim=0)
    lr = -0.003
    s = len(train_features)
    p = len(poisson_features)
    # print(model.net.layer.weight - model.w0)
    model.net.layer.weight = Parameter((s * (model.net.layer.weight - model.w0) - lr * grad) / (s-p) + model.w0)  # why plus
    y_hat = model.predict(test_features)
    roc2 = roc_auc_score(test_t, y_hat)
    stats['roc_cured'] = roc2

    return stats, poisson_idx


def plot_learning_curves(stat: dict):
    plt.title(f"AUC {stat['roc']}")
    plt.plot(stat['train_loss'], label='train_loss')
    plt.plot(stat['eval_loss'], label='eval_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_df, test_df = prepare()
    stat, pi = main(train_df, test_df)
    stat1 = train(train_df, test_df, pi)
    df = pd.DataFrame(stat)
    # print(df)
    # print(f"AUC {stat['roc_auc'][-1]}")
    print(f"AUC no poisson {stat1['roc']}")
    print(f"AUC poisson {stat['roc']}")
    print(f"AUC healed {stat['roc_cured']}")
    # imp is a sample importance
    # plot_learning_curves(stat)


