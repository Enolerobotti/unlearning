import torch
from torch.nn import Parameter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# from unlearning.classic_model import Model
from unlearning.grad_per_sample import Model

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def main():
    df = load_iris(as_frame=True)
    data = df.frame
    data = data[data.target != 2]

    train_df, test_df = train_test_split(data)

    scaler = StandardScaler()
    train_f = scaler.fit_transform(train_df.iloc[:, :-1].values)
    train_t = train_df.target.values

    poisson_idx = np.random.choice(range(len(train_f)), size=(40, ), replace=False)
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
        # model.net.layer.weight = Parameter(model.net.layer.weight - grad * 0.003)  # TODO this is not I was gonna do. Use my formula instead
    lr = 0.003
    s = len(train_features)
    p = len(poisson_features)
    model.net.layer.weight = Parameter((s * (model.net.layer.weight - model.w0) - lr * grad) / (s-p) + model.w0)
    y_hat = model.predict(test_features)
    roc2 = roc_auc_score(test_t, y_hat)
    stats['roc_cured'] = roc2


    return stats, None


def plot_learning_curves(stat: dict):
    plt.title(f"AUC {stat['roc']}")
    plt.plot(stat['train_loss'], label='train_loss')
    plt.plot(stat['eval_loss'], label='eval_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    stat, imp = main()
    df = pd.DataFrame(stat)
    # print(df)
    # print(f"AUC {stat['roc_auc'][-1]}")
    print(f"AUC {stat['roc']}")
    print(f"AUC {stat['roc_cured']}")
    # imp is a sample importance
    # plot_learning_curves(stat)


