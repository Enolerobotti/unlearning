import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from unlearning.classic_model import Model

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
    test_f = scaler.transform(test_df.iloc[:, :-1].values)
    test_t = test_df.target.values
    train_features = torch.tensor(train_f, dtype=torch.float32)
    train_targets = torch.tensor(train_t, dtype=torch.float32).unsqueeze(dim=1)
    test_features = torch.tensor(test_f, dtype=torch.float32)
    test_targets = torch.tensor(test_t, dtype=torch.float32).unsqueeze(dim=1)
    model = Model(500)
    stats = model.fit(train_features, train_targets, test_features, test_targets)
    y_hat = model.predict(test_features)
    roc = roc_auc_score(test_t, y_hat)
    stats['roc'] = roc
    importance = model.importance
    grouth_rate = np.diff(importance, axis=0)
    return stats, importance


def plot_learning_curves(stat: dict):
    plt.title(f"AUC {stat['roc']}")
    plt.plot(stat['train_loss'], label='train_loss')
    plt.plot(stat['eval_loss'], label='eval_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    stat, imp = main()
    print(f"AUC {stat['roc']}")
    # imp is a sample importance
    # plot_learning_curves(stat)

    # TODO see https://pytorch.org/tutorials/intermediate/per_sample_grads.html

