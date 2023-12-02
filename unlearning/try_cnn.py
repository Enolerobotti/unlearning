import torch
from torch.nn import Parameter
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce

from unlearning.data import prepare_CIFAR10 as prepare
from unlearning.cnn_model import CNNModel as Model

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def train(train_dataloader, test_dataloader, n_epochs: int, c_epochs: int):
    model = Model(n_epochs, c_epochs)
    stats = model.fit(train_dataloader, test_dataloader)
    return stats


def main(poisoned_dataloader, test_dataloader, healing_loader, n_epochs: int, c_epochs: int):
    model = Model(n_epochs, c_epochs)

    stats = model.fit(poisoned_dataloader, test_dataloader)

    # Cure the model / unlearn
    grad = torch.zeros_like(list(model.controller.parameters())[-1])  # Last layer size
    for e in range(n_epochs):
        for i, batch in enumerate(healing_loader):
            poisoned_features, _ = batch
            features = model.grad_features(poisoned_features, e)
            grad_per_sample = model.predict_grad(features)
            grad += grad_per_sample.mean(0)

    lr = -0.003  # default Adam's lr
    s = len(poisoned_dataloader.dataset)  # len(train_features)
    p = len(healing_loader.dataset)  # len(poisoned_features)
    params = list(model.net.parameters())
    _params = torch.concat([torch.flatten(p) for p in params])
    w0 = torch.concat([torch.flatten(p) for p in model.w0])
    _params = (s * (_params - w0) - lr * grad) / (s-p) + w0
    cursor = 0
    state_dict = model.net.state_dict()
    for n, p in state_dict.items():
        length = reduce(lambda x, y: x * y, p.shape, 1)
        chunk = _params[cursor:cursor+length]
        cursor += length
        state_dict[n] = Parameter(chunk.view(p.shape))
    model.net.load_state_dict(state_dict)
    eval_loss = model.eval_model(model.net, torch.nn.CrossEntropyLoss(reduction='none'), test_dataloader)
    stats['eval_loss_cured'] = eval_loss

    return stats


def plot_learning_curves(stat: dict):
    plt.title(f"Cross Entropy")
    plt.plot(stat['train_loss'], label='train_loss')
    plt.plot(stat['eval_loss'], label='eval_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    seed = 42
    n_epochs = 10
    controller_epochs = 3  # actually it is n_epochs * controller_epochs
    train_set_size = 400
    test_set_size = 300
    number_of_poisoned_samples = 100
    batch_size = 200

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_set_ids = np.arange(train_set_size)
    train_dataloader, test_dataloader, poison_dataloader, healing_loader = prepare(
        train_set_ids=train_set_ids,
        test_set_ids=np.arange(test_set_size),
        poisoned_ids=np.random.choice(train_set_ids, (number_of_poisoned_samples, ), replace=False),
        batch_size=batch_size)

    stat = main(poison_dataloader, test_dataloader, healing_loader, n_epochs=n_epochs, c_epochs=controller_epochs)
    stat1 = train(train_dataloader, test_dataloader, n_epochs=n_epochs, c_epochs=controller_epochs)
    print(f"CE no poison {stat1['eval_loss']}")
    print(f"CE poison {stat['eval_loss']}")
    print("######################################3")
    print(f"CE no poison {stat1['eval_loss'][-1]:.3f}")
    print(f"CE poison {stat['eval_loss'][-1]:.3f}")
    print(f"CE healed {stat['eval_loss_cured']:.3f}")
    plot_learning_curves(stat)


