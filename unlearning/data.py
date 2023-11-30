from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def prepare(*args, **kwargs):
    df = load_iris(as_frame=True)
    data = df.frame
    data = data[data.target != 2]

    return train_test_split(data)


def prepare_CIFAR10(
        train_set_ids: np.ndarray, test_set_ids: np.ndarray, poisoned_ids: np.ndarray, batch_size: int):
    """
    :param train_set_ids: (np.array[int]) of train set samples in both original and poisoned data sets
    :param test_set_ids: (np.array[int]) of test set samples
    :param poisoned_ids: (np.array[int]) subset of poisoned train set pointing poisoned samples
    :param batch_size: (int) batch size
    :return: 4-tuple of dataloaders
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    trainset = Subset(trainset, train_set_ids)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    poisoned_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    targets = np.array(poisoned_set.targets)
    targets[poisoned_ids] = 9 - targets[poisoned_ids]
    poisoned_set.targets = targets.tolist()
    poisoned_set = Subset(poisoned_set, train_set_ids)
    poisoned_samples = Subset(poisoned_set, poisoned_ids)
    poisoned_trainloader = DataLoader(poisoned_set, batch_size=batch_size, shuffle=True, num_workers=0)
    healing_loader = DataLoader(poisoned_samples, batch_size=batch_size, shuffle=False, num_workers=0)

    testset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    testset = Subset(testset, test_set_ids)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader, poisoned_trainloader, healing_loader
