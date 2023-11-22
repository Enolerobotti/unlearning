from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def prepare():
    df = load_iris(as_frame=True)
    data = df.frame
    data = data[data.target != 2]

    return train_test_split(data)
