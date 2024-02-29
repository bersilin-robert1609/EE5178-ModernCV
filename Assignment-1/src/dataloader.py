import torch
from torchvision.transforms import v2
import torch.utils.data as dataloader
import numpy as np
import torchvision.datasets as datasets

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Loading the CIFAR-10 dataset
def load_data(transform: v2.Compose):
    '''
    Load the CIFAR-10 dataset
    '''
    train_data = datasets.CIFAR10(root='./data', 
                                  train=True,
                                  download=True, 
                                  transform=transform)

    test_data = datasets.CIFAR10(root='./data',
                                 train=False,
                                 download=True,
                                 transform=transform)

    return train_data, test_data


# Split the training set into a training and validation set
def val_split(train_data: datasets, split=0.2, shuffle=True):
    '''
    Split the training set into a training and validation set

    Args:
    train_set: the training set
    split: the proportion of the validation set
    shuffle: whether to shuffle the indices before splitting
    '''
    train_size = len(train_data)
    indices = list(range(train_size))
    split = int(np.floor(split * train_size))

    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_set = dataloader.Subset(train_data, train_indices)
    val_set = dataloader.Subset(train_data, val_indices)

    return train_set, val_set


# Create a dataloader
def create_dataloader(train_set: datasets, val_set: datasets, test_set: datasets, batch_size: int):
    '''
    Create a dataloader for the training and test sets
    '''
    train_loader = dataloader.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = dataloader.DataLoader(val_set, batch_size=5 * batch_size, shuffle=False)
    test_loader = dataloader.DataLoader(test_set, batch_size=5 * batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
