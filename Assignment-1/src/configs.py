"""
This file contains the configuration for the project.
"""

PROJECT_NAME = 'mlp_cifar10_pytorch'
PROJECT_ENTITY = 'cs20b013-bersilin'

# Labels for the CIFAR-10 dataset

LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

configs = {
    'batch_size': 100,
    'learning_rate': 0.07,
    'num_epochs': 10,
    'momentum': 0.9,

    'wandb_log': False,
    'batch_norm': False
}