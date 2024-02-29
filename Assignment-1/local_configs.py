from torchvision.transforms import v2
import torch

configs = {
    'batch_size': 100,
    'learning_rate': 0.07,
    'num_epochs': 10,
    'momentum': 0.9,

    'wandb_log': False,
    'batch_norm': False
}

train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])