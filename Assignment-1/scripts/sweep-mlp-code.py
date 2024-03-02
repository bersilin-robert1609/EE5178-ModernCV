import torch
import torchvision.datasets as datasets
from torchvision.transforms import v2
import torch.utils.data as dataloader
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import wandb

PROJECT_NAME = 'mlp_cifar10_pytorch'
PROJECT_ENTITY = 'cs20b013-bersilin'

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

ARCH = [500, 250, 100]
DATA_DIR = "../data"


def get_transform(mean, std):
    '''
    Returns a transform to convert a CIFAR image to a tensor of type float32
    '''
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])


def get_dataloader(batch_size: int, val_split: float = 0.2, shuffle: bool = True):
    '''
    Load the CIFAR-10 dataset
    '''
    train_data = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    test_data = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    mean = np.array(train_data.data).mean(axis=(0, 1, 2)) / 255
    std = np.array(train_data.data).std(axis=(0, 1, 2)) / 255

    transform = get_transform(mean, std)
    train_data.transform = transform
    test_data.transform = transform

    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size

    train_data, val_data = dataloader.random_split(train_data, [train_size, val_size])

    train_loader = dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_loader = dataloader.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = dataloader.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_data, test_data, train_loader, val_loader, test_loader


def show_random_image(dataset: datasets.CIFAR10, index: int = None):
    '''
    Shows a random image from the dataset
    '''
    if index is None:
        index = np.random.randint(0, len(dataset))
    else:
        index = index
                
    image, label = dataset[index]
    
    plot = plt.imshow(image.permute(1, 2, 0))
    plt.title("True Label:", LABELS[label])

    return plot, index, label


def plot_accuracies(train_acc, val_acc):
    '''
    Plot the training and validation accuracies
    '''
    plot = plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    return plot


# Architecture of the model

class MLP(nn.Module):
    '''
    Multi-layer perceptron model

    Activation function: ReLU
    Output activation function: Softmax
    '''
    def __init__(self, arch, in_size, out_size):
        super(MLP, self).__init__()

        self.sequence  = self.get_layers(arch, in_size, out_size)
        self.fc = nn.Sequential(*self.sequence)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
    def get_layers(self, arch, in_size, out_size):
        '''
        Returns a list of layers for the model
        '''
        layers = []
        layers.append(nn.Linear(in_features=in_size, out_features=arch[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(arch)):
            layers.append(nn.Linear(in_features=arch[i-1], out_features=arch[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=arch[-1], out_features=out_size))

        return layers


def get_accuracy(model: nn.Module, data_loader: dataloader.DataLoader, device: torch.device):
    '''
    Get the accuracy of the model on the data_loader
    '''
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total


def get_predicted_labels(model: nn.Module, data_loader: dataloader.DataLoader, device: torch.device):
    '''
    Get the predicted labels of the model on the data_loader
    '''
    labels = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            _, predicted = torch.max(preds, 1)
            labels.append(predicted)

    return torch.cat(labels)


# Training the model

def train(configs, train_loader: dataloader.DataLoader, val_loader: dataloader.DataLoader, criterion: nn.CrossEntropyLoss,
          optimizer: optim.Optimizer, model: nn.Module, device: torch.device):
    '''
    Train the model
    '''
    
    print('Training the model...')
    print('---------------------')

    val_accuracies, train_accuracies = [], []

    for epoch in range(configs['num_epochs']):
        model.train()
        running_loss = 0.0

        total_iterations = len(train_loader)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            running_loss += loss.item()

            if (i != total_iterations-1):
                print(f'Epoch {epoch + 1}, Iteration {i + 1}/{total_iterations}, Loss: {loss.item()}', end='\r')
            else:
                print(f'Epoch {epoch + 1}, Iteration {i + 1}/{total_iterations}, Loss: {loss.item()}')

        print(f'Epoch {epoch + 1} done, Training Loss: {running_loss / len(train_loader)}')

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}')

        train_accuracy = get_accuracy(model, train_loader, device)
        val_accuracy = get_accuracy(model, val_loader, device)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy} \n')

        if configs['wandb_log']:
            wandb.log({'Epoch:': epoch + 1,
                       'Training Loss': running_loss / len(train_loader),
                       'Validation Loss': val_loss / len(val_loader),
                       'Training Accuracy': train_accuracy,
                       'Validation Accuracy': val_accuracy})

    print('Finished Training')
    print('---------------------')
    
    return model, configs, train_accuracies, val_accuracies


sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'Validation Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {
            'values': [50, 80, 100, 120, 150, 200]
        },
        'learning_rate': {
            'values': [0.0005, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05]
        },
        'num_epochs': {
            'values': [10]
        },
        'batch_norm': {
            'values': [False]
        },
        'momentum': {
            'values': [0.87, 0.9, 0.93, 0.99]
        },
    }
}


sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME, entity=PROJECT_ENTITY)

def train_sweep():
    run = wandb.init()

    configs = {
        'num_epochs': wandb.config.num_epochs,
        'batch_size': wandb.config.batch_size,
        'learning_rate': wandb.config.learning_rate,
        'batch_norm': wandb.config.batch_norm,
        'momentum': wandb.config.momentum,

        'wandb_log': True
    }

    run.name = f"lr={configs['learning_rate']}_bs={configs['batch_size']}_epochs={configs['num_epochs']}_bn={configs['batch_norm']}_r{np.random.randint(0, 1000)}"

    model = MLP(ARCH, 3*32*32, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=configs['learning_rate'], momentum=configs['momentum'])
    
    wandb.watch(model, criterion, log='all')

    train_data, test_data, train_loader, val_loader, test_loader = get_dataloader(configs['batch_size'])

    model, configs, train_accuracies, val_accuracies = train(configs, train_loader, val_loader, criterion, optimizer, model, device)

    test_accuracy = get_accuracy(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy}')

    if configs['wandb_log']:
        wandb.log({'Test Accuracy': test_accuracy})
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None,
                                                                  y_true=test_data.targets,
                                                                  preds=get_predicted_labels(model, test_loader, device).cpu().numpy(),
                                                                  class_names=list(LABELS.values()))})
        wandb.finish()

    return test_accuracy


wandb.agent(sweep_id, train_sweep, count=50)
