import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.configs import LABELS


def get_accuracy(model: nn.Module, data_loader: dataloader, device: torch.device) -> float:
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


def get_predicted_labels(model: nn.Module, data_loader: dataloader, device: torch.device) -> torch.Tensor:
    '''
    Get the predicted labels of the model on the data_loader
    '''
    predicted_labels = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            _, predicted = torch.max(preds, 1)
            predicted_labels.append(predicted)

    predicted_labels = torch.cat(predicted_labels, dim=0)

    return predicted_labels


def print_info(train_set, val_set, test_set):
    '''
    Print the size of the training, validation and test sets
    '''
    print(f"Training set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")


# Show a random image from the dataset
def show_random_image(dataset: datasets, index: int = None):
    '''
    Shows a random image from the dataset
    '''
    if index is None:
        index = torch.randint(0, len(dataset), (1,)).item()
    else:
        index = index
                
    image, label = dataset[index]
    plt.imshow(image.permute(1, 2, 0)) # change the shape from (3, 32, 32) to (32, 32, 3)
    plt.title(LABELS[label])
    plt.show()

    return index, label


# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    '''
    Plot the confusion matrix
    '''
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS.values())
    disp.plot(cmap='Blues', xticks_rotation='vertical')


# Plot the training and validation losses
def plot_accuracies(train_acc, val_acc):
    '''
    Plot the training and validation accuracies
    '''
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()