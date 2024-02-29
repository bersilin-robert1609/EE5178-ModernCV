import torch
import torchvision.datasets as datasets
import torch.utils.data as dataloader
import torch.nn as nn
import torch.optim as optim

from src.utils import get_accuracy
from src.configs import PROJECT_NAME, PROJECT_ENTITY

# Training the model
def train(configs,
          train_loader: dataloader.DataLoader,
          val_loader: dataloader.DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          model: nn.Module,
          device: torch.device):
    
    if configs['wandb_log']:
        import wandb
        run = wandb.init(project=PROJECT_NAME, entity=PROJECT_ENTITY, config=configs)
        run.name = f"lr={configs['learning_rate']}_bs={configs['batch_size']}_epochs={configs['num_epochs']}"
        wandb.watch(model, criterion, log='all')

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

            outputs = model(inputs)
            
            # change labels to one-hot encoding
            labels = nn.functional.one_hot(labels, num_classes=10).float()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

        train_accuracy = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)

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

    if configs['wandb_log']:
        wandb.finish()
    
    return model, configs, train_accuracies, val_accuracies