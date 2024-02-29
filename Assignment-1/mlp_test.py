import local_configs as lcfg
import torch
import torch.nn as nn
import torch.optim as optim

import src.global_configs as cfg
from src.MLP import MLP
from src.dataloader import load_data, val_split, create_dataloader
from src.train import train

from sklearn.metrics import confusion_matrix

import src.utils as utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

train_set, test_set = load_data(lcfg.train_transform)
train_set, val_set = val_split(train_set)
train_loader, val_loader, test_loader = create_dataloader(train_set, val_set, test_set, lcfg.configs['batch_size'])

utils.print_info(train_set, val_set, test_set)

# index, label= utils.show_random_image(train_set)

print("The size of one image is: ", train_set[0][0].size())
print("The label of the image is: ", cfg.LABELS[train_set[0][1]])

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lcfg.configs['learning_rate'], momentum=lcfg.configs['momentum'])

model, configs, val_accuracies, train_accuracies = train(lcfg.configs, train_loader, val_loader, criterion, optimizer, model, device)

test_accuracy = utils.get_accuracy(model, test_loader, device)
print(f'Test accuracy: {test_accuracy}')

predicted_labels = utils.get_predicted_labels(model, test_loader, device).to('cpu').numpy()
true_y = torch.tensor(test_loader.dataset.targets).to('cpu').numpy()

cm = confusion_matrix(true_y, predicted_labels)
# Store the confusion matrix in a file
import numpy as np
np.savetxt('confusion_matrix.txt', cm, fmt='%d')

index = np.random.randint(0, len(test_set), 1)

print(f"Predicted label: {cfg.LABELS[predicted_labels[index][0]]}")
print(f"True label: {cfg.LABELS[true_y[index][0]]}")