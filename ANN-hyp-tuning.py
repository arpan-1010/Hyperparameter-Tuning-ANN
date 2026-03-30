import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#set random seed for reproducibility
torch.manual_seed(42)

#check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")

df = pd.read_csv('fashion-mnist_train.csv')
df.head()

#create a 4x4 grid of images
fig, axes = plt.subplots(4, 4, figsize = (10, 10))
fig.suptitle("First 16 images", fontsize = 16)

#plot the first 16 images from the dataset
for i, ax in enumerate(axes.flat):
  img = df.iloc[i, 1:].values.reshape(28, 28)
  ax.imshow(img)
  ax.axis('off')
  ax.set_title(f"Label : {df.iloc[i, 0]}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaling
X_train = X_train / 255.0
X_test = X_test / 255.0

#create custom dataset class
class CustomDataset(Dataset):

  def __init__(self, features, labels):
    self.features = torch.tensor(features, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)

  def __len__(self):
    return len(self.features)

  def __getitem__(self, index):
    return self.features[index], self.labels[index]

#create train_dataset obj
train_dataset = CustomDataset(X_train, y_train)

#create test_dataset obj
test_dataset = CustomDataset(X_test, y_test)


class MyNN(nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate):
        super().__init__()

        layers = []

        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, neurons_per_layer))
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = neurons_per_layer

        layers.append(nn.Linear(neurons_per_layer, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# objective function
def objective(trial):
    # next hyperparameter values from search space
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
    neurons_per_layer = trial.suggest_int("neurons_per_layer", 8, 128, step=8)
    epochs = trial.suggest_int("epochs", 10, 50, step=10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop'])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # create train and test loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # model init
    input_dim = 784
    output_dim = 10

    model = MyNN(input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate)
    model.to(device)

    # optimizer selection
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

    if optimizer_name == 'Adam':
        optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # training loop
    for epoch in range(epochs):

        for batch_features, batch_labels in train_loader:
            # move data to gpu
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # forward pass
            outputs = model(batch_features)

            # loss calculation
            loss = criterion(outputs, batch_labels)

            # back pass
            optimizer.zero_grad()
            loss.backward()

            # update grads
            optimizer.step()

        # evaluation
        model.eval()
        # evaluation
        total = 0
        correct = 0

        with torch.no_grad():

            for batch_features, batch_labels in test_loader:
                # move data to gpu
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                outputs = model(batch_features)

                _, predicted = torch.max(outputs, 1)

                total = total + batch_labels.shape[0]

                correct = correct + (predicted == batch_labels).sum().item()

        accuracy = (correct / total)
        return accuracy

import optuna

study = optuna.create_study(direction = 'maximize')

print(study.best_value)

print(study.best_params)