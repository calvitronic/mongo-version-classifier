import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define the Feedforward Neural Network Model
class FFN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom Dataset
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Ensure all feature values are numeric
        self.features = self.data.iloc[:, :-5].apply(pd.to_numeric, errors='coerce').fillna(0).values
        # Ensure all label values are numeric
        self.labels = self.data.iloc[:, -5:].apply(pd.to_numeric, errors='coerce').fillna(0).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label

# Initialize the Dataset and DataLoader
csv_file = 'Master_CSV_Low_Variance_Removed.csv'  # replace with your CSV file path
dataset = CSVDataset(csv_file)

# Split the dataset into training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the Model, Loss Function, and Optimizer
input_dim = dataset.features.shape[1]  # number of input features
output_dim = 5  # number of output dimensions
model = FFN(input_dim, output_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the Training Loop
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Define the Evaluation Loop
def evaluate(model, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            actuals.append(labels.numpy())
            predictions.append(outputs.numpy())
    
    actuals = np.vstack(actuals)
    predictions = np.vstack(predictions)

# Train the model
train(model, train_loader, criterion, optimizer, epochs=5)

# Evaluate the model
evaluate(model, test_loader)



