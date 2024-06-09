import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns # Plotting stuff

# Define a random seed so results are reproducable...
random_seed = 42

# Define the Feedforward Neural Network Model
class FFN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Custom Dataset
class CSVDataset(Dataset):
    def __init__(self, features_in, labels_in):
    # def __init__(self, csv_file):
        # self.data = pd.read_csv(csv_file)
        # # Ensure all feature values are numeric
        # # Needs to be modified to take into account feature values are now alphabetic...
        # self.features = torch.from_numpy(self.data.select_dtypes(include=['number']).fillna(0).values)
        # #self.features = self.data.iloc[:, :-5].apply(pd.to_numeric, errors='coerce').fillna(0).values
        # # Ensure all label values are numeric
        # self.labels = F.one_hot(torch.from_numpy(self.data['label'].apply(map_text_to_number).values))
        # # Legacy method of grabbing labels
        # # self.labels = self.data['label'] #.iloc[:, -5:].apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        self.features = torch.from_numpy(features_in).float()
        # self.labels = F.one_hot(torch.from_numpy(labels_in)).long()
        self.labels = torch.from_numpy(labels_in).long()

    def __len__(self):
        # return len(self.data)
        return len(self.labels)

    def __getitem__(self, idx):
        #sample = torch.tensor(self.features[idx], dtype=torch.float32)
        sample = self.features[idx].to(dtype=torch.float32).clone().detach()
        #label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = self.labels[idx].to(dtype=torch.float32).clone().detach()
        return sample, label

def map_text_to_number(text):
    mapping = {
        "three": 0,
        "four": 1,
        "five": 2,
        "six": 3,
        "seven": 4
    }
    return mapping.get(text, -1)

def get_class_numbers(label_in):
    label_dict = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0
    }
    for i in label_in:
        label_dict[str(i)] += 1
    return label_dict

# Create the device to use
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize the Dataset and DataLoader
csv_file = 'Master_CSV_Low_Variance_Removed.csv'  # replace with your CSV file path
# sns.countplot(x = "label", data=pd.read_csv(csv_file))
# dataset = CSVDataset(csv_file)

dataset = pd.read_csv(csv_file)

X = dataset.select_dtypes(include=['number']).fillna(0).values
y = dataset['label'].apply(map_text_to_number).values

# Splitting dataset into training and testing

trainval_X, X_test, trainval_y, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=random_seed)
print(f"Shape of trainval_X: {trainval_X.shape}\nShape of trainval_y: {trainval_y.shape}")
X_train, X_val, y_train, y_val = train_test_split(trainval_X, trainval_y, test_size=0.1, stratify=trainval_y, random_state=random_seed)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

train_dataset = CSVDataset(X_train, y_train)
validation_dataset = CSVDataset(X_val, y_val)
test_dataset = CSVDataset(X_test, y_test)

# Prepare the weighted sampler...
target_list = []
for _, t in train_dataset:
    target_list.append(t)

# print(target_list)
target_list = torch.tensor(target_list)

class_count = [i for i in get_class_numbers(y_train).values]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(f"Class weights: {class_weights}")

class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=weighted_sampler)
val_loader = DataLoader(validation_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the Model, Loss Function, and Optimizer
input_dim = dataset.features.shape[1]  # number of input features
output_dim = 5  # number of output dimensions
model = FFN(input_dim, output_dim)
model.to(device)
criterion = nn.CrossEntropyLoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Keep track of loss statistics with a dictionary of lists...
loss_stats = {'training': [], 'validation': []}

# Define the Training Loop
def train(model, train_loader, criterion, optimizer, epochs=20):
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        validation_loss = 0.0
        # Work with training set
        for features, labels in train_loader:
            # Send them to gpu if you can!
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            # print(outputs)
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Done with training for the epoch; let's evaluate now.
        with torch.no_grad():
            # switch model to eval mode for the time being
            model.eval()
            for features, labels in val_loader:
                # Again, send to gpu if possible
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                validation_loss += loss.item()
        
        total_running_loss = running_loss/len(train_loader)
        total_validation_loss = validation_loss/len(val_loader)

        loss_stats['training'].append(total_running_loss)
        loss_stats['validation'].append(total_validation_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] | Loss: {total_running_loss:.4f} | Validation Loss: {total_validation_loss:.4f}')

# Define the Evaluation Loop
def evaluate(model, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for features, labels in test_loader:
            # One last time: send things to gpu if you can
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            actuals.append(labels[0])
            predictions.append(outputs[0])
    
    #print(F.softmax(predictions[0]))
    # actuals = actuals.to("cpu")
    # predictions = predictions.to("cpu")
    predictions = [torch.softmax(pred, dim=-1) for pred in predictions]
    actuals = [x.to("cpu") for x in actuals]
    predictions = [pred.to("cpu") for pred in predictions]
    actuals = np.array(actuals)
    predictions = np.array(predictions)

    actuals = np.argmax(actuals, axis=1).reshape(-1, 1)
    predictions = np.argmax(predictions, axis=1).reshape(-1, 1)

    actuals = np.squeeze(actuals)
    predictions = np.squeeze(predictions)
    
    print(f"After squeezing predictions:\n{predictions}")
    print(f"After squeezing actuals:\n{actuals}")

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='weighted')
    f1 = f1_score(actuals, predictions, average='weighted')
    print(f'Accuracy: {accuracy}, Precision: {precision}, F1-Score: {f1}')



# Train the model
train(model, train_loader, criterion, optimizer, epochs=50)

# Evaluate the model
evaluate(model, test_loader)

# Save the model weights / results... maybe make it toggleable
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_stats
}, "./model_checkpoint_out_1.tar")