# Copied wholesale from https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from trustee import ClassificationTrustee
import graphviz
from sklearn import tree


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
    
    

class MultiClassModelWrapper():
    # def __init__(self, model_in, device, dataload_in):
    #     self.model = model_in
    #     self.device = device
    #     self.dataloader = dataload_in
    def __init__(self, model_in, device):
        self.model = model_in
        self.device = device
    
    def predict(self, input):
        result = torch.tensor([])
        with torch.no_grad():
            self.model.eval()
            input = input.to_numpy(dtype=np.float32)
            # input = input.flatten()
            tensor_in = torch.from_numpy(input)
            # tensor_in = tensor_in.to(self.device)
            # input = torch.tensor(input[0], dtype=float)
            # print(f"Input here is {tensor_in}")
            temp = self.model(tensor_in)
            _, result = torch.max(temp, dim = 1)
            # print(f"Result here is {result}")
        return result.numpy()

    # def predict(self, input):
    #     input = input # Shut it up
    #     result = []
    #     with torch.no_grad():
    #         self.model.eval()
    #         for X_batch, _ in self.dataloader:
    #             X_batch = X_batch.to(self.device)
    #             y_test_pred = self.model(X_batch)
    #             _, y_pred_tags = torch.max(y_test_pred, dim = 1)
    #             result.append(y_pred_tags.cpu().numpy())
    #         result = [a.squeeze().tolist() for a in result]
    #     return result



df = pd.read_csv('./first_and_second_no_removal_fixed.csv')

# # Fix label column
# for ind in range(len(df['label'])):
#     old_label = df['label'][ind]
#     if "three" in old_label:
#         df.at[ind, 'label'] = 'three'
#     elif "four" in old_label:
#         df.at[ind, 'label'] = 'four'
#     elif "five" in old_label:
#         df.at[ind, 'label'] = 'five'
#     elif "six" in old_label:
#         df.at[ind, 'label'] = 'six'
#     elif "seven" in old_label:
#         df.at[ind, 'label'] = 'seven'

# df.to_csv('./first_and_second_var5_removal_fixed.csv')

class2idx = {
    'three':0,
    'four':1,
    'five':2,
    'six':3,
    'seven':4
}

idx2class = {v: k for k, v in class2idx.items()}
# df['label'].replace(class2idx, inplace=True)

X = df.select_dtypes(include=['number']).fillna(0)
X = X.drop(['dst_port', 'src_port'], axis=1)
df['label'] = df['label'].replace(class2idx)
y = df['label']

# Split into train+val and test
print(f"y first two or so: {y[:2]}")
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

def get_class_distribution(obj):
    count_dict = {
        "three": 0,
        "four": 0,
        "five": 0,
        "six": 0,
        "seven": 0
    }
    
    for i in obj:
        if i == 0: 
            count_dict['three'] += 1
        elif i == 1: 
            count_dict['four'] += 1
        elif i == 2: 
            count_dict['five'] += 1
        elif i == 3: 
            count_dict['six'] += 1
        elif i == 4: 
            count_dict['seven'] += 1          
        else:
            print("Check classes.")
            
    return count_dict

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)

class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float)

class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

EPOCHS = 50#300
BATCH_SIZE = 16
LEARNING_RATE = 0.0007
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 5

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device in use: {device}")

model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# print(model)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

print("Beginning training...")
for e in tqdm(range(1, EPOCHS+1)):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    displayed_tloss = train_epoch_loss/len(train_loader)
    displayed_vloss = val_epoch_loss/len(val_loader)
    displayed_tacc = train_epoch_acc/len(train_loader)
    displayed_vacc = val_epoch_acc/len(val_loader)
    loss_stats['train'].append(displayed_tloss)
    loss_stats['val'].append(displayed_vloss)
    accuracy_stats['train'].append(displayed_tacc)
    accuracy_stats['val'].append(displayed_vacc)
                              
    print(f'Epoch {e+0:03}: | Train Loss: {displayed_tloss:.5f} | Val Loss: {displayed_vloss:.5f} | Train Acc: {displayed_tacc:.3f}| Val Acc: {displayed_vacc:.3f}')

# # Create dataframes
# train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# # Plot the dataframes
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
# sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
# sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

# Testing the model:
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
# heatmap_results = sns.heatmap(confusion_matrix_df, annot=True)
# plt.show()

print(classification_report(y_test, y_pred_list))

# test_model = MultiClassModelWrapper(model, device, train_loader)

# Try sending model to cpu first
model.to("cpu")

test_model = MultiClassModelWrapper(model, device)

# print(f"First two values of y_train {y_train[:2]}")

trustee = ClassificationTrustee(expert=test_model)
# trustee.fit(X_train, y_train.to_numpy(), num_iter=50, num_stability_iter=10, samples_size=0.2)
trustee.fit(X_train, y_train.astype(int), num_iter=50, num_stability_iter=10, samples_size=0.2)
dt, pruned_dt, agreement, reward = trustee.explain()
dt_y_pred = dt.predict(X_test)

print("Model explanation global fidelity report:")
print(classification_report(y_pred_list, dt_y_pred))
print("Model explanation score report:")
print(classification_report(y_test, dt_y_pred))

# Output decision tree to pdf
dot_data = tree.export_graphviz(
    dt,
    class_names=['three', 'four', 'five', 'six', 'seven'],
    feature_names=X.columns,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("dt_explanation")

# Output pruned decision tree to pdf
dot_data = tree.export_graphviz(
    pruned_dt,
    class_names=['three', 'four', 'five', 'six', 'seven'],
    feature_names=X.columns,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("pruned_dt_explanation")
