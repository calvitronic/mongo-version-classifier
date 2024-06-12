# Copied wholesale from https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
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
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)
        
        
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

        x = self.layer_4(x)
        x = self.batchnorm4(x)
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
    
    # def preprocess_input(self, input):
    #     if isinstance(input, pd.DataFrame):
    #         input = input.values  # Convert DataFrame to numpy array
    #     input = torch.tensor(input, dtype=torch.float32).to(self.device)
    #     return input

    def predict(self, input):
        with torch.no_grad():
            self.model.eval()
            if isinstance(input, pd.DataFrame):
                input = input.values  # Convert DataFrame to numpy array
            input = torch.tensor(input, dtype=torch.float32).to(device)
            result = self.model(input)
            _, predictions = torch.max(result, 1)
            # print(predictions.cpu().numpy())
        result = predictions.cpu().numpy()
        # print(f"Type of result: {type(result)}")
        return result

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



df = pd.read_csv('Master_CSV_Low_Variance_Removed.csv')

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
print(X.columns)
df['label'] = df['label'].replace(class2idx)

y = df['label']

# Split into train+val and test
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
print(class_weights)

class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.00001 #0.005 #0.0001
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
scheduler = MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
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
        scheduler.step()
        
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

# confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
# heatmap_results = sns.heatmap(confusion_matrix_df, annot=True)
# plt.show()


# Check the type of X_train and y_train
# print(f"X_train type: {type(X_train)}")
# print(f"y_train type: {type(y_train)}")
print(classification_report(y_test, y_pred_list))


torch.save({
                    'epoch': EPOCHS,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_stats
                }, "./expert_model_decayingLR_3_checkpoint"
                )

# test_model = MultiClassModelWrapper(model, device, train_loader)
test_model = MultiClassModelWrapper(model, device)

trustee = ClassificationTrustee(expert=test_model)
trustee.fit(X_train, y_train, top_k=5, num_iter=50, num_stability_iter=10, samples_size=0.2, verbose=False)
# trustee.fit(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), num_iter=50, num_stability_iter=10, samples_size=0.2, verbose=True)
dt, pruned_dt, agreement, reward = trustee.explain()
dt_y_pred = dt.predict(X_test)

print("Model explanation global fidelity report:")
print(classification_report(y_pred_list, dt_y_pred))
print("Model explanation score report:")
print(classification_report(y_test, dt_y_pred))

# # Get the best explanation from Trustee
# dt, pruned_dt, agreement, reward = trustee.explain()
# print(f"Model explanation training (agreement, fidelity): ({agreement}, {reward})")
# print(f"Model Explanation size: {dt.tree_.node_count}")
# print(f"Top-k Prunned Model explanation size: {pruned_dt.tree_.node_count}")

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
graph.render("pruned_dt_explation")
