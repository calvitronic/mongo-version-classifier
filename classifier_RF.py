import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from trustee import ClassificationTrustee
import graphviz
from sklearn import tree

# Load data
df = pd.read_csv('../First_Second_NoRemovals.csv')

# Label encoding
class2idx = {
    'three':0,
    'four':1,
    'five':2,
    'six':3,
    'seven':4
}
idx2class = {v: k for k, v in class2idx.items()}

# Prepare features and labels
X = df.select_dtypes(include=['number']).fillna(0)
X = X.drop(['dst_port', 'src_port'], axis=1)
df['label'] = df['label'].replace(class2idx)
y = df['label']

# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
print("Validation classification report:")
print(classification_report(y_val, y_val_pred))

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
print("Test classification report:")
print(classification_report(y_test, y_test_pred))

# Initialize Trustee with the random forest model
trustee = ClassificationTrustee(expert=model)
trustee.fit(X_train, y_train, top_k=5, num_iter=50, num_stability_iter=10, samples_size=0.2, verbose=False)
dt, pruned_dt, agreement, reward = trustee.explain()
dt_y_pred = dt.predict(X_test)

print("Model explanation global fidelity report:")
print(classification_report(y_test_pred, dt_y_pred))
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
