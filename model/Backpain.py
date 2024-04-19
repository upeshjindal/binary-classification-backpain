#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[15]:


df = pd.read_csv("./data/Dataset_spine.csv")
df.head()


# In[16]:


sns.countplot(x="Class_att", data=df)


# In[17]:


# Encode output class
# df["Class_att"] = df["Class_att"].astype("category")

encode_map = {
    "Abnormal": 1,
    "Normal": 0
}

df["Class_att"].replace(encode_map, inplace=True)


# In[18]:


df["Class_att"]


# In[24]:


X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

print(f"{X.head()=}")
print(f"{y.head()=}")


# In[26]:


RANDOM_SEED = 42
# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_SEED)

print(f"{X_train.shape=}")
print(f"{X_test.shape=}")
print(f"{y_train.shape=}")
print(f"{y_test.shape=}")


# In[27]:


# Standardize input
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[35]:


X_train


# In[28]:


# Hyper parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01


# In[37]:


class TrainData(Dataset):
    
    def __init__(self, X_train, y_train):
        
        self.X_data = X_train
        self.y_data = y_train
        
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
train_dataset = TrainData(X_train=torch.FloatTensor(X_train), y_train=torch.FloatTensor(y_train))
train_dataset


# In[38]:


class TestData(Dataset):
    
    def __init__(self, X_test) -> None:
        
        self.X_data = X_test
        
    def __len__(self):
        return len(self.X_data)
        
    def __getitem__(self, index):
        return self.X_data[index]
    
test_dataset = TestData(X_test=torch.FloatTensor(X_test))
test_dataset


# In[39]:


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

print(f"{train_loader=}")
print(f"{test_loader=}")


# In[41]:


# Test the data loader
batch = next(iter(train_loader))
print(batch)


# In[44]:


class BackPainNN(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm_1 = nn.BatchNorm1d(64)
        self.batchnorm_2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        
        x = self.relu(self.layer_1(x))
        x = self.batchnorm_1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm_2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    


# In[45]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[46]:


model = BackPainNN(12, 64, 1).to(device=device)
model


# In[47]:


loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[65]:


def binary_accuract(pred, truth):
    
    y_pred_tag = torch.round(torch.sigmoid(pred))
    
    correct = torch.eq(y_pred_tag, truth).sum().item()
    return correct / len(truth)


# In[66]:


model.train()

for epoch in range(EPOCHS):
    
    epoch_loss, epoch_accuracy = 0, 0
    for X, y in train_loader:
        
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        
        # print(y_logits.shape)
        # print(y.shape)
        
        loss = loss_fn(y_logits.squeeze(), y)
        acc = binary_accuract(y_logits.squeeze(), y)
        
        epoch_loss += loss.item()
        epoch_accuracy += acc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    
    print(f"Epoch: {epoch} | Loss: {epoch_loss:.5f} | Accuracy: {epoch_accuracy:.2f}")
        
        
        


# In[69]:


model.eval()
y_pred_list = []
with torch.inference_mode():
    for test_batch in test_loader:
        
        test_batch = test_batch.to(device)
        logits = model(test_batch)
        
        y_test_pred = torch.round(torch.sigmoid(logits))
        y_pred_list.append(y_test_pred.squeeze().cpu().numpy())

y_pred_list


# In[70]:


confusion_matrix(y_test, y_pred_list)


# In[72]:


print(classification_report(y_test, y_pred_list))


# In[75]:


MODEL_PATH = "./models"
MODEL_NAME = "BackPain.pth"

torch.save(model.state_dict(), MODEL_NAME)


# In[83]:


weights = torch.load(MODEL_NAME)

X, y = next(iter(train_loader))

loaded_model = BackPainNN(12, 64, 1)
loaded_model.load_state_dict(weights)

loaded_model.eval()
with torch.inference_mode():
    
    preds = loaded_model(X)
    
    print(preds)


# In[ ]:




