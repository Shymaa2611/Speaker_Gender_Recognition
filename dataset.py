import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    features = df.drop(columns=['label']).values  
    labels = df['label'].values
    labels = np.where(labels == 'male', 1, 0)
    
    return np.array(features), np.array(labels)

def get_data_loader(csv_file="voice.csv", batch_size=32):
    X, y = load_data_from_csv(csv_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = SpeechDataset(X_train, y_train)
    test_dataset = SpeechDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, train_loader, test_loader

class SpeechDataset(Dataset):
    def __init__(self, X, y):
        self.data = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = self.data[index]
        label = self.labels[index]
        return features, label

def get_data():
  X_train, X_test, y_train, y_test, train_loader, test_loader = get_data_loader()
  input_dim = X_train.shape[1]  
  return input_dim,train_loader,test_loader
