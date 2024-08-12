import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SGR(nn.Module):
    def __init__(self, input_dim):
        super(SGR, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)
