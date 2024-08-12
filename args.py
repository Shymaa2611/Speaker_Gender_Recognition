from dataset import get_data
from model import SGR
from torch import nn,optim

input_dim, train_loader, val_loader = get_data()  
model = SGR(input_dim)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []