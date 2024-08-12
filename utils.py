import torch
import os
from args import *

def save_checkpoint(epoch, model, optimizer, train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_path='checkpoint/checkpoint.pt'):
    # Ensure the checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved at epoch {epoch+1}')
    except Exception as e:
        print(f'Failed to save checkpoint: {e}')



def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = SGR(input_dim)  
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model