import torch
from utils import load_checkpoint
from dataset import get_data_loader
def inference(model, X_new):
    model.eval()
    with torch.no_grad():
        X_new = torch.tensor(X_new).float()
        if X_new.ndimension() == 1:
            X_new = X_new.unsqueeze(0)  
        output = model(X_new)
        prediction = torch.round(output)
    return prediction.item()



if __name__ == "__main__":
    checkpoint_path = 'checkpoint/checkpoint.pt'
    model = load_checkpoint(checkpoint_path)
    
    _, X_test, _, _, _, _ = get_data_loader()
    result = inference(model, X_test[0])
    if result==1:print("Male")
    else:
        print("Female")