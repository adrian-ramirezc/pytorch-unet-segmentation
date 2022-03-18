import torch
import torchvision
from dataset import TomographyDataset
from torch.utils.data import DataLoader
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint ...")
    torch.save(state, filename)
    print("=> Saved ")

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint ...")
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Loaded")


# Locate batch in pinned memory -> faster transfer to GPU
# info: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

# num_workers: The batches are loaded using additional worker processes 
# and are queued up in memory -> faster training
# num_workers ideal? just test!
# info: https://deeplizard.com/learn/video/kWVgvsejXsE#:~:text=The%20num_workers%20attribute%20tells%20the,sequentially%20inside%20the%20main%20process.
def get_loaders(batch_size,
                train_transform,
                val_transform,
                num_workers = 1,
                pin_memory = True
    ):
    
    train_ds = TomographyDataset(transform = train_transform, split_set ='train')
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True
    )

    val_ds = TomographyDataset(transform = val_transform, split_set ='validation')
    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False
    )

    return train_loader, val_loader

# IMPLEMENT A METRIC TO EVALUATE THE MODEL
def check_accuracy(loader, model, device = 'cuda'):   
    print("=> Checking accuracy of model ...")
    model.eval() # set forward for evaluation
    ji_sum = torch.zeros(1,4).to(device)
    ji_avg_sum = torch.zeros(1).to(device)
    count = 0
    with torch.no_grad():
        for (X, y) in loader: 
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X) 
            
            shape = y.size() #size = (B, CHANNELS = 4, H, W) 
            
            # Intersection Over Union - Jaccard Index 
            ji = torch.sum((y * y_pred), dim=(2,3,0))/torch.sum((y + y_pred - y * y_pred), dim=(2,3,0))
            ji_avg = torch.mean(ji)
            
            ji_sum = ji_sum + ji
            ji_avg_sum = ji_avg_sum + ji_avg
            
            count += shape[0] # images per batch
    
    ji_sum = ji_sum.cpu().detach().numpy()
    ji_avg_sum = ji_avg_sum.cpu().detach().numpy()

    ji_sum = ji_sum/count
    ji_avg_sum = ji_avg_sum/count

    ji_sum = np.around(ji_sum, 2)
    ji_avg_sum = np.around(ji_avg_sum, 2)

    print(f'Jaccard Index for each class: {ji_sum}')
    print(f'Average Jaccard Index: {ji_avg_sum}')
    model.train() # set forward for training
    print('=> End ')

def save_predictions_as_imgs(loader, model, folder='saved_images/', device='cuda'):
    print('=> Saving predictions as images...')
    model.eval() # set forward for evaluation
    for idx, (X, y) in enumerate(loader):
        X = X.to(device=device)

        with torch.no_grad():
            y_pred = model(X)
        
        torchvision.utils.save_image(y_pred, f'{folder}/pred_{idx}.png')
        torchvision.utils.save_image(y, f'{folder}/true_{idx}.png')
    model.train() # set forward for training
    print('=> End ')
