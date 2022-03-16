import torch
import torchvision
from dataset import TomographyDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["stae_dict"])


# Locate batch in pinned memory -> faster transfer to GPU
# info: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

# num_workers: The batches are loaded using additional worker processes 
# and are queued up in memory -> faster training
# num_workers ideal? just test!
# info: https://deeplizard.com/learn/video/kWVgvsejXsE#:~:text=The%20num_workers%20attribute%20tells%20the,sequentially%20inside%20the%20main%20process.
def get_loaders(
    batch_size,
    train_transform,
    val_transform,
    num_workers = 4,
    pin_memory = True
):
    train_ds = TomographyDataset(transform = train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True
    )

    val_ds = TomographyDataset(transform = val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device = 'cuda'):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for X, y in loader: 
            X = X.to(device)
            y = y.to(device)
            y_pred = model(x)

            # Metric to evaluate the performance ? 
            # go here
            metric = 0.8
        
    print(f'Metric {metric}')
    model.train()

def save_predictions_as_imgs(loader, 
                            model, 
                            folder='saved_images/', 
                            device='cuda'
    ):

    model.eval()
    for idx, (X, y) in enumerate(loader):
        X = X.to(device=device)

        with torch.no_grad():
            y_pred = model(x)
        
        torchvision.utils.save_image(y_pred, f'{folder}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}/true_{idx}.png' )
    model.train()

