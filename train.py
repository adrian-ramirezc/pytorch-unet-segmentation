import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn 
import torch.optim as optim
from model import UNET

import argparse


from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# Run the script like this:
# python train.py -pm -bs 5 -e 20 -nw 4 -imgh 256 -imgw 256 

# Hyperparameters
parser = argparse.ArgumentParser(   prog='hyperparameter selection', 
                                    description='Modify the hyperparameters by default'
                                    )

parser.add_argument('-lr'  ,'--learning_rate', type=float, default=1e-4,  help='learning rate')
parser.add_argument('-dvc' ,'--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', help='Use cpu or gpu?')
parser.add_argument('-bs'  ,'--batch_size', type=int, default=1, help='Number of imgs per batch')
parser.add_argument('-e'   ,'--epochs',type=int, default=1, help='Number of epochs')
parser.add_argument('-nw'  ,'--num_workers', type=int, default=1, help='Number of workers')

parser.add_argument('-imgh'  ,'--image_height', type=int, default=64, help='Image Height')
parser.add_argument('-imgw'  ,'--image_width', type=int, default=64, help='Image Width')

parser.add_argument('-pm'  ,'--pin_memory', action='store_true', help='Pin memory')
parser.add_argument('-lm'  ,'--load_model', action='store_true', help='Load model')

#parser.add_argument('-Xy', 'Xy_directory', default='./data/dataset.csv', help='Reconstruction and Mask images path')

args = parser.parse_args()

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (X, y) in enumerate(loop):
        X = X.to(device = args.device)
        y = y.to(device = args.device)

        # forward 
        with torch.cuda.amp.autocast(): # Using FP16 (accelerates training / Floating point 16)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        # backward
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss = loss.item())
     

def main():
    train_transform = A.Compose(
        [
            A.Resize(height = args.image_height, width = args.image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            ToTensorV2()
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height = args.image_height, width = args.image_width),
            ToTensorV2()
        ],
    )

    model = UNET(in_channels =1, out_channels=4).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    train_loader, val_loader = get_loaders(
                            batch_size = args.batch_size,
                            train_transform = train_transform,
                            val_transform = val_transform,                            
                            num_workers = args.num_workers,
                            pin_memory = args.pin_memory)

    if args.load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_accuracy(val_loader, model, device = args.device)

    scaler = torch.cuda.amp.GradScaler() # Using FP16, we risk of underflowing -> scale the gradient

    for epoch in range(args.epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save load_model
        checkpoint = {
            "state_dict": model.state_dict(), 
            "optimizer" : optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        if (epoch+1) % 5 == 0:
            # check accuracy
            check_accuracy(val_loader, model, device = args.device)

        if epoch == args.epochs - 1:
            # print some examples to a folder 
            save_predictions_as_imgs(val_loader, model, folder='saved_images/', device=args.device)


if __name__ == '__main__':
    main()