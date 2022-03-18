# Dataset loading and preprocessing 

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

# Dataset Information:
# All images of 512 x 512
# X : Tomography - image reconstructed - already normalized (0-1)
# y : Mask - unique values (0: background - 1: kidney - 2: tumor - 3: cyst)

# LABEL_AGGREGATION_ORDER = (1, 3, 2) The order matters!
# This means that we first place the kidney, then the cyst and finally the tumor.
# If parts of a later label (example tumor) overlap with a prior label (kidney or cyst) the prior label is overwritten.
#
# KITS_LABEL_NAMES = {
#     1: "kidney",
#     2: "tumor",
#     3: "cyst"
# }

class TomographyDataset(Dataset):  
    
    def __init__(self, data_dir = './data/', dataset_csv = './data/dataset.csv', split_set ='train', transform = None):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_df = pd.read_csv(dataset_csv)
        self.split_set = split_set
        self.transform = transform 

        if self.split_set == 'train':
            self.dataset_df = self.dataset_df[self.dataset_df['split_set'] == 'train']

        elif self.split_set == 'validation':
            self.dataset_df = self.dataset_df[self.dataset_df['split_set'] == 'validation']
        
        elif self.split_set == 'test':
            self.dataset_df = self.dataset_df[self.dataset_df['split_set'] == 'test']
        
        self.dataset_df.reset_index(inplace=True, drop=True)

    
    def __getitem__(self, index):
        X_path = self.dataset_df['reconstruction_file'][index]
        y_path = self.dataset_df['mask_file'][index]

        X_img = np.load(self.data_dir + X_path)

        # condition to create an image if it does not exist
        try:
            y_img = np.load(self.data_dir + y_path)
        except:
            y_img = np.zeros(shape = (512,512))

        if self.transform is not None:
            augmentations = self.transform(image = X_img, mask = y_img)
            X_img = augmentations['image']
            y_img = augmentations['mask']
            #print(f'X_img shape: {X_img.shape}')  # 1,H,W
            #print(f'y_img shape: {y_img.shape}')  # H,W
         
        y_img4 = self.one_hot_encoding(y_img)
        item = X_img, y_img4
        
        return item
    
    def __len__(self): 
        return self.dataset_df.shape[0] 
    
    @staticmethod
    def one_hot_encoding(mask_img):
        shape = mask_img.shape # H,W
        ohe = np.zeros(shape = (4,shape[0],shape[1]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                label = mask_img[i,j]
                if label == 0:
                    ohe[:,i,j] = [1,0,0,0] # background, kidney, tumor, cyst
                elif label == 1: 
                    ohe[:,i,j] = [0,1,0,0]
                elif label == 2:
                    ohe[:,i,j] = [0,1,1,0]
                elif label == 2:
                    ohe[:,i,j] = [0,1,0,1]

        return ohe # 4 channels 
    

if __name__ == '__main__':
    pass