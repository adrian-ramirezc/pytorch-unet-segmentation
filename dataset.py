# Dataset loading and preprocessing 

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


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

class myDataset(Dataset):  
    
    def __init__(self, data_dir = './data/', dataset_csv = './data/dataset.csv'):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_df = pd.read_csv(dataset_csv)
    
    def __getitem__(self, index):
        X_path = self.dataset_df['reconstruction_file'][index]
        y_path = self.dataset_df['mask_file'][index]

        X_img = np.load(self.data_dir + X_path)

        # condition to create an image if it does not exist
        try:
            y_img = np.load(self.data_dir + y_path)
        except:
            y_img = np.zeros(shape = (512,512))

        y_img4 = self.one_hot_encoding(y_img)
        y_img4 = np.append(y_img4, y_img.reshape(1,512,512), axis=0)

        item = X_img, y_img4
        
        return item
    
    def __len__(self): 
        return self.dataset_df.shape[0] 
    
    @staticmethod
    def one_hot_encoding(mask_img, shape = (4,512,512)):
        ohe = np.zeros(shape = shape)
        for i in range(shape[1]):
            for j in range(shape[2]):
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
    
    @staticmethod
    def show_item(X, y_img4):
        fig, ax = plt.subplots(ncols=6, figsize=(20,10))
        ax=ax.reshape(-1)
        ax[0].imshow(X, cmap='Greys')
        ax[1].imshow(y_img4[0], cmap='Greys', vmin = 0, vmax = 1) # background
        ax[2].imshow(y_img4[1], cmap='Greys', vmin = 0, vmax = 1) # kidney
        ax[3].imshow(y_img4[2], cmap='Greys', vmin = 0, vmax = 1) # tumor
        ax[4].imshow(y_img4[3], cmap='Greys', vmin = 0, vmax = 1) # cyst
        ax[5].imshow(y_img4[4]) # mask

if __name__ == '__main__':
    pass