import os
import glob
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class CelebA(Dataset):
    def __init__(self, image_dir, csv_dir, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.transform = transform

        # read filenames
        filenames = glob.glob(os.path.join(image_dir, '*.png'))
        filenames.sort(key=lambda x: int(x.split('/')[-1][:-4])) #sort by png order
        # read attribute label
        df = pd.read_csv('hw3_data/face/train.csv')
        attri_label = df['Blond_Hair'].values.tolist()
    
        #print(filenames)
        #txtnames = [os.path.join(txt_dir, item.split('/')[-1][:-4]+'.txt') for item in filenames]
        for fn,tn in list(zip(filenames,attri_label)):
            self.filenames.append((fn,tn))
            #print(fn,tn)
            
    def __getitem__(self, index):
        # a image and a corresponding label dictionary
        image_fn, label = self.filenames[index]
        #print(image_fn)
        image = Image.open(image_fn)
        label = torch.tensor(label)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)