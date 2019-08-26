import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 
              'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  
              'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def center(bbox):
    '''
    @param a list of xmin, ymin, xmax, ymax of a bbox
    @return the center coords of the bbox (x, y)
    '''
    return sum(bbox[::2])//2, sum(bbox[1::2])//2

def parse_gt(filename):
    objects = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ')  for x in lines]
        for splitline in splitlines:
            object_struct = {}
            
            object_struct['name'] = splitline[8]
            # resize 512 to 448 ([xmin, ymin, xmax, ymax])
            object_struct['bbox'] = [int(float(item)*448/512) for index, item in enumerate(splitline) if index in [0,1,4,5]]
            
            object_struct['area'] = (object_struct['bbox'][2] - object_struct['bbox'][0]) * (object_struct['bbox'][3] - object_struct['bbox'][1])

            objects.append(object_struct)
            
    return objects

def parse_label_dic(label):
    #label is a dictionary
    #one image output a 7*7*26 的ground truth
    #一次判斷一個grid
    gt_label = np.zeros((7,7,26), dtype=np.float32)    # dtype: Default is numpy.float64.
    for col in range(7): #col -> row
        for row in range(7): #row -> col
            for objects in range(len(label)):
                x_center, y_center = center(label[objects]['bbox'])
                if (col*64 <= x_center <= (col+1)*64) and (row*64 <= y_center <= (row+1)*64):
                    #算object相對應的x,y,w,h,c，先假設gt_label只有一個真正的bbox
                    #print(x_center, y_center)
                    x = (x_center-col*64)/64  # 448 / 7 = 64
                    y = (y_center-row*64)/64  # each grid contains 64 cols & 64 rows)
                    w = (label[objects]['bbox'][2] - label[objects]['bbox'][0])/448
                    h = (label[objects]['bbox'][3] - label[objects]['bbox'][1])/448
                    #print('w,h:',w,h)
                    gt_label[col,row][0:5] = x, y, w, h, 1
                    gt_label[col,row][10+classnames.index(label[objects]['name'])] = 1 #one-hot
    
    return gt_label

class Ariel(Dataset):
    def __init__(self, image_dir, txt_dir, transform=None):
        """ Intialize the Ariel dataset """
        
        self.filenames = []
        self.transform = transform
        
        filenames = glob.glob(os.path.join(image_dir, '*.jpg'))
        txtnames = [os.path.join(txt_dir, f.split('/')[-1][:-4]+'.txt') for f in filenames]
        
        for fn,tn in list(zip(filenames, txtnames)):
            self.filenames.append((fn,tn))
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        # a image and a corresponding label dictionary
        image_fn, image_tn = self.filenames[index]
        
        image = Image.open(image_fn)
        image = image.resize((448, 448), Image.BILINEAR)
        
        label = parse_gt(image_tn)
        label = parse_label_dic(label)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.filenames)