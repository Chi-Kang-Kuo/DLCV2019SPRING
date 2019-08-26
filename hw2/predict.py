'''
python predict.py $1 $2
'''
import os
import cv2
import sys
import glob
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config_path', type=str, help='The path of the config.py file.', default='config.py')
parser.add_argument('-utils', '--utils_path', type=str, default='./utils', help='The path to save trained models.')
args = parser.parse_args()

# import from utils folder
sys.path.append(os.path.dirname(args.config_path))
from models import *
from torch.utils.data import Dataset, DataLoader
from utils import nms, decoder, predict_gpu

# import the config file
sys.path.append(os.path.dirname(args.config_path))
import config
CFG = config.cfg

#import model architecture
which_model = int(sys.argv[3])
if which_model == 0:
    import models
    MODEL_PATH = 'models/baseline_model.pth'
elif which_model == 1:
    import improved_model as models
    MODEL_PATH = 'save/improved/improved_model.pth'
else:
    print('wrong model!')
    
# PATH
IMAGE_PATH = sys.argv[1]  #'hw2_train_val/val1500/images'
SAVE_PATH = sys.argv[2]   #'Prediction_14to7_2'

# mkdir
if os.path.isdir(SAVE_PATH) == False:
    os.makedirs(SAVE_PATH)
    
# 指定gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = 0"

# Use CPU
#device = torch.device("cpu")
# Use GPU if available, otherwise stick with cpu

use_cuda = torch.cuda.is_available() # return True or False 
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

# 16 Classes
classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 
              'ship', 'tennis-court','basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 
              'harbor', 'swimming-pool', 'helicopter', 'container-crane']

Color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
         [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], 
         [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

# load model
model = models.Yolov1_vgg16bn(pretrained=True).to(device)
print('load model...')
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # set eval mode

img_list = glob.glob(os.path.join(IMAGE_PATH, '*.jpg'))
#print(img_list)
print('predicting...')
for file in img_list:
    ans = []
    #print(file)
    filename = os.path.basename(file)
    print(filename)
    
    results = predict_gpu(model, file, 0)
    print(results)
    
    for res in results:
        xmin, ymin = res[0]
        xmax, ymax = res[1]
        #print(xmin, ymin)
        ans.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, res[2], res[4]])
        #print(ans)
    save_name = os.path.join(SAVE_PATH, filename[:-4]+'.txt')
    pd.DataFrame(ans).to_csv(save_name , sep = ' ', header = False , index = False) 
    print('save to : ', save_name)
    