'''
python train.py -cfg=./config.py -save=./save/? -utils=./utils
'''
import os
import sys
import cv2
import glob
import time
import torch
import argparse
import datetime
import numpy as np
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config_path', type=str, help='The path of the config.py file.', default='config.py')
parser.add_argument('-save', '--save_path', type=str, help='The path to save trained models.')
parser.add_argument('-save_img', '--save_img_path', type=str, help='The path to save output imgs.')
parser.add_argument('-utils', '--utils_path', type=str, default='./utils', help='The path to save trained models.')
args = parser.parse_args()

# import warm-up learning rate module  
# https://github.com/ildoonet/pytorch-gradual-warmup-lr
from warmup_scheduler import GradualWarmupScheduler

# import from utils folder
sys.path.insert(1, args.utils_path)
from models import Yolov1_vgg16bn, Yolov1_resnet101, Yolov1_resnet50
from dataset import Ariel
from yoloLoss import yoloLoss
from torch.utils.data import DataLoader
from utils import nms, check_labels, predict_gpu, Color, classnames

# import the config file
sys.path.append(os.path.dirname(args.config_path))
import config
CFG = config.cfg

# Create the Ariel dataset. 
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1] !!!

train_set = Ariel(image_dir='hw2_train_val/train15000/images', txt_dir='hw2_train_val/train15000/labelTxt_hbb',
                 transform=transforms.ToTensor())

valid_set = Ariel(image_dir='hw2_train_val/val1500/images', txt_dir='hw2_train_val/val1500/labelTxt_hbb/',
                 transform=transforms.ToTensor())

print('# images in train_set:', len(train_set)) # Should print 15000
print('# images in valid_set:', len(valid_set)) # Should print 1500

# Use the torch dataloader to iterate through the dataset
train_loader = DataLoader(train_set, batch_size=CFG.train_batch_size, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_set, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=1)

for images, labels in train_loader:
    print('Image tensor in each batch:', images.shape, images.dtype, images.type())
    print('Label tensor in each batch:', labels.shape, labels.dtype, labels.type())
    break

print('Output a image to check gt labels are correct.')
check_labels(images[0], labels[0], args.save_img_path)

# Use GPU if available, otherwise stick with cpu
cuda = torch.cuda.is_available() # return True or False 
torch.manual_seed(123)
device = torch.device("cuda" if cuda else "cpu")
print('Device used:', device)

# model
net = eval(CFG.model)(pretrained=True)

criterion = yoloLoss(7,2,5,0.5)

net.train()
net.to(device)

#print(net)

params=[]
params_dict = dict(net.named_parameters())
#params_dict.keys()
learning_rate = CFG.learning_rate

for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
        
optimizer = torch.optim.Adam(params,lr=learning_rate,weight_decay=1e-4)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.train_epochs)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)

print('the dataset has %d images' % (len(train_set)))

num_iter = 0
num_epoch = CFG.train_epochs
best_loss = 100000.0

# clear old txt file and write log time
with open(os.path.join(args.save_path,'loss.txt'),'w') as f:
    f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")
    
for epoch in range(num_epoch):
    
    epoch_start_time = time.time()
    
    scheduler_warmup.step()     # 10 epoch warmup, after that schedule as scheduler_plateau
    
    train_loss = 0.0
    val_loss = 0.0
    
    net.train()
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epoch))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    #predicting
    image_list = ['hw2_train_val/val1500/images/0076.jpg','hw2_train_val/val1500/images/0086.jpg',
                  'hw2_train_val/val1500/images/0907.jpg']
    
    if epoch in [1,10,20,30,40,49]:
        for image_name in image_list:
            print(image_name)
            image = cv2.imread(image_name)
            print('predicting...')
            result = predict_gpu(net, image_name, device)
            for left_up,right_bottom,class_name,_,prob in result:
                color = Color[classnames.index(class_name)]
                cv2.rectangle(image,left_up,right_bottom,color,2)
                label = class_name+str(round(prob,2))
    
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (left_up[0], left_up[1]- text_size[1])
                cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

            cv2.imwrite(os.path.join(args.save_img_path, 'epoch_'+str(epoch)+'_'+image_name.split('/')[-1]), image)
            
            
    for i,(images,target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        if cuda:
            images, target = images.to(device), target.to(device)
        
        train_pred = net(images)
        batch_loss = criterion(train_pred,target)
        batch_loss.backward()
        optimizer.step()
        
        train_loss += batch_loss.item() #.data
        
        progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)

    #validation
    net.eval()
    with torch.no_grad(): # This will free the GPU memory used for back-prop

        for i,(images,target) in enumerate(valid_loader):
            
            if cuda:
                images, target = images.to(device), target.to(device)

            val_pred = net(images)
            batch_loss = criterion(val_pred,target)
            val_loss += batch_loss.item()
            
            progress = ('#' * int(float(i)/len(valid_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                    (time.time() - epoch_start_time), progress), end='\r', flush=True)
            
        print ('[%03d/%03d] %2.2f sec(s) Train Loss: %3.6f | Val Loss: %3.6f' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, train_loss, val_loss))

    with open(os.path.join(args.save_path,'loss.txt'),'a') as f:
        f.write('[%03d/%03d] %2.2f sec(s) Train Loss: %3.6f | Val Loss: %3.6f\n' % \
                (epoch + 1, num_epoch, time.time()-epoch_start_time, train_loss, val_loss))
        
        if (val_loss < best_loss):
        
            torch.save(net.state_dict(), os.path.join(args.save_path, \
                       'epoch_{}_loss_{:3.3f}.pth'.format(epoch+1,val_loss)))
            best_loss = val_loss
            print ('Model Saved!')
            f.write('Epoch: %d Model Saved!\n' % epoch+1)
        
