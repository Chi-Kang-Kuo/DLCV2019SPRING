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
import numpy as np
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config_path', type=str, help='The path of the config.py file.', default='config.py')
parser.add_argument('-save', '--save_path', type=str, help='The path to save trained models.')
parser.add_argument('-utils', '--utils_path', type=str, default='./utils', help='The path to save trained models.')
args = parser.parse_args()

# import from utils folder
sys.path.append(os.path.dirname(args.config_path))
from models import *
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
check_labels(images[0], labels[0], args.save_path)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available() # return True or False 
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
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
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
#optimizer = torch.optim.Adam(params,lr=CFG.learning_rate,weight_decay=1e-4)

print('the dataset has %d images' % (len(train_set)))

num_iter = 0
num_epoch = CFG.train_epochs
best_acc = 0.0

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    net.train()
    if epoch == 0:
        learning_rate = 0.0005
    elif epoch == 1:
        learning_rate = 0.00075
    elif epoch == 2:
        learning_rate = 0.001
    elif epoch == 30:
        learning_rate = 0.0001
    elif epoch == 40:
        learning_rate = 0.00001
    
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

            cv2.imwrite('save/epoch_'+str(epoch)+'_baseline_'+image_name.split('/')[-1],image)
            
            
    for i,(images,target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        if use_cuda:
            images, target = images.to(device), target.to(device)
        
        train_pred = net(images)
        batch_loss = criterion(train_pred, target)
        batch_loss.backward()
        optimizer.step()
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == target.cpu().numpy())
        train_loss += batch_loss.item() #.data
        
        progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
        print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
    
#         if (i+1) % 5 == 0:
#             print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
#             %(epoch+1, num_epochs, i+1, len(train_set_loader), loss.data, total_loss / (i+1)))
#             num_iter += 1

    #validation
    net.eval()
    with torch.no_grad(): # This will free the GPU memory used for back-prop

        for i,(images,target) in enumerate(valid_loader):
            
            if use_cuda:
                images, target = images.to(device), target.to(device)

            val_pred = net(images)
            batch_loss = criterion(val_pred,target)
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == target.cpu().numpy())
            val_loss += batch_loss.item()
            
            progress = ('#' * int(float(i)/len(valid_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
                    (time.time() - epoch_start_time), progress), end='\r', flush=True)
            
        val_acc = val_acc/valid_set.__len__()
        
        
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss, val_acc, val_loss))

    if (val_acc > best_acc):
        with open('save/acc.txt','w') as f:
            f.write(str(epoch)+'\t'+str(val_acc)+'\n')
        torch.save(model.state_dict(), args.save_path)  #'save/model.pth'
        best_acc = val_acc
        print ('Model Saved!')
        
        
#     if best_valid_loss > validation_loss:
        
#         best_valid_loss = validation_loss
#         print('get best test loss %.5f' % best_valid_loss)
#         torch.save(net.state_dict(), args.save_path) #'save/baseline_model.pth'
#         print ('Model Saved!')
        
#         # log
#         with open('save/log_baseline_model.txt','w') as f:
#             f.write(str(epoch)+'\t'+str(best_valid_loss)+'\n')
        

