'''
generate fig1_2.jpg (fake image) from DCGAN
'''
import os
import sys
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help='The path of the config.py file.', default='config.py')
parser.add_argument('-ckpt', '--checkpoint', type=str, help='The path to load trained models.')
parser.add_argument('-save_img', '--save_img', type=str, help='The path to save output imgs.')
parser.add_argument('-utils', '--utils', type=str, default='./utils', help='The path to save trained models.')
args = parser.parse_args()

# import from utils folder
sys.path.insert(1, args.utils)
from utils import CelebA
from torch.utils.data import DataLoader

# import from config file
sys.path.insert(1, args.config)
import config
from DCGAN import Generator, Discriminator, weights_init
CFG = config.cfg
nc = CFG.nc
nz = CFG.nz
ngf = CFG.ngf
ndf = CFG.ndf
ngpu = CFG.ngpu
img_size = CFG.image_size

# make output folder
os.makedirs(args.save_img, exist_ok=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Initialize generator and discriminator
netG = Generator(nc, nz, ngf, ndf, ngpu).to(device)
netD = Discriminator(nc, nz, ngf, ndf, ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Print the model
print(netG)
print(netD)

# Load the model
checkpoint = torch.load(args.checkpoint)
netG.load_state_dict(checkpoint['state_dict'][0])
netD.load_state_dict(checkpoint['state_dict'][1])
netG.eval()
netD.eval()

# set random seed
manualSeed = 777
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(32, nz, 1, 1, device=device)

print("Starting inference...")

# Check how the generator is doing by saving G's output on fixed_noise
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
    
fake_img = vutils.make_grid(fake, padding=2, normalize=True)

# Plot the fake images from the last epoch
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(fake_img,(1,2,0)))
plt.savefig(os.path.join(args.save_img, 'fig1_2.jpg'))