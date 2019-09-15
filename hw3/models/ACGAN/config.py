from easydict import EasyDict as edict

# use 'from config import cfg' to get these parameters 
cfg = edict()

# training settings
cfg.batch_size = 128
cfg.learning_rate = 0.0001
cfg.num_epochs = 20

# Beta1 hyperparam for Adam optimizers
cfg.beta1 = 0.5
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
cfg.image_size = 64

# Number of channels in the training images. For color images this is 3
cfg.nc = 3
# Size of z latent vector (i.e. size of generator input)
cfg.nz = 100
# Size of feature maps in generator
cfg.ngf = 64
# Size of feature maps in discriminator
cfg.ndf = 64
# Number of GPUs available. Use 0 for CPU mode.
cfg.ngpu = 1