# TODO: create shell script for running your GAN/ACGAN model
# bash ./hw3_p1p2.sh $1  
# $1 is the folder to which you should output your fig1_2.jpg and fig2_2.jpg.
python3 dcgan_inference.py -save_img=$1 -cfg=./models/GAN/ -ckpt=./save/dcgan/models/epoch10_checkpoint.pth.tar
#python3 acgan_inference.py -save_img=$1