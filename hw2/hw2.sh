# TODO: create shell script for running your YoloV1-vgg16bn model
# bash ./hw2.sh ./hw2_train_val/val1500/images ./save/vgg16_bn/predict/
python3 predict.py -img=$1 -save=$2 -cfg=./models/vgg16_bn/config.py -model=./save/vgg16_bn/epoch_59_loss_547.597.pth 