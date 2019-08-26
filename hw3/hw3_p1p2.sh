# TODO: create shell script for running your GAN/ACGAN model
# bash ./hw3_p1p2.sh $1
python3 dcgan_inference.py --output-file=$1
python3 acgan_inference.py --output-file=$1