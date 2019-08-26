# TODO: create shell script for Problem 1
# bash ./hw4_p1.sh $1 $2 $3   hw4_data/TrimmedVideos/video/valid
# bash ./hw4_p1.sh hw4_data/TrimmedVideos/video/valid hw4_data/TrimmedVideos/label/gt_valid.csv ./output/
python3 p1_inference.py $1 $2 $3
