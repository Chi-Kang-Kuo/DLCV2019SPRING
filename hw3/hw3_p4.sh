# TODO: create shell script for running your improved UDA model
# bash ./hw3_p4.sh $1 $2 $3
# bash ./hw3_p4.sh hw3_data/digits/mnistm/test mnistm hw3_data/digits/mnistm/test_pred.csv
# bash ./hw3_p4.sh hw3_data/digits/svhn/test svhn hw3_data/digits/svhn/test_pred.csv
# bash ./hw3_p4.sh hw3_data/digits/usps/test usps hw3_data/digits/usps/test_pred.csv
python3 DSN_predict.py $1 $2 $3