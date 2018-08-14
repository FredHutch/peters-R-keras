#!/bin/bash

set -e # exit after any error
set -x # echo each command line

# we expect to find the following environment variables set:
# X_TRAIN_S3_URL
# X_TEST_S3_URL
# SCORE_OUTPUT_S3_URL

git clone https://github.com/dtenenba/peters-R-keras.git

cd peters-R-keras

aws s3 cp $X_TRAIN_S3_URL ./x_train.csv
aws s3 cp $X_TEST_S3_URL ./x_test.csv

R -f fitmodel.R

aws s3 cp score.csv $SCORE_OUTPUT_S3_URL



