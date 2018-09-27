#!/bin/bash

set -e # exit after any error
set -x # echo each command line

# we expect to find the following environment variables set:
# X_TRAIN_S3_URL
# X_TEST_S3_URL
# SCORE_OUTPUT_S3_URL

# And in the job definition for the python version only,
# we set USE_PYTHON to indicate that we will
# use Python instead of R.


git clone https://github.com/dtenenba/peters-R-keras.git

cd peters-R-keras

aws s3 cp $X_TRAIN_S3_URL ./x_train.csv
#aws s3 cp $X_TEST_S3_URL ./x_test.csv
aws s3 cp $X_TEST_uk_S3_URL ./test_uk.csv
#aws s3 cp $X_TEST_plco_S3_URL ./test_plco.csv
#aws s3 cp model_reg.hdf5 $MODEL_OUTPUT_S3_URL
if [ -z ${USE_PYTHON+x} ];
then
    echo USE_PYTHON is not set, using R
    time R -f fitmodel.R
    aws s3 cp history.reg.pdf $PDF_OUTPUT_S3_URL
    aws s3 cp model_reg.hdf5 $MODEL_OUTPUT_S3_URL
else
    echo USE_PYTHON is set, using python
    time python fitmodel.py
fi


aws s3 cp score.csv $SCORE_OUTPUT_S3_URL


