#!/bin/bash

####################################
#   GET PTBXL DATABASE
####################################
mkdir -p data
mkdir -p output
cd data
wget https://storage.googleapis.com/ptb-xl-1.0.1.physionet.org/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ptbxl
cd ..
