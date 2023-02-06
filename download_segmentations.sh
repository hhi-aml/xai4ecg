#!/bin/bash

####################################
#   GET PTBXL SEGMENTATIONS FROM ZENODO
####################################

cd data/ptbxl/
wget https://zenodo.org/record/7610236/files/ptbxl_segmentations_8bit.zip
unzip ptbxl_segmentations_8bit.zip
mv ptbxl_segmentations_8bit ptbxl_segmentations
cd ../../