#!/bin/bash

####################################
#   GET PTBXL DATABASE
####################################
./download_data.sh

####################################
#   GET PTBXL SEGMENTATIONS
####################################
./download_segmentations.sh

####################################
#  Preprocess Data and Labels
####################################
python code/preprocessing.python

####################################
#  Train Regression Models for Sanity Checks
####################################
# P_Wave_Amplitude
python code/train_model.py --logdir=./output  --modelname=lenet --task=P_Wave_Amplitude --gpu
python code/train_model.py --logdir=./output  --modelname=xresnet --task=P_Wave_Amplitude --gpu
# R_Peak_Amplitude
python code/train_model.py --logdir=./output  --modelname=lenet --task=R_Peak_Amplitude --gpu
python code/train_model.py --logdir=./output  --modelname=xresnet --task=R_Peak_Amplitude --gpu
# T_Wave_Amplitude
python code/train_model.py --logdir=./output  --modelname=lenet --task=T_Wave_Amplitude --gpu
python code/train_model.py --logdir=./output  --modelname=xresnet --task=T_Wave_Amplitude --gpu

####################################
#  Train Classification Models for XAI
####################################
python code/train_model.py --logdir=./output  --modelname=lenet --task=subdiagnostic --gpu
python code/train_model.py --logdir=./output  --modelname=xresnet --task=subdiagnostic --gpu
python code/train_model.py --logdir=./output  --modelname=lenet --task=superdiagnostic --gpu
python code/train_model.py --logdir=./output  --modelname=xresnet --task=superdiagnostic --gpu

####################################
#  Global XAI (TCAV)
####################################
python code/tcav_analysis.py --modeltype=lenet --model_checkpoint_path=output/lenet_subdiagnostic/checkpoints/best_model.ckpt --logdir=output/lenet_subdiagnostic/
python code/tcav_analysis.py --modeltype=xresnet --model_checkpoint_path=output/xreset_subdiagnostic/checkpoints/best_model.ckpt --logdir=output/xresnet_subdiagnostic/