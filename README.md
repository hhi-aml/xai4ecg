# XAI4ECG (Interpretability for ECG Analysis with Neural Networks)

This repository is related to our paper and allows for reproducing our results.


## Installation
Before the project can be used, the conda environment has to be created:

```bash
conda env create -f environment.yml
```

To use the environment it must be activate first. After using the environment it has to be deactivated:

```bash
conda activate xai4ecg
conda deactivate
```

## Download PTB-XL database and segmentations
Download PTB-XL via the follwing bash-script:

    ./download_data.sh

This script downloads [PTB-XL from PhysioNet](https://physionet.org/content/ptb-xl/) and stores at `data/ptbxl/`. In addition we provide soft-segmentations in order to reproduce our Glocal XAI results with segmented attributions, for this, please execute follwing bash-script:


    ./download_segmentations.sh

This will download [soft segmentations for PTB-XL 1.0.1]() from zenodo and stores at `data/ptbxl/ptbxl_segmentations/`

## 1. Preprocess Data
In order to perform analysis as in our paper, the data needs to be preprocessed with

### Load raw data and dump as numpy array and further enrich databse

```bash
python code/preprocessing.py
```
This will load the raw data into numpy array and stores it at `data/ptbxl/raw100.pkl` (in case of 100 Hz sampling frequency). Furthermore the mapping between labels and indices (as given in `code/utils.py`) is applied and stored as pickle dumps at `data/ptbxl/` for all possible tasks (`diagnostic`, `subdiagnostic`, `superdiagnostic`, `rhythm`, `form`, `all`) with `multihot_` as prefix and suffix `.npy`.

## 2. Train a model (optional)
In order to perform interpretability analysis as in our paper, a model needs to be trained via the following command:

```bash
python code/train_model.py --logdir=./output  --modelname=lenet --task=subdiagnostic --gpu
```
This will initialize and train a model (`resnet`, `xresnet` or `lenet`) on a specified task and stores it and its tensorboard logs in  `--logdir`.

We provide following regression-tasks for sanity checks:
```
    - T_Wave_Amplitude, P_Wave_Amplitude, R_Peak_Amplitude
```
and following multi-label-classification-tasks:
```
    - superdiagnostic, subdiagnostic, diagnostic, rhythm, form, all
```

## 3. Global TCAV Analysis
```bash
python code/tcav_analysis.py --modeltype=lenet --model_checkpoint_path=path/to/model.ckpt --logdir=path/to/output/
```
This will apply (relativ) TCAV for model (`--model_path`) and a list of concepts (`--concepts`) to a list of pathologies (`--pathologies`) for all layers and stores the results in `--output_path`.

## 4. Local XAI Analysis (Saliency, GradCAM, IG, LRP)
```
python code/lrp_analysis.py --output_path=output/ --model_path=output/bla. --pathologies=[blub1, blub2] --strategy=single_best
```
This will apply LRP for model (`--model_path`) and a list of pathologies (`--pathologies`) and stores the results in `--output_path`.


## Development

When you install new packages, please update the environment file like so:

```bash
conda env export | grep -v "prefix" > environment.yml
```

When someone else has added dependencies to the environment, then you can update your environment from the `environment.yml` like so (the `--prune` switch makes sure that dependencies that have been removed from the `environment.yml` are uninstalled):

```bash 
conda env update --file environment.yml --prune
