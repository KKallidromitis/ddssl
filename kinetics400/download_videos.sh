#!/usr/bin/env bash

# set up environment
conda env create -f environment.yml
source activate kinetics
pip3 install --upgrade youtube-dl

DATA_DIR="./"
ANNO_DIR="./annotations"
python download.py ${ANNO_DIR}/kinetics_train.csv ${DATA_DIR}/videos_train
python download.py ${ANNO_DIR}/kinetics_val.csv ${DATA_DIR}/videos_val

source deactivate kinetics
conda remove -n kinetics --all