#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_nmnist.py
# Author: Emre Neftci
#
# Creation Date : Thu Nov  7 20:30:14 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from pytorch_datasets.nmnist.nmnist_dataloaders import *
from pytorch_datasets.utils import plot_frames_imshow

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader(
            root='data/nmnist/n_mnist.hdf5',
            batch_size=32,
            ds=1,
            num_workers=0)
    ho = iter(train_dl)
    frames, labels = next(ho)
