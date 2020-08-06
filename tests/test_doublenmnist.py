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
from torchneuromorphic.doublenmnist.doublenmnist_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow

if __name__ == "__main__":
    train_dl, test_dl = sample_double_mnist_task(
            meta_dataset_type = 'val',
            N = 4,
            K = 2,
            root='data/nmnist/n_mnist.hdf5',
            batch_size=10,
            ds=1,
            num_workers=0)
    iter_meta_train = iter(train_dl)
    iter_meta_test = iter(test_dl)
    frames_train, labels_train = next(iter_meta_train)
    frames_test , labels_test  = next(iter_meta_test)
