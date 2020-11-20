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
from torchneuromorphic.dvssign.dvssign_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader(
            root='data/ASL-DVS/dvssign.hdf5',
            batch_size=8,
            ds=1,
            chunk_size_train = 100,
            chunk_size_test = 100,
            num_workers=0)
    
    iter_meta_train = iter(train_dl)
    iter_meta_test = iter(test_dl)
    frames_train, labels_train = next(iter_meta_train)
    frames_test , labels_test  = next(iter_meta_test)

    print(frames_train.shape)
    print(labels_train.shape)
    plot_frames_imshow(frames_train, labels_train, do1h=False, nim=4, avg=25)
    plt.savefig('dvssigns4.png')
