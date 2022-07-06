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
from torchneuromorphic.nomniglot.nomniglot_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root = 'nomniglot_all.hdf5' #'/home/kennetms/Documents/snn_maml/data/nomniglot/nomniglot.hdf5'
    train_dl, valid_dl, test_dl = create_dataloader(
        root=root,
        batch_size=8,
        ds=8,
        dt=30000,
        chunk_size_train = 100,
        chunk_size_test = 100,
        num_workers=0)
    
    iter_meta_train = iter(train_dl)
    iter_meta_valid = iter(valid_dl)
    iter_meta_test = iter(test_dl)
    
    # make sure can make it through all data
    for x, t in train_dl:
        print(t.shape)
        
    for x, t in valid_dl:
        print(t.shape)
        
    for x, t in test_dl:
        print(t.shape)
    
    frames_train, labels_train = next(iter_meta_train)
    frames_valid, labels_valid = next(iter_meta_valid)
    frames_test , labels_test  = next(iter_meta_test)
    
    with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if 1:
                keys = f['extra']['train_keys']
                print(keys)
            else:
                key = f['extra']['test_keys'][0]

    print(frames_train.shape)
    print(labels_train.shape)
    plot_frames_imshow(frames_test, labels_test, do1h=False, nim=1, avg=100)
    plt.savefig('nomniglot.png')
