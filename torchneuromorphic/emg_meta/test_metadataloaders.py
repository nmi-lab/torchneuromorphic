#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_nmnist.py
# Author: Kenneth Stewart
#
# Copyright : (c) UC Regents, Kenneth Stewart
# Licence : Apache License version 2.0
#----------------------------------------------------------------------------- 
from emg_dataloaders_meta import *
from torchneuromorphic.utils import plot_frames_imshow
import matplotlib.pyplot as plt

# need a later version of lava-dl to use lava code that michael wrote, uses learning dense layer

if __name__ == "__main__":
    root = '/Users/k_stewart/chouti/data/emg_meta.hdf5' 
    train_dl, test_dl = create_dataloader(
        root=root,
        batch_size=5,
        num_workers=0)
    
    iter_meta_train = iter(train_dl)
    iter_meta_test = iter(test_dl)
    
    # make sure can make it through all data
#     for x, t in train_dl:
#         print(t.shape)
#         print(x.shape)
#         i=1/0
        
#     for x, t in test_dl:
#         print(t.shape)
    
    frames_train, labels_train = next(iter_meta_train)
    frames_test , labels_test  = next(iter_meta_test)
    
    with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if 1:
                keys = f['extra']['train_keys']
                print(keys)
            else:
                key = f['extra']['test_keys'][0]

    print(frames_train.shape)
    print(labels_train.shape)
    # plot_frames_imshow(frames_test, labels_test, do1h=False, nim=1, avg=100)
    # plt.savefig('meta.png')
