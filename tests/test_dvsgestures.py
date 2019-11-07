#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_dvsgestures.py
# Author: Emre Neftci
#
# Creation Date : Fri 19 Sep 2019 
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv3
#----------------------------------------------------------------------------- 
from pytorch_datasets.dvs_gestures.dvsgestures_dataloaders import *
from pytorch_datasets.utils import plot_frames_imshow
import pytorch_datasets.transforms as transforms

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader(
            root='data/DvsGesture/dvs_gestures_build19.hdf5',
            batch_size=32,
            num_workers=0,
            transform=transforms.ToTensor())
    ho = iter(test_dl)
    i=0
    for data,label in ho:
        print('hi', i)
        i+=1
    #ad,tm = next(ho)
