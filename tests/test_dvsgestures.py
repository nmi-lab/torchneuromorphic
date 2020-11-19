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
from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
import torchneuromorphic.transforms as transforms

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader(
            root='data/dvsgesture/dvs_gestures_build19.hdf5',
            batch_size=64,
            ds=4,
            num_workers=0)
    ho = iter(train_dl)
    frames, labels = next(ho)
