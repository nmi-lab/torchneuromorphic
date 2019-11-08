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
from torchneurmorphic.dvs_gestures.dvsgestures_dataloaders import *
from torchneurmorphic.utils import plot_frames_imshow
import torchneuromorphic.transforms as transforms

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader(
            root='data/dvsgesture/dvs_gestures_build19.hdf5',
            batch_size=32,
            ds=2,
            num_workers=2)
    ho = iter(test_dl)
    frames, labels = next(ho)
