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
from pytorch_datasets.dvsgestures.dvsgestures_dataloaders import *

if __name__ == "__main__":
    out = DVSGestursDataset(filename='/share/data/dvs_gestures_events.hdf5')
