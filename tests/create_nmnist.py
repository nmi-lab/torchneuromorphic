#!/bin/python
#-----------------------------------------------------------------------------
# File Name : create_dvsgestures.py
# Author: Emre Neftci
#
# Creation Date : Tue Nov  5 13:20:05 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from torchneuromorphic.nmnist.nmnist_dataloaders import *

if __name__ == "__main__":
    out = create_events_hdf5('data/nmnist/', 'data/nmnist/n_mnist.hdf5')




