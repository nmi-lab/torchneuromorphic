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
from pytorch_datasets.nmnist.create_hdf5 import *

if __name__ == "__main__":
    out = create_events_hdf5('data/N-MNIST/', 'data/N-MNIST/n_mnist.hdf5')




