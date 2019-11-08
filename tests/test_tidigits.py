#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_tidigits.py
# Author: Emre Neftci
#
# Creation Date : Fri 13 Sep 2019 06:44:14 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from torchneuromorphic.ntidigits.ntidigits_dataloaders import *

if __name__ == '__main__':
    train_dl, test_dl = create_data(filename = 'data/tidigits/n-tidigits.hdf5',
        chunk_size_train=1000, chunk_size_test=1000, batch_size=50, size=[64], dt = 1000, ds = [1], num_workers=5)
    data_batch, target_batch = next(iter(train_dl))
    
