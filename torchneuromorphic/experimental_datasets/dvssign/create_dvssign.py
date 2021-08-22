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
from torchneuromorphic.dvssign.dvssign_dataloaders import *

if __name__ == "__main__":
    DVSSignDataset = DVSSignDataset(root='data/ASL-DVS/dvssign.hdf5')
    
    f = h5py.File('data/ASL-DVS/dvssign.hdf5', 'r')
    
    print(list(f.keys()))
    
    data = f['data']
    
    print(data['0'].keys())
    
    print("addrs",data['0']['addrs'][0])
    
    print("label",data['0']['labels'])
        
    print("time",data['0']['times'][0])


