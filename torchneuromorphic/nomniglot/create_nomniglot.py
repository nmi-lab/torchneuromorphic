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
from torchneuromorphic.nomniglot.nomniglot_dataloaders import *

if __name__ == "__main__":
    
    #out = create_events_hdf5('/home/kennetms/Documents/torchneuromorphic/torchneuromorphic/dvssign/data/ASL-DVS', '/home/kennetms/Documents/torchneuromorphic/torchneuromorphic/dvssign/data/ASL-DVS/dvssign.hdf5')
    
    NOmniglotDataset = NOmniglotDataset(root='/home/kennetms/Documents/torchneuromorphic/torchneuromorphic/nomniglot/data/nomniglot/nomniglot.hdf5')
    

    f = h5py.File('/home/kennetms/Documents/torchneuromorphic/torchneuromorphic/nomniglot/data/nomniglot/nomniglot.hdf5', 'r')
    
    print(list(f.keys()))
    
    data = f['data']
    
    print(data['0'].keys())
    
    print("addrs",data['0']['addrs'][0])
    
    print("label",data['0']['labels'])
        
    print("time",data['0']['times'][0])


