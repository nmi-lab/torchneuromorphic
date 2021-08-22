#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : Fri 01 Dec 2017 10:05:17 PM PST
# Last Modified : Sun 29 Jul 2018 01:39:06 PM PDT
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------
import struct
import time
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from .download_create import create_events_hdf5
from torhneuromorphic.neuromorphic_dataset import NeuromorphicDataset 
from torhneuromorphic.events_timeslices import *
from torhneuromorphic.transforms import *
import os


class AedatDataset(NeuromorphicDataset):
    directory = 'data/aedat/'
    resources_url = [['',None, '']]
    resources_local = []

    def __init__(
            self, 
            root,
            name=None,
            transform=None,
            target_transform=None,
            time_window_size = 500):

        self.n = 0
        self.root = root
        self.time_window_size = time_window_size

        super(AedatDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform )

        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            self.n = f['extra'].attrs['N']
            self.keys = f['extra']['keys'][()]

    def download(self):
        super(DVSGestureDataset, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.resources_local[0], self.root)

    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            assert key < len(self.keys)
            data, target, meta_info = random_sample_spiketrain(
                    f,
                    key,
                    T = self.time_window_size,
                    shuffle=False)
            target = self.keys[key]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_meta is True:
            return data, target, meta_info
        else:
            return data, target

def random_sample_spiketrain(hdf5_file,
        key,
        T = 500,
        shuffle = False):
    dset = hdf5_file['data'][str(key)]
    tbegin = dset['times'][0]
    tend = np.maximum(0,dset['times'][-1]- 2*T*1000 )
    start_time = np.random.randint(tbegin, tend) if shuffle else 0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad[:, [0,3,1,2]], None, dset.attrs['meta_info']


