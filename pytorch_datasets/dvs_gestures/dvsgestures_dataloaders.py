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
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
import os

mapping = { 0 :'Hand Clapping'  ,
            1 :'Right Hand Wave',
            2 :'Left Hand Wave' ,
            3 :'Right Arm CW'   ,
            4 :'Right Arm CCW'  ,
            5 :'Left Arm CW'    ,
            6 :'Left Arm CCW'   ,
            7 :'Arm Roll'       ,
            8 :'Air Drums'      ,
            9 :'Air Guitar'     ,
            10:'Other'}

class DVSGestureDataset(NeuromorphicDataset):

    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            ds = 2,
            size = [2, 64, 64],
            dt=1000, #transform
            download=False,
            batch_size = 72, 
            chunk_size = 500):
        super(DVSGestureDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform )


        if download:
            self.download()

        self.root = root

        self.train = train 
        self.dt = dt
        self.ds = ds
        self.size = size
        self.num_classes = 11
        self.batch_size = batch_size
        self.chunk_size = chunk_size


        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if train:
                self.n = f['extra'].attrs['Ntrain']
                self.keys = f['extra']['train_keys']
            else:
                self.n = f['extra'].attrs['Ntest']
                self.keys = f['extra']['test_keys']

    def download(self):
        raise NotImplementedError()

    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if not self.train:
                key = key + f['extra'].attrs['Ntrain']
            data, target = sample(
                    f,
                    key,
                    T = self.chunk_size,
                    n_classes = self.num_classes,
                    size = self.size,
                    ds = self.ds,
                    dt = self.dt)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

def sample(hdf5_file,
        key,
        T = 500,
        n_classes = 11,
        ds = 2,
        size = [2, 64, 64],
        dt = 1000):
    label1h = np.zeros(n_classes, dtype='float32')
    data = np.empty([T]+size, dtype='float32')
    dset = hdf5_file['data'][str(key)]
    label1h[dset['labels'][()]-1]=1
    tbegin = np.maximum(0,dset['times'][0]- 2*T*dt)
    tend = dset['times'][-1] 
    start_time = np.random.randint(tbegin, tend)
    data = get_event_slice(dset['times'][()], dset['addrs'][()], start_time, T, ds=ds, size=size, dt=dt)

    return data, label1h 

 
def create_dataloader(
        root = 'data/DvsGesture/dvs_gestures_build19.hdf5',
        batch_size = 72 ,
        chunk_size_train = 500,
        chunk_size_test = 1800,
        size = [2, 32, 32],
        ds = 4,
        dt = 1000,
        transform = None,
        target_transform_train = None,
        target_transform_test = None,
        **dl_kwargs):

    if not os.path.isfile(root):
        raise Exception("File {} does not exist".format(root))

    if transform is None:
        transform = ToTensor()
    if target_transform_train is None:
        target_transform_train = Repeat(chunk_size_train)
    if target_transform_test is None:
        target_transform_test = Repeat(chunk_size_test)

    train_d = DVSGestureDataset(root,
                                train=True,
                                transform = transform, 
                                target_transform = target_transform_train, 
                                ds = ds,
                                size = size,
                                dt = dt,
                                batch_size=batch_size,
                                chunk_size = chunk_size_train)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, **dl_kwargs)

    test_d = DVSGestureDataset(root,
                               transform = transform, 
                               target_transform = target_transform_test, 
                               train=False,
                               ds = ds,
                               size = size,
                               dt = dt,
                               batch_size=batch_size,
                               chunk_size = chunk_size_test)

    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl



