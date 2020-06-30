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
from .create_hdf5 import create_events_hdf5
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
    directory = 'data/dvsgesture/'
    resources_url = [['Manually Download dataset here: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/file/211521748942?sb=/details and place under {0}'.format(directory),None, 'DvsGesture.tar.gz']]
    resources_local = [directory+'raw']

    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            chunk_size = 500):

        self.n = 0
        self.download_and_create = download_and_create
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size

        super(DVSGestureDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform )

        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if train:
                self.n = f['extra'].attrs['Ntrain']
                self.keys = f['extra']['train_keys'][()]
            else:
                self.n = f['extra'].attrs['Ntest']
                self.keys = f['extra']['test_keys'][()]

    def download(self):
        super(DVSGestureDataset, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.resources_local[0], self.root)

    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if not self.train:
                key = key + f['extra'].attrs['Ntrain']
            assert key in self.keys
            data, target = sample(
                    f,
                    key,
                    T = self.chunk_size,
                    shuffle=self.train)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

def sample(hdf5_file,
        key,
        T = 500,
        shuffle = False):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tbegin = dset['times'][0]
    tend = np.maximum(0,dset['times'][-1]- 2*T*1000 )
    start_time = np.random.randint(tbegin, tend) if shuffle else 0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad[:, [0,3,1,2]], label

 
def create_dataloader(
        root = 'data/dvsgesture/dvs_gestures_build19.hdf5',
        batch_size = 72 ,
        chunk_size_train = 500,
        chunk_size_test = 1800,
        ds = None,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        n_events_attention=None,
        **dl_kwargs):
    if ds is None:
        ds = 4
    if isinstance(ds,int):
        ds = [ds,ds]
        
    size = [2, 128//ds[0], 128//ds[1]]

    if n_events_attention is None:
        default_transform = lambda chunk_size: Compose([
            Downsample(factor=[dt,1,ds[0],ds[1]]),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()
        ])
    else:
        default_transform = lambda chunk_size: Compose([
            Downsample(factor=[dt,1,1,1]),
            Attention(n_events_attention, size=size),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()
        ])

    if transform_train is None:
        transform_train = default_transform(chunk_size_train)
    if transform_test is None:
        transform_test = default_transform(chunk_size_test)

    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(11)])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(11)])

    train_d = DVSGestureDataset(root,
                                train=True,
                                transform = transform_train, 
                                target_transform = target_transform_train, 
                                chunk_size = chunk_size_train)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True, **dl_kwargs)

    test_d = DVSGestureDataset(root,
                               transform = transform_test, 
                               target_transform = target_transform_test, 
                               train=False,
                               chunk_size = chunk_size_test)

    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl



