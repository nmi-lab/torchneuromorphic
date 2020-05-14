#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Jacques Kaiser
#
# Creation Date : Wed 13 May 2020
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------
import h5py
import torch.utils.data
import numpy as np
import glob
import os

from .create_hdf5 import create_events_hdf5
from ..neuromorphic_dataset import NeuromorphicDataset
from ..events_timeslices import *
from ..transforms import *

class RosbagDataset(NeuromorphicDataset):
    def __init__(
            self,
            root,
            resources_local=None,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            chunk_size = 500):

        self.directory = 'data/rosbags/'
        if resources_local is None:
            resources_local = [os.path.join(self.directory, 'raw')]
        self.resources_local = resources_local
        no_rosbag_found = 'No rosbag found. Place your rosbags in {}/{}'.format(self.directory, self.resources_local[0])
        self.resources_url = [[no_rosbag_found, None, 'your rosbags']]

        self.n = 0
        self.download_and_create = download_and_create
        self.root = root
        self.train = train
        self.chunk_size = chunk_size

        super(RosbagDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform )

        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            self.label_order = f['extra']['label_order'][()]
            print("Labels in order: {}".format(self.label_order))
            self.n_labels = len(self.label_order)
            if train:
                self.n = f['extra'].attrs['Ntrain']
                self.keys = f['extra']['train_keys'][()]
            else:
                self.n = f['extra'].attrs['Ntest']
                self.keys = f['extra']['test_keys'][()]

    def download(self):
        super(RosbagDataset, self).download()

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
    tend = dset['times'][-1]
    start_time = 0
    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad, label


def create_dataloader(
        root = 'data/rosbags/rosbags_build.hdf5',
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
    size = [2, 128//ds, 128//ds]
    center = [64,64]

    default_transform = lambda chunk_size: Compose([
        CropCenter(center, size),
        Downsample(factor=[dt,1,1,1]),
        ToCountFrame(T = chunk_size, size = size),
        ToTensor()
    ])

    if transform_train is None:
        transform_train = default_transform(chunk_size_train)
    if transform_test is None:
        transform_test = default_transform(chunk_size_test)

    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(4)]) # HACK HACK HACK! Can't get n_labels from here
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(4)])

    train_d = RosbagDataset(root,
                            train=True,
                            transform = transform_train,
                            target_transform = target_transform_train,
                            chunk_size = chunk_size_train)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True, **dl_kwargs)

    test_d = RosbagDataset(root,
                           transform = transform_test,
                           target_transform = target_transform_test,
                           train=False,
                           chunk_size = chunk_size_test)

    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl
