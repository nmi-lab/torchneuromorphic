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
import time, copy
import numpy as np
import scipy.misc
import torch.utils.data
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
import os
import torchneuromorphic.randman.randman_gen as randman



class RandManDataset(NeuromorphicDataset):
    resources_url = []
    directory = 'data/nmnist/'
    resources_local = [directory+'Train', directory+'Test']

    def __init__(
            self, 
            seed=None,
            train=True,
            transform=None,
            target_transform=None,
            chunk_size = 100):

        nb_classes = 5
        nb_units = 100
        nb_samples = 100

        data,labels = randman.make_tempo_randman(
                dim_manifold=1,
                nb_classes=nb_classes,
                nb_units=nb_units,
                nb_steps=chunk_size,
                step_frac=0.5,
                nb_samples=nb_samples,
                nb_spikes=2,
                alpha=1.0,
                seed = seed,
                shuffle = True)

        self.data = data
        self.labels = labels

        self.nclasses = self.num_classes = 10
        self.chunk_size = chunk_size

        super(RandManDataset, self).__init__(
                None,
                transform=transform,
                target_transform=target_transform )
        


    def download(self):
        #isexisting = super(NMNISTDataset, self).download()
        pass

    def create_hdf5(self):
        #create_events_hdf5(self.directory, self.root)
        pass


    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        x_ = self.data[key]
        y_ = self.labels[key]

        if self.transform is not None:
            x_ = self.transform(x_)

        if self.target_transform is not None:
            y_ = self.target_transform(y_)
        return x_, y_

def create_dataloader(
        seed = None,
        batch_size = 72 ,
        chunk_size = 100, 
        transform_train = None,
        target_transform_train = None):

        #train_ds, test_ds = [storkdatasets.RasDataset(ds,**ds_kwargs) for ds in storkdatasets.split_dataset(data, labels, splits=[0.8, 0.2], shuffle=False)]

    if transform_train is None:
        transform_train = Compose([
            ToCountFrame(T = chunk_size, size = [100]),
            ToTensor()])

    if target_transform_train is None:
        target_transform_train =Compose([])

    dataset = RandManDataset(seed=None,transform=transform_train, target_transform=target_transform_train)
    #TODO
    # - split
    # create data loader
    # test
    create_dataloader(, batch_size = batch_size)
    return 


        

#def sample(hdf5_file,
#        key,
#        T = 300):
#    dset = hdf5_file['data'][str(key)]
#    label = dset['labels'][()]
#    tend = dset['times'][-1] 
#    start_time = 0
#    ha = dset['times'][()]
#
#    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
#    tmad[:,0]-=tmad[0,0]
#    return tmad, label
#
#def create_datasets(
#        root = 'data/nmnist/n_mnist.hdf5',
#        batch_size = 72 ,
#        chunk_size_train = 300,
#        chunk_size_test = 300,
#        ds = 1,
#        dt = 1000,
#        transform_train = None,
#        transform_test = None,
#        target_transform_train = None,
#        target_transform_test = None):
#
#    size = [2, 32//ds, 32//ds]
#
#    if transform_train is None:
#        transform_train = Compose([
#            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
#            Downsample(factor=[dt,1,1,1]),
#            ToCountFrame(T = chunk_size_train, size = size),
#            ToTensor()])
#    if transform_test is None:
#        transform_test = Compose([
#            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
#            Downsample(factor=[dt,1,1,1]),
#            ToCountFrame(T = chunk_size_test, size = size),
#            ToTensor()])
#    if target_transform_train is None:
#        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(10)])
#    if target_transform_test is None:
#        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(10)])
#
#    train_ds = NMNISTDataset(root,train=True,
#                                 transform = transform_train, 
#                                 target_transform = target_transform_train, 
#                                 chunk_size = chunk_size_train,
#                                 dt = dt)
#
#    test_ds = NMNISTDataset(root, transform = transform_test, 
#                                 target_transform = target_transform_test, 
#                                 train=False,
#                                 chunk_size = chunk_size_test,
#                                 dt = dt)
#
#    return train_ds, test_ds
#
#def create_dataloader(
#        root = 'data/nmnist/n_mnist.hdf5',
#        batch_size = 72 ,
#        chunk_size_train = 300,
#        chunk_size_test = 300,
#        ds = 1,
#        dt = 1000,
#        transform_train = None,
#        transform_test = None,
#        target_transform_train = None,
#        target_transform_test = None,
#        **dl_kwargs):
#
#    train_d, test_d = create_datasets(
#        root = root,
#        batch_size = batch_size,
#        chunk_size_train = chunk_size_train,
#        chunk_size_test = chunk_size_test,
#        ds = ds,
#        dt = dt,
#        transform_train = transform_train,
#        transform_test = transform_test,
#        target_transform_train = target_transform_train,
#        target_transform_test = target_transform_test)
#
#
#    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
#    test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)
#
#    return train_dl, test_dl



