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
import h5py
import torch.utils.data
from ..nmnist.nmnist_dataloaders import NMNISTDataset, sample, create_datasets
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
import os

mapping = { 0 :'0',
            1 :'1',
            2 :'2',
            3 :'3',
            4 :'4',
            5 :'5',
            6 :'6',
            7 :'7',
            8 :'8',
            9 :'9'}

class DoubleNMNISTDataset(NeuromorphicDataset):

    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            chunk_size = 500,
            nclasses = 5,
            samples_per_class = 2,
            labels_u = range(5)):

        self.n = samples_per_class * nclasses
        self.nclasses = nclasses
        self.download_and_create = download_and_create
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size
        self.labels_u = labels_u
        labels_left =  self.labels_u // 10
        labels_right =  self.labels_u % 10
        self.labels = np.repeat(self.labels_u, samples_per_class)
        self.labels_map =  dict(zip(np.unique(self.labels),np.arange(nclasses)))


        super(DoubleNMNISTDataset, self).__init__(root = None,
                                            transform=transform,
                                            target_transform=target_transform )
       
        self.data_train = NMNISTDataset( root,
                      train=True,
                      transform=transform,
                      target_transform=target_transform,
                      download_and_create=download_and_create,                    
                      chunk_size = chunk_size)


        keys_filt_left  = np.array([ np.random.choice(s, samples_per_class) for s in self.data_train.keys_by_label[labels_left]]).reshape(-1)
        keys_filt_right = np.array([ np.random.choice(s, samples_per_class) for s in self.data_train.keys_by_label[labels_right]]).reshape(-1)
        self.keys = list(zip(keys_filt_left, keys_filt_right))

    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        key_l, key_r = self.keys[key]
        data_l, label_l =  self.data_train[key_l]
        data_r, label_r =  self.data_train[key_r]
        size_x, size_y = data_r.shape[2:4]
        data = torch.zeros(data_r.shape[:2]+(size_x*2,size_y*2))
        r1 = np.random.randint(0,size_y)
        r2 = np.random.randint(0,size_y)
        data[:, :, :size_x, r1:r1+size_y] = data_l
        data[:, :, size_x:, r2:r2+size_y] = data_r
        target = self.labels_map[self.labels[key]]
        return data, target

def create_datasets(
        root = 'data/nmnist/n_mnist.hdf5',
        batch_size = 72 ,
        chunk_size_train = 300,
        chunk_size_test = 300,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        nclasses = 5,
        samples_per_class = 2,
        samples_per_test = 2,
        classes_meta = np.arange(100, dtype='int')):

    size = [2, 32//ds, 32//ds]

    if transform_train is None:
        transform_train = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_train, size = size),
            ToTensor()])
    if transform_test is None:
        transform_test = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()])
    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(nclasses)])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(nclasses)])


    labels_u = np.random.choice(classes_meta, nclasses,replace=False) #100 here becuase we have two pairs of digits between 0 and 9

    train_ds = DoubleNMNISTDataset(root,train=True,
                                 transform = transform_train, 
                                 target_transform = target_transform_train, 
                                 chunk_size = chunk_size_train,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_class,
                                 labels_u = labels_u)

    test_ds = DoubleNMNISTDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 chunk_size = chunk_size_test,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_test,
                                 labels_u = labels_u)

    return train_ds, test_ds


def create_dataloader(
        root = 'data/nmnist/n_mnist.hdf5',
        batch_size = 72 ,
        chunk_size_train = 300,
        chunk_size_test = 300,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        nclasses = 5,
        samples_per_class = 2,
        samples_per_test = 2,
        classes_meta = np.arange(100, dtype='int'),
        **dl_kwargs):


    train_d, test_d = create_datasets(
        root = root,
        batch_size = batch_size,
        chunk_size_train = chunk_size_train,
        chunk_size_test = chunk_size_test,
        ds = ds,
        dt = dt,
        transform_train = transform_train,
        transform_test = transform_test,
        target_transform_train = target_transform_train,
        target_transform_test = target_transform_test,
        classes_meta = classes_meta,
        nclasses = nclasses,
        samples_per_class = samples_per_class,
        samples_per_test = samples_per_test)


    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl




def sample_double_mnist_task( N = 5,
                              K = 2,
                              K_test = 2,
                              meta_split = [range(64), range(64,80), range(80,100)],
                              meta_dataset_type = 'train',
                              **kwargs):
    classes_meta = {}
    classes_meta['train'] = np.array(meta_split[0], dtype='int')
    classes_meta['val']   = np.array(meta_split[1], dtype='int')
    classes_meta['test']  = np.array(meta_split[2], dtype='int')

    assert meta_dataset_type in ['train', 'val', 'test']
    return create_dataloader(classes_meta = classes_meta[meta_dataset_type], nclasses= N, samples_per_class = K, samples_per_test = K_test, **kwargs)
