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
import torchmeta
from torchmeta.transforms import Categorical

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

# Splits used for the double MNIST task https://github.com/shaohua0116/MultiDigitMNIST
splits = {}
splits['train'] = ['00', '01', '04', '05', '06', '08', '09', '11', '12', '13', '14', '15', '16', '18', '19', '20', '21', '23', '24', '26', '28', '29', '30', '31', '33', '35', '37', '38', '41', '42', '43', '44', '45', '50', '51', '53', '54', '56', '59', '60', '62', '63', '65', '69', '70', '72', '74', '75', '76', '77', '79', '81', '82', '84', '85', '87', '88', '89', '90', '91', '94', '95', '97', '98']
splits['val'] = ['03', '07', '10', '22', '27', '34', '39', '40', '48', '52', '58', '61', '64', '71', '93', '99']
splits['test'] = ['02', '17', '25', '32', '36', '46', '47', '49', '55', '57', '66', '67', '68', '73', '78', '80', '83', '86', '92', '96']

class DoubleNMNISTClassDataset(NeuromorphicDataset):
    def __init__(
            self, 
            root : str,
            train : bool = True,
            transform : object = None ,
            target_transform=None,
            download_and_create=True,
            chunk_size = 500,
            samples_per_class = 1,
            label_u = 0):

        self.transform = transform
        self.target_transform = target_transform
        
        self.samples_per_class = samples_per_class
        self.download_and_create = download_and_create
        self.root = root
        self.train = train 

        self.chunk_size = chunk_size
        self.label_u = label_u


        lu = self.label_u
        self.labels_left = lu // 10
        self.labels_right = lu % 10

        ll = self.labels_left
        lr = self.labels_right

        #self.labels.append( np.repeat(self.labels_u, self.samples_per_class))
        #self.labels_map.append( dict(zip(np.unique(self.labels[i]),np.arange(self.nclasses))))

        super(DoubleNMNISTClassDataset, self).__init__(root = None,
                                            transform=transform,
                                            target_transform=target_transform )
       
        self.data_orig = NMNISTDataset( root,
                      train=train,
                      transform=transform,
                      target_transform=None,
                      download_and_create=download_and_create,                    
                      chunk_size = chunk_size)

        self.nl = len(self.data_orig.keys_by_label[ll])
        self.nr = len(self.data_orig.keys_by_label[lr])
        self.n = self.nl * self.nr


    def __len__(self):
        return 1000 #self.n

    def __getitem__(self, key):
        ll = self.labels_left
        lr = self.labels_right
        key_l = self.data_orig.keys_by_label[ll][ key // self.nl] 
        key_r = self.data_orig.keys_by_label[lr][ key % self.nl ] 

        data_l, label_l =  self.data_orig[key_l]
        data_r, label_r =  self.data_orig[key_r]

        size_x, size_y = data_r.shape[2:4]
        data = torch.zeros(data_r.shape[:2]+(size_x*2,size_y))
        data[:, :, :size_x, :] = data_l
        data[:, :, size_x:, :] = data_r
        target = self.label_u
        #Note that data is already transformed in the base dataset class (data_orig)
        return data, self.target_transform(target)




class ClassNMNISTDataset(torchmeta.utils.data.ClassDataset):
    def __init__(self, root = 'data/nmnist/n_mnist.hdf5', chunk_size=300, meta_train=False, meta_val=False, meta_test=False, meta_split=None, transform=None, target_transform=None, download=False, class_augmentations=None):
        self.root=root
        self.chunk_size = chunk_size
        if meta_train is True:
            train = True
        else:
            train = False
        
        if meta_train:
            split_name = 'train'
        if meta_val:
            split_name = 'val'
        if meta_test:
            split_name = 'test'
        self.split_name = split_name

        self.transform = transform
        self.target_transform = target_transform


        super(ClassNMNISTDataset, self).__init__(
                meta_train=meta_train,
                meta_val=meta_val,
                meta_test=meta_test,
                meta_split=meta_split,
                class_augmentations=class_augmentations)


        self._labels = [int(s) for s in splits[split_name]]
        self._num_classes = len(self._labels)

    @property
    def labels(self):
        return self._labels

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        label = self._labels[index]
        d = DoubleNMNISTClassDataset(root =self.root, 
                                     train= self.meta_train, 
                                     label_u = label, 
                                     transform = self.transform, 
                                     target_transform = self.target_transform, 
                                     chunk_size = self.chunk_size)
        d.index = index
        #d.target_transform_append = lambda x: None
        return d

def create_class_dataset(dset, meta_split = 'train'):
    ds = []
    for n in range(dset.nclasses):
        indices = dset.keys_by_label[n]
        d = torch.utils.data.Subset(dset, indices)
        ds.append(d)
    return ds

class DoubleNMNIST(torchmeta.utils.data.CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False,chunk_size=300):

        if target_transform is None:
            target_tranform = Categorical(num_classes_per_task)
            
        dataset = ClassNMNISTDataset(root,
            meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download,chunk_size=chunk_size)

        super(DoubleNMNIST, self).__init__(dataset, 
                                           num_classes_per_task,
                                           target_transform=target_transform,
                                           dataset_transform=dataset_transform)


