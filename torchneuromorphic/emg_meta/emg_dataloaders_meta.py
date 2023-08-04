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
#from .create_hdf5_meta import create_events_hdf5
from torchneuromorphic.neuromorphic_dataset import NeuromorphicDataset 
from torchneuromorphic.events_timeslices import *
from torchneuromorphic.transforms import *
import os

import torchmeta
from torchmeta.transforms import Categorical

import pdb

mapping = { 0 :'Resting'  ,
            1 :'Hand Closing',
            2 :'Hand Opening' ,
            3 :'Wrist Pronation'   ,
            4 :'Wrist Supination'  ,
            5 :'Wrist Flexion'    ,
            6 :'Wrist Extension'   ,
            7 :'Thumb Flexion'       ,
            8 :'Thumb Extension'      ,
            9 :'Tridigit'
            }

train_mapping = {
    '01_03_2023': 0,
    '07_03_2023': 1,
    '04_04_2023': 2,
    '28_03_2023': 3,
    '21_02_2023': 4,
}

test_mapping = {
    '18_04_2023': 5,
    '28_03_2023_2': 6,
    '28_03_2023_3': 7,
    '14_03_2023': 8,
    '21_03_2023': 9,
}

class EMGMetaDataset(NeuromorphicDataset):

    def __init__(
            self, 
            root,
            train=True,
            target_transform=None,
            label=None):

        self.n = 0

        self.root = root
        self.train = train 

        
        self.label = label # If not None, do meta learning, so should not be None

        super(EMGMetaDataset, self).__init__(
                root,
                target_transform=target_transform )

        
        self.keys_by_label = {}
        
        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if train:
                self.n = f['extra'].attrs['Ntrain']
                self.keys = f['extra']['train_keys'][()]
                for k in f['train_keys_by_label'].keys():
                    self.keys_by_label[k] = f['train_keys_by_label'][k][()] 
            else:
                self.n = f['extra'].attrs['Ntest']
                self.keys = f['extra']['test_keys'][()]
                for k in f['test_keys_by_label'].keys():
                    self.keys_by_label[k] = f['test_keys_by_label'][k][()]
            self._num_classes = len(self.keys_by_label.keys())
                
                
        if label is not None:
            self.n = len(self.keys_by_label[str(self.label)])
            
    
    # assume this is already done, leave just in case
    def create_hdf5(self):
        create_events_hdf5(self.resources_local[0], self.root)

    def __len__(self):
        return self.n
        
    def __getitem__(self, index):
        
        if self.label != None:
            #print("N",self.n)
            #pdb.set_trace()
            ind = self.keys_by_label[str(self.label)][index%self.n]#//self.n]
            #print("THE KEY IS", key)
        #Important to open and close in getitem to enable num_workers>0
            with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
                if self.train:
                    key = f['extra']['train_keys'][ind]
                else:
                    key = f['extra']['test_keys'][ind]
                data, target = sample(
                        f,
                        key)
        else:
            with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
                if self.train:
                    key = f['extra']['train_keys'][index]
                else:
                    key = f['extra']['test_keys'][index]
                data, target = sample(
                        f,
                        key)
                
            f.close()


        return data, self.target_transform(target.astype('int'))
              



class ClassEMGMetaDataset(torchmeta.utils.data.ClassDataset):
    def __init__(self, root = 'emg_meta.hdf5',
                 meta_train=False,
                 meta_test=False,
                 meta_split=None,
                 target_transform=None,
                 class_augmentations=None):
        
        super(ClassEMGMetaDataset, self).__init__(
                meta_train=meta_train,
                meta_test=meta_test,
                meta_split=meta_split,
                class_augmentations=class_augmentations)
        
        
        self.root=root
        
        if meta_train is True:
            self.train = True
            self.test = False
        elif meta_test is True:
            self.train = False
            self.test = True
        
        if meta_train:
            split_name = 'train'
        if meta_test:
            split_name = 'test'
        self.split_name = split_name
        
        
        self.target_transform = target_transform
        
        self.dataset =   EMGMetaDataset(root =self.root, 
                                     label=None,
                                     target_transform = self.target_transform)

    @property
    def labels(self):
        return np.arange(self.n, dtype='int')

    @property
    def num_classes(self):
        return self.dataset._num_classes

    def __getitem__(self, index):
        label = index
        
        #print("label is", label)
        
        d = EMGMetaDataset(root =self.root, 
                                     label=label,
                                     target_transform = self.target_transform)
        d.index = index
        #d.target_transform_append = lambda x: None
        return d



class EMGMeta(torchmeta.utils.data.CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_test=False, meta_split=None, dataset_transform=None,
                 class_augmentations=None):

        
        target_transform = Categorical(num_classes_per_task)
            
        print("NUM PER TASK", num_classes_per_task)
            
        dataset = ClassEMGMetaDataset(root,
            meta_train=meta_train,
            meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        super(EMGMeta, self).__init__(dataset, 
                                           num_classes_per_task,
                                           target_transform=target_transform,
                                           dataset_transform=dataset_transform)
        



def create_class_dataset(dset, meta_split = 'train'):
    ds = []
    for n in range(dset.nclasses):
        indices = dset.keys_by_label[str(n)]
        d = torch.utils.data.Subset(dset, indices)
        ds.append(d)
    return ds



def sample(hdf5_file,
        key):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    data = dset['samples'][0]
    
    data = data.reshape((data.shape[2], data.shape[0], data.shape[3], data.shape[1]))

    return data, label

def create_dataloader(
        root = 'data/emg_meta.hdf5',
        batch_size = 1 ,
        target_transform_train = None,
        target_transform_test = None,
        sample_shuffle=True,
        **dl_kwargs):
     


    train_d = EMGMetaDataset(root,
                                train=True)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=sample_shuffle, **dl_kwargs)

    test_d = EMGMetaDataset(root,
                               train=False)

    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl



