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
from .create_hdf5_meta import create_events_hdf5
from torchneuromorphic.neuromorphic_dataset import NeuromorphicDataset 
from torchneuromorphic.events_timeslices import *
from torchneuromorphic.transforms import *
import os

import torchmeta
from torchmeta.transforms import Categorical

import pdb

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

train_mapping = {
    'user01': 0,
    'user02': 1,
    'user03': 2,
    'user04': 3,
    'user05': 4,
    'user06': 5,
    'user07': 6,
    'user08': 7,
    'user09': 8,
    'user10': 9,
    'user11': 10,
    'user12': 11,
    'user13': 12,
    'user14': 13,
    'user15': 14,
    'user16': 15,
    'user17': 16,
    'user18': 17,
    'user19': 18,
    'user20': 19,
    'user21': 20,
    'user22': 21,
    'user23': 22,
}

test_mapping = {
    'user24': 23,
    'user25': 24,
    'user26': 25,
    'user27': 26,
    'user28': 27,
    'user29': 28,
}

class DVSGestureMetaDataset(NeuromorphicDataset):
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
            chunk_size = 500,
            return_meta = False,
            time_shuffle=False,
            label=None):

        self.n = 0
        self.download_and_create = download_and_create
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size
        self.return_meta = return_meta
        self.time_shuffle = time_shuffle
        
        self.label = label # If not None, do meta learning, so should not be None

        super(DVSGestureMetaDataset, self).__init__(
                root,
                transform=transform,
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
            

    def download(self):
        super(DVSGestureMetaDataset, self).download()

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
                        key,
                        chunk_size = self.chunk_size)
        else:
            with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
                if self.train:
                    key = f['extra']['train_keys'][index]
                else:
                    key = f['extra']['test_keys'][index]
                data, target = sample(
                        f,
                        key,
                        chunk_size = self.chunk_size)
                
            f.close()

        if self.transform is not None:
            data = self.transform(data)

        return data, self.target_transform(target.astype('int'))
              



class ClassDVSGestureMetaDataset(torchmeta.utils.data.ClassDataset):
    def __init__(self, root = 'dvs_gesture_meta.hdf5',
                 chunk_size=300,
                 meta_train=False,
                 meta_test=False,
                 meta_split=None,
                 transform=None,
                 target_transform=None,
                 download=False,
                 class_augmentations=None):
        
        super(ClassDVSGestureMetaDataset, self).__init__(
                meta_train=meta_train,
                meta_test=meta_test,
                meta_split=meta_split,
                class_augmentations=class_augmentations)
        
        
        self.root=root
        self.download = download
        
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
        
        self.transform = transform
        
        self.chunk_size = chunk_size        
        
        self.target_transform = target_transform
        
        self.dataset =   DVSGestureMetaDataset(root =self.root, 
                                     label=None,
                                     transform = self.transform, 
                                     target_transform = self.target_transform, 
                                     chunk_size = self.chunk_size)

    @property
    def labels(self):
        return np.arange(self.n, dtype='int')

    @property
    def num_classes(self):
        return self.dataset._num_classes

    def __getitem__(self, index):
        label = index
        
        #print("label is", label)
        
        d = DVSGestureMetaDataset(root =self.root, 
                                     label=label,
                                     transform = self.transform, 
                                     target_transform = self.target_transform, 
                                     chunk_size = self.chunk_size)
        d.index = index
        #d.target_transform_append = lambda x: None
        return d



class DVSGestureMeta(torchmeta.utils.data.CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False,chunk_size=300):

        if target_transform is None:
            target_tranform = Categorical(num_classes_per_task)
            
        print("NUM PER TASK", num_classes_per_task)
            
        dataset = ClassDVSGestureMetaDataset(root,
            meta_train=meta_train,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download,chunk_size=chunk_size)

        super(DVSGestureMeta, self).__init__(dataset, 
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
        key,
        chunk_size = 500,
        shuffle = False):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tbegin = dset['times'][0]
    tend = np.maximum(0,dset['times'][-1]- 2*chunk_size*1000 )
    start_time = np.random.randint(tbegin, tend+1) if shuffle else 0
    #print(start_time)
    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, chunk_size*1000)
    tmad[:,0]-=tmad[0,0]
    meta = eval(dset.attrs['meta_info'])
    return tmad[:, [0,3,1,2]], label#, meta['light condition'], meta['gesture']

def create_dataloader(
        root = 'data/dvs_gesture_meta.hdf5',
        batch_size = 1 ,
        chunk_size_train = 500,
        chunk_size_test = 1800,
        ds = None,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        n_events_attention=None,
        return_meta=False,
        sample_shuffle=True,
        time_shuffle=True,
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

    # if target_transform_train is None:
    #     target_transform_train =Compose([Repeat(chunk_size_train), toOneHot()])
    # if target_transform_test is None:
    #     target_transform_test = Compose([Repeat(chunk_size_test), toOneHot()])

    train_d = DVSGestureMetaDataset(root,
                                train=True,
                                transform = transform_train, 
                                target_transform = target_transform_train, 
                                chunk_size = chunk_size_train,
                                return_meta = return_meta,
                                time_shuffle=time_shuffle)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=sample_shuffle, **dl_kwargs)

    test_d = DVSGestureMetaDataset(root,
                               transform = transform_test, 
                               target_transform = target_transform_test, 
                               train=False,
                               chunk_size = chunk_size_test,
                               return_meta = return_meta,
                               time_shuffle=time_shuffle) # WAS FALSE

    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl



