#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Kenneth Stewart and Emre Neftci
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
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
from .create_hdf5_omniglot import create_events_hdf5
import os
import torchmeta
from torchmeta.transforms import Categorical
import pdb

NUM_CLASSES = 50 # for each writing system


train_mapping = {
    'Alphabet_of_the_Magi': 0,
    'Anglo-Saxon_Futhorc': 1,
    'Arcadian': 2,
    'Armenian': 3,
    'Asomtavruli_(Georgian)': 4,
    'Balinese': 5,
    'Bengali': 6,
    'Blackfoot_(Canadian_Aboriginal_Syllabics)': 7,
    'Braille': 8,
    'Burmese_(Myanmar)': 9,
    'Cyrillic': 10,
    'Early_Aramaic': 11,
    'Futurama': 12,
    'Grantha': 13,
    'Greek': 14,
    'Gujarati': 15
    }

validation_mapping = {
    'Hebrew': 16,
    'Inuktitut_(Canadian_Aboriginal_Syllabics)': 17,
    'Japanese_(hiragana)': 18,
    'Japanese_(katakana)': 19,
    'Korean': 20,
    'Latin': 21,
    'Malay_(Jawi_-_Arabic)': 22,
    'Mkhedruli_(Georgian)': 23,
    'N_Ko': 24,
    'Ojibwe_(Canadian_Aboriginal_Syllabics)': 25,
    'Sanskrit': 26,
    'Syriac_(Estrangelo)': 27,
    'Tagalog': 28,
    'Tifinagh': 29
    }
test_mapping = {
    'Angelic': 30,
    'Atemayar_Qelisayer': 31,
    'Atlantean': 32,
    'Aurek-Besh': 33,
    'Avesta': 34,
    'Ge_ez': 35,
    'Glagolitic': 36,
    'Gurmukhi': 37,
    'Kannada': 38,
    'Keble': 39,
    'Malayalam': 40,
    'Manipuri': 41,
    'Mongolian': 42,
    'Old_Church_Slavonic_(Cyrillic)': 43,
    'Oriya': 44,
    'Sylheti': 45,
    'Syriac_(Serto)': 46,
    'Tengwar': 47,
    'Tibetan': 48,
    'ULOG': 49
    }

def sample(hdf5_file,
        key,
        chunk_size = 300):
    '''
    
    '''
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tend = dset['times'][-1] 
    start_time = dset['times'][0] #0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, chunk_size)
    if tmad.size!=0:
        tmad[:,0]-=tmad[0,0]
    
    return tmad, label

class NOmniglotDataset(NeuromorphicDataset):
    resources_url = [['https://figshare.com/ndownloader/files/31104472',None, 'dvs_background_1.rar'],
                     ['https://figshare.com/ndownloader/files/31104475', None, 'dvs_background_2.rar'],
                     ['https://figshare.com/ndownloader/files/31104481', None, 'dvs_evaluation.rar'],]
    directory = 'data/nomniglot/'#'data/nmnist/'
    resources_local = [directory+'dvs_background_1', directory+'dvs_background_2', directory+'dvs_evaluation']

    def __init__(
            self, 
            root,
            train=True,
            valid=False,
            test=False,
            transform=None,
            target_transform=None,
            label=None,
            download=True,
            chunk_size = 100):

        self.label = label # If not None, do meta learning
        
        self.n = 0

        self.download_and_create = download
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size
        
        self.train = train
        self.valid = valid
        self.test = test
        
        self.target_transform = target_transform
        
        #print("TARGET TRANSFORM", target_transform)

        super(NOmniglotDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform)
        
        
        #if label is None:
        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            try:
                if self.train:
                    self.n = f['extra'].attrs['Ntrain']
                    self.keys = f['extra']['train_keys'][()]
                    self.keys_by_label = f['extra']['train_keys_by_label'][()]
                elif self.valid or self.test:
                    self.n = f['extra'].attrs['Nvalidation']
                    self.keys = f['extra']['validation_keys'][()]
                    self.keys_by_label = f['extra']['validation_keys_by_label'][()]
                    #self.keys_by_label[:,:] -= self.keys_by_label[0,0] #normalize
                self._num_classes = len(self.keys_by_label)

            except AttributeError:
                print('Attribute not found in hdf5 file. You may be using an old hdf5 build. Delete {0} and run again'.format(root))
                raise
                
        if label is not None:
            self.n = len(self.keys_by_label[self.label])
        
#         


    def download(self):
        isexisting = super(NOmniglotDataset, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.directory, self.root)

    def __len__(self):
        return self.n
        
    def __getitem__(self, index):
        
        if self.label != None:
            #print("N",self.n)
            ind = self.keys_by_label[self.label][index%self.n]#//self.n]
            #print("THE KEY IS", key)
        #Important to open and close in getitem to enable num_workers>0
            with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
                if self.train:
                    key = f['extra']['train_keys'][ind]
                elif self.valid or self.test:
                    key = f['extra']['validation_keys'][ind]
                data, target = sample(
                        f,
                        key,
                        chunk_size = self.chunk_size)
        else:
            with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
                if self.train:
                    key = f['extra']['train_keys'][index]
                elif self.valid or self.test:
                    key = f['extra']['validation_keys'][index]
                data, target = sample(
                        f,
                        key,
                        chunk_size = self.chunk_size)
                
            f.close()
                
            if data.size==0:
                with h5py.File(self.root, 'a', libver="latest") as f:
                    if self.train:
                        del f['extra']['train_keys'][index]
                    elif self.valid or self.test:
                        del f['extra']['validation_keys'][index]
                print("REMOVED BAD DATA")
                f.close()
                i=1/0

        if self.transform is not None:
            data = self.transform(data)

        return data, self.target_transform(target.astype('int'))
              



class ClassNOmniglotDataset(torchmeta.utils.data.ClassDataset):
    def __init__(self, root = 'data/nomniglot/nomniglot.hdf5',
                 chunk_size=300,
                 meta_train=False,
                 meta_val=False,
                 meta_test=False,
                 meta_split=None,
                 transform=None,
                 target_transform=None,
                 download=False,
                 class_augmentations=None):
        
        super(ClassNOmniglotDataset, self).__init__(
                meta_train=meta_train,
                meta_val=meta_val,
                meta_test=meta_test,
                meta_split=meta_split,
                class_augmentations=class_augmentations)
        
        
        self.root=root
        self.download = download
        
        if meta_train is True:
            self.train = True
            self.valid = False
            self.test = False
        elif meta_val is True:
            self.train = False
            self.valid = True
            self.test = False
        elif meta_test is True:
            self.train = False
            self.valid = False
            self.test = True
        
        if meta_train:
            split_name = 'train'
        if meta_val:
            split_name = 'val'
        if meta_test:
            split_name = 'test'
        self.split_name = split_name
        
        self.transform = transform
        
        self.chunk_size = chunk_size        
        
        self.target_transform = target_transform
        
        self.dataset =   NOmniglotDataset(root =self.root, 
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
        
        d = NOmniglotDataset(root =self.root, 
                                     label=label,
                                     transform = self.transform, 
                                     target_transform = self.target_transform, 
                                     chunk_size = self.chunk_size)
        d.index = index
        #d.target_transform_append = lambda x: None
        return d



class NOmniglot(torchmeta.utils.data.CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False,chunk_size=300):

        if target_transform is None:
            target_tranform = Categorical(num_classes_per_task)
            
        print("NUM PER TASK", num_classes_per_task)
            
        dataset = ClassNOmniglotDataset(root,
            meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download,chunk_size=chunk_size)

        super(NOmniglot, self).__init__(dataset, 
                                           num_classes_per_task,
                                           target_transform=target_transform,
                                           dataset_transform=dataset_transform)
        



def create_class_dataset(dset, meta_split = 'train'):
    ds = []
    for n in range(dset.nclasses):
        indices = dset.keys_by_label[n]
        d = torch.utils.data.Subset(dset, indices)
        ds.append(d)
    return ds
        
def create_datasets(
        root = 'data/nomniglot/nomniglot.hdf5',
        batch_size = 72 ,
        chunk_size_train = 300,
        chunk_size_test = 300,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_valid = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_valid = None,
        target_transform_test = None):

    size = [2, 346//ds, 260//ds]

    if transform_train is None:
        transform_train = Compose([
            CropDims(low_crop=[0,0], high_crop=[346,260], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_train, size = size),
            ToTensor()])
    if transform_valid is None:
        transform_test = Compose([
            CropDims(low_crop=[0,0], high_crop=[346,260], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()])
    if transform_test is None:
        transform_test = Compose([
            CropDims(low_crop=[0,0], high_crop=[346,260], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()])
        
    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train)])
    if target_transform_valid is None:
        target_transform_valid = None
    if target_transform_test is None:
        target_transform_test = None

    train_ds = NOmniglotDataset(root,train=True,
                                 transform = transform_train, 
                                 target_transform = target_transform_train, 
                                 chunk_size = chunk_size_train*dt)
    
    valid_ds = NOmniglotDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 valid=True,
                                 test=False,
                                 chunk_size = chunk_size_test*dt)

    test_ds = NOmniglotDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 valid=False,
                                 test=True,
                                 chunk_size = chunk_size_test*dt)

    return train_ds, valid_ds , test_ds

def create_dataloader(
        root = 'data/nomniglot/nomniglot.hdf5',
        batch_size = 72 ,
        chunk_size_train = 100,
        chunk_size_test = 100,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_valid = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_valid = None,
        target_transform_test = None,
        **dl_kwargs):

    train_d, valid_d, test_d = create_datasets( #, test_d = create_datasets(
        root = root,
        batch_size = batch_size,
        chunk_size_train = chunk_size_train,
        chunk_size_test = chunk_size_test,
        ds = ds,
        dt = dt,
        transform_train = transform_train,
        transform_valid = transform_valid,
        transform_test = transform_test,
        target_transform_train = target_transform_train,
        target_transform_valid = target_transform_valid,
        target_transform_test = target_transform_test)


    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
    valid_dl = torch.utils.data.DataLoader(valid_d, shuffle=False, batch_size=batch_size, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)

    return train_dl, valid_dl, test_dl