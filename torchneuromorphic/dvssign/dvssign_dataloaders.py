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
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
from .create_hdf5_sign import create_events_hdf5
import os

NUM_CLASSES = 24 # A-Y excluding j

mapping = { 'a':0,
            'b':1,
            'c':2,
            'd':3,
            'e':4,
            'f':5,
            'g':6,
            'h':7,
            'i':8,
            'k':9,
            'l':10,
            'm':11,
            'n':12,
            'o':13,
            'p':14,
            'q':15,
            'r':16,
            's':17,
            't':18,
            'u':19,
            'v':20,
            'w':21,
            'x':22,
            'y':23}

class DVSSignDataset(NeuromorphicDataset):
    resources_url = [['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAB0jgWqXDn3sZB_YXEjZLv4a/Yin%20Bi%20-%20a.zip?dl=0',None, 'Yin Bi - a.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAC-671H-Z7XTAQcT7GJXFsGa/Yin%20Bi%20-%20b.zip?dl=0', None, 'Yin Bi - b.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADa8hkEbpgnNbBtmRIuAw3ha/Yin%20Bi%20-%20c.zip?dl=0', None, 'Yin Bi - c.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACUrdhDl_tYnNkb8OpAJ5k4a/Yin%20Bi%20-%20d.zip?dl=0', None, 'Yin Bi - d.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABmXnWZ2hI2dPQYn3FOClnba/Yin%20Bi%20-%20e.zip?dl=0', None, 'Yin Bi - e.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAzopIcHXTmjPuYomjAiPtfa/Yin%20Bi%20-%20f.zip?dl=0', None, 'Yin Bi - f.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAB0PMA-VwZMM1PpJXg6q4efa/Yin%20Bi%20-%20g.zip?dl=0', None, 'Yin Bi - g.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAD9-W_I4n3lTSCMgFGgJ5Dra/Yin%20Bi%20-%20h.zip?dl=0', None, 'Yin Bi - h.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADCm3yMnJcwGK70bYk-ycF0a/Yin%20Bi%20-%20i.zip?dl=0', None, 'Yin Bi - i.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAChe4QTo2DduVOVuT5hN9fxa/Yin%20Bi%20-%20k.zip?dl=0', None, 'Yin Bi - k.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAeUazC7PHK85V6wkEub1iMa/Yin%20Bi%20-%20l.zip?dl=0', None, 'Yin Bi - l.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADrzW_ts5UxlulXaItiNuCSa/Yin%20Bi%20-%20m.zip?dl=0', None, 'Yin Bi - m.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABlYTbweHA22nA6PFujaBKFa/Yin%20Bi%20-%20n.zip?dl=0', None, 'Yin Bi - n.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABzMSVYuZ0hb5FsUoHB53xBa/Yin%20Bi%20-%20o.zip?dl=0', None, 'Yin Bi - o.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACOJ3z96KaLAMxgNLP1eOwga/Yin%20Bi%20-%20p.zip?dl=0', None, 'Yin Bi - p.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAC6B6UajjSuf6aYYOfOY3I7a/Yin%20Bi%20-%20q.zip?dl=0', None, 'Yin Bi - q.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAC-Gq6qzv0yiAnvozEozqoaa/Yin%20Bi%20-%20r.zip?dl=0', None, 'Yin Bi - r.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABDbuK8B0Mferpf0x3xbDJQa/Yin%20Bi%20-%20s.zip?dl=0', None, 'Yin Bi - s.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAEgawjqMHuY_TvqCYNC-uIa/Yin%20Bi%20-%20t.zip?dl=0', None, 'Yin Bi - t.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABTIlScbaSqFIahMy_NRNUna/Yin%20Bi%20-%20u.zip?dl=0', None, 'Yin Bi - u.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAp8JvzLPfhGlG5jL943W_pa/Yin%20Bi%20-%20v.zip?dl=0', None, 'Yin Bi - v.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAA4qvsatKEDeoykc2I4a6FRa/Yin%20Bi%20-%20w.zip?dl=0', None, 'Yin Bi - w.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABfnDu7rtZve1w9VVQQwuFia/Yin%20Bi%20-%20x.zip?dl=0', None, 'Yin Bi - x.zip'],
                     ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADKEQBAXFQ9P0GoGHTY4ig8a/Yin%20Bi%20-%20y.zip?dl=0', None, 'Yin Bi - y.zip']
                    ]
    directory = 'data/ASL-DVS/'#'data/nmnist/'
    resources_local = [directory+'a', directory+'b', directory+'c', directory+'d',directory+'e',directory+'f',directory+'g',directory+'h',directory+'i',
                      directory+'k',directory+'l',directory+'m',directory+'n',directory+'o',directory+'p',directory+'q',directory+'r',directory+'s',
                      directory+'t',directory+'u',directory+'v',directory+'w',directory+'x',directory+'y',directory+'z']

    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            chunk_size = 100):

        self.n = 0
        self.nclasses = self.num_classes = 10
        self.download_and_create = download_and_create
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size

        super(DVSSignDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform )
        
        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            try:
                if train:
                    self.n = f['extra'].attrs['Ntrain']
                    self.keys = f['extra']['train_keys'][()]
                    self.keys_by_label = f['extra']['train_keys_by_label'][()]
                else:
                    self.n = f['extra'].attrs['Ntest']
                    self.keys = f['extra']['test_keys'][()]
                    self.keys_by_label = f['extra']['test_keys_by_label'][()]
                    self.keys_by_label[:,:] -= self.keys_by_label[0,0] #normalize
            except AttributeError:
                print('Attribute not found in hdf5 file. You may be using an old hdf5 build. Delete {0} and run again'.format(root))
                raise


    def download(self):
        isexisting = super(DVSSignDataset, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.directory, self.root)


    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if self.train:
                key = f['extra']['train_keys'][key]
            else:
                key = f['extra']['test_keys'][key]
            data, target = sample(
                    f,
                    key,
                    T = self.chunk_size)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

def sample(hdf5_file,
        key,
        T = 300):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tend = dset['times'][-1] 
    start_time = 0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad, label

def create_datasets(
        root = 'data/dvssign/dvssign.hdf5',
        batch_size = 72 ,
        chunk_size_train = 300,
        chunk_size_test = 300,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None):

    size = [2, 240//ds, 180//ds]

    if transform_train is None:
        transform_train = Compose([
            CropDims(low_crop=[0,0], high_crop=[240,180], dims=[2,3]),
            Downsample(factor=[dt,1,1,1]),
            ToCountFrame(T = chunk_size_train, size = size),
            ToTensor()])
    if transform_test is None:
        transform_test = Compose([
            CropDims(low_crop=[0,0], high_crop=[240,180], dims=[2,3]),
            Downsample(factor=[dt,1,1,1]),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()])
    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(NUM_CLASSES)])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(NUM_CLASSES)])

    train_ds = DVSSignDataset(root,train=True,
                                 transform = transform_train, 
                                 target_transform = target_transform_train, 
                                 chunk_size = chunk_size_train)

    test_ds = DVSSignDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 chunk_size = chunk_size_test)

    return train_ds, test_ds

def create_dataloader(
        root = 'data/dvssign/dvssign.hdf5',
        batch_size = 72 ,
        chunk_size_train = 100,
        chunk_size_test = 100,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
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
        target_transform_test = target_transform_test)


    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl



