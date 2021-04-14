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
from ..dvssign.dvssign_dataloaders import DVSSignDataset, sample, create_datasets
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
#from .create_hdf5_sign import create_events_hdf5
import os
import torchmeta
from torchmeta.transforms import Categorical


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

# splits = {}
# splits['train'] = ['aa', 'ab', 'ae', 'af', 'ag', 'ai', 'ak', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bi', 'bk', 'ca', 'cb', 'cd', 'ce', 'cg', 'ci', 'ck', 'da', 'db', 'dd', 'df', 'dh', 'di', 'eb', 'ec', 'ed', 'ee', 'ef', 'fa', 'fb', 'fd', 'fe', 'fg', 'fk', 'ga', 'gc', 'gd', 'gf', 'gk', 'ha', 'hc', 'he', 'hf', 'hg', 'hh', 'hk', 'ib', 'ic', 'ie', 'if', 'ih', 'ii', 'ik', 'ka', 'kb', 'ke', 'kf', 'kh', 'ki']
# splits['val'] = ['ad', 'ah', 'ba', 'cc', 'ch', 'de', 'dk', 'ea', 'ei', 'fc', 'fi', 'gb', 'ge', 'hb', 'kd', 'kk']
# splits['test'] = ['ac', 'bh', 'cf', 'dc', 'dg', 'eg', 'eh', 'ek', 'ff', 'fh', 'gg', 'gh', 'gi', 'hd', 'hi', 'ia', 'id', 'ig', 'kc', 'kg']

splits = {}
splits['train'] = ['00', '01', '04', '05', '06', '08', '09', '11', '12', '13', '14', '15', '16', '18', '19', '20', '21', '23', '24', '26', '28', '29', '30', '31', '33', '35', '37', '38', '41', '42', '43', '44', '45', '50', '51', '53', '54', '56', '59', '60', '62', '63', '65', '69', '70', '72', '74', '75', '76', '77', '79', '81', '82', '84', '85', '87', '88', '89', '90', '91', '94', '95', '97', '98']
splits['val'] = ['03', '07', '10', '22', '27', '34', '39', '40', '48', '52', '58', '61', '64', '71', '93', '99']
splits['test'] = ['02', '17', '25', '32', '36', '46', '47', '49', '55', '57', '66', '67', '68', '73', '78', '80', '83', '86', '92', '96']

class DoubleDVSSignClassDataset(NeuromorphicDataset):
#     resources_url = [['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAB0jgWqXDn3sZB_YXEjZLv4a/Yin%20Bi%20-%20a.zip?dl=0',None, 'Yin Bi - a.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAC-671H-Z7XTAQcT7GJXFsGa/Yin%20Bi%20-%20b.zip?dl=0', None, 'Yin Bi - b.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADa8hkEbpgnNbBtmRIuAw3ha/Yin%20Bi%20-%20c.zip?dl=0', None, 'Yin Bi - c.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACUrdhDl_tYnNkb8OpAJ5k4a/Yin%20Bi%20-%20d.zip?dl=0', None, 'Yin Bi - d.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABmXnWZ2hI2dPQYn3FOClnba/Yin%20Bi%20-%20e.zip?dl=0', None, 'Yin Bi - e.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAzopIcHXTmjPuYomjAiPtfa/Yin%20Bi%20-%20f.zip?dl=0', None, 'Yin Bi - f.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAB0PMA-VwZMM1PpJXg6q4efa/Yin%20Bi%20-%20g.zip?dl=0', None, 'Yin Bi - g.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAD9-W_I4n3lTSCMgFGgJ5Dra/Yin%20Bi%20-%20h.zip?dl=0', None, 'Yin Bi - h.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADCm3yMnJcwGK70bYk-ycF0a/Yin%20Bi%20-%20i.zip?dl=0', None, 'Yin Bi - i.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAChe4QTo2DduVOVuT5hN9fxa/Yin%20Bi%20-%20k.zip?dl=0', None, 'Yin Bi - k.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAeUazC7PHK85V6wkEub1iMa/Yin%20Bi%20-%20l.zip?dl=0', None, 'Yin Bi - l.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADrzW_ts5UxlulXaItiNuCSa/Yin%20Bi%20-%20m.zip?dl=0', None, 'Yin Bi - m.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABlYTbweHA22nA6PFujaBKFa/Yin%20Bi%20-%20n.zip?dl=0', None, 'Yin Bi - n.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABzMSVYuZ0hb5FsUoHB53xBa/Yin%20Bi%20-%20o.zip?dl=0', None, 'Yin Bi - o.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACOJ3z96KaLAMxgNLP1eOwga/Yin%20Bi%20-%20p.zip?dl=0', None, 'Yin Bi - p.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAC6B6UajjSuf6aYYOfOY3I7a/Yin%20Bi%20-%20q.zip?dl=0', None, 'Yin Bi - q.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAC-Gq6qzv0yiAnvozEozqoaa/Yin%20Bi%20-%20r.zip?dl=0', None, 'Yin Bi - r.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABDbuK8B0Mferpf0x3xbDJQa/Yin%20Bi%20-%20s.zip?dl=0', None, 'Yin Bi - s.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAEgawjqMHuY_TvqCYNC-uIa/Yin%20Bi%20-%20t.zip?dl=0', None, 'Yin Bi - t.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABTIlScbaSqFIahMy_NRNUna/Yin%20Bi%20-%20u.zip?dl=0', None, 'Yin Bi - u.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAAp8JvzLPfhGlG5jL943W_pa/Yin%20Bi%20-%20v.zip?dl=0', None, 'Yin Bi - v.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AAA4qvsatKEDeoykc2I4a6FRa/Yin%20Bi%20-%20w.zip?dl=0', None, 'Yin Bi - w.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AABfnDu7rtZve1w9VVQQwuFia/Yin%20Bi%20-%20x.zip?dl=0', None, 'Yin Bi - x.zip'],
#                      ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AADKEQBAXFQ9P0GoGHTY4ig8a/Yin%20Bi%20-%20y.zip?dl=0', None, 'Yin Bi - y.zip']
#                     ]
#     directory = 'data/ASL-DVS/'#'data/nmnist/'
#     resources_local = [directory+'a', directory+'b', directory+'c', directory+'d',directory+'e',directory+'f',directory+'g',directory+'h',directory+'i',
#                       directory+'k',directory+'l',directory+'m',directory+'n',directory+'o',directory+'p',directory+'q',directory+'r',directory+'s',
#                       directory+'t',directory+'u',directory+'v',directory+'w',directory+'x',directory+'y',directory+'z']

    def __init__(
            self, 
            root : str,
            train : bool = False,
            transform : object = None ,
            target_transform=None,
            download_and_create=True,
            chunk_size = 100,
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
        #print("labels_left", ll)
        lr = self.labels_right

        super(DoubleDVSSignClassDataset, self).__init__(
                root=None,
                transform=transform,
                target_transform=target_transform )
        
        self.data_orig = DVSSignDataset( root,
                      train=True,
                      transform=transform,
                      target_transform=target_transform,
                      download_and_create=download_and_create,                    
                      chunk_size = chunk_size)
        
        self.nl = len(self.data_orig.keys_by_label[ll])
        #print(f"nl{self.nl}")
        self.nr = len(self.data_orig.keys_by_label[lr])
        #print(f"nr{self.nr}")
        self.n = self.nl * self.nr


    def __len__(self):
        return 1000 #not sure why it's 1000, that's from DoubleNMNIST #self.n
        
    def __getitem__(self, key):
        ll = self.labels_left
        lr = self.labels_right
        key_l = self.data_orig.keys_by_label[ll][ key // self.nl] # // nl 
        key_r = self.data_orig.keys_by_label[lr][ key % self.nl]  # % nl
        
        #print(self.train)
        
#         if self.train:
#             if key_l > 3989:
#                 key_l = key_l - 1000 # this is a hack, I'm not sure why it's trying to find keys outside of the scope yet
#             if key_r > 3989:
#                 key_r = key_r - 1000

        data_l, label_l =  self.data_orig[key_l] # This is a hack because for some reason it's trying to find keys outside of where it's supposed to (all train and test instead of just train for some reason)
        data_r, label_r =  self.data_orig[key_r]

        size_x, size_y = data_r.shape[2:4]
        data = torch.zeros(data_r.shape[:2]+(size_x*2,size_y*2))
        np.random.seed(key%1313)
        r1 = np.random.randint(0,size_y)
        r2 = np.random.randint(0,size_y)
        data[:, :, :size_x, r1:r1+size_y] = data_l
        data[:, :, size_x:, r2:r2+size_y] = data_r
        target = self.label_u
        return data, self.target_transform(target)

def create_datasets(
        root = 'data/ASL-DVS/dvssign.hdf5',
        batch_size = 72 ,
        chunk_size_train = 100,
        chunk_size_test = 100,
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

    size = [2, 240//ds, 180//ds]

    if transform_train is None:
        transform_train = Compose([
            CropDims(low_crop=[0,0], high_crop=[240,180], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_train, size = size),
            ToTensor()])
    if transform_test is None:
        transform_test = Compose([
            CropDims(low_crop=[0,0], high_crop=[240,180], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()])
    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(NUM_CLASSES)])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(NUM_CLASSES)])
        
        
    labels_u = np.random.choice(classes_meta, nclasses,replace=False) #100 (10*10) here becuase we have two pairs of gestures between A and Y but not J/. only using first 10 letters
    

    train_ds = DoubleDVSSignDataset(root,train=True,
                                 transform = transform_train, 
                                 target_transform = target_transform_train, 
                                 chunk_size = chunk_size_train,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_class,
                                 labels_u = labels_u)

    test_ds = DoubleDVSSignDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 chunk_size = chunk_size_test,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_test,
                                 labels_u = labels_u)
    
    

    return train_ds, test_ds

class ClassDVSSignDataset(torchmeta.utils.data.ClassDataset):
    def __init__(self, root = 'data/ASL-DVS/dvssign.hdf5', chunk_size=100, meta_train=False, meta_val=False, meta_test=False, meta_split=None, transform=None, target_transform=None, download=False, class_augmentations=None):
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


        super(ClassDVSSignDataset, self).__init__(
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
        d = DoubleDVSSignClassDataset(root =self.root, train= self.meta_train, label_u = label, transform = self.transform, target_transform = None, chunk_size = self.chunk_size)
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

class DoubleDVSSign(torchmeta.utils.data.CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False,chunk_size=100):

        target_transform = Categorical(num_classes_per_task)
        dataset = ClassDVSSignDataset(root,
            meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download,chunk_size=100)

        super(DoubleDVSSign, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform,
            dataset_transform=dataset_transform)



