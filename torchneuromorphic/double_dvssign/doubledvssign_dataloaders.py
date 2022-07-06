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

import random

# Data taken from https://github.com/PIX2NVS/NVS2Graph


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

double_digit_letters = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


# NOTE: These splits only use the first 10 classes. Will need to incorporate the double digit classes for more possible meta tasks
#splits = {}
# splits['train'] = ['00', '01', '04', '05', '06', '08', '09', '11', '12', '13', '14', '15', '16', '18', '19', '20', '21', '23', '24', '26', '28', '29', '30', '31', '33', '35', '37', '38', '41', '42', '43', '44', '45', '50', '51', '53', '54', '56', '59', '60', '62', '63', '65', '69', '70', '72', '74', '75', '76', '77', '79', '81', '82', '84', '85', '87', '88', '89', '90', '91', '94', '95', '97', '98']
# splits['val'] = ['03', '07', '10', '22', '27', '34', '39', '40', '48', '52', '58', '61', '64', '71', '93', '99']
# splits['test'] = ['02', '17', '25', '32', '36', '46', '47', '49', '55', '57', '66', '67', '68', '73', '78', '80', '83', '86', '92', '96']

# These splits use all of the letters i.e. all 24 classes for a possible 576 tasks. Actually, why don't I make a way to randomize the splits to make it so that all 576 possible tasks really are used and not a subset. 

def split_generator():
    splits = {}
    
    splits['train'] = []
    splits['val'] = []
    splits['test'] = []
    
    label_combos = []
    
    # create list of possible digit label combos
    for i in range(24):
        for j in range(24):
            combo = str(i) + "." + str(j)
            
            label_combos.append(combo)
            
    print(len(label_combos))
            
    # randomly shuffle the combos
    random.shuffle(label_combos)
    
    # generate a train, val, and test dataset from the possible class configurations
    for i in range(len(label_combos)):
        if i < 369:
            # put in training
            splits['train'].append(label_combos[i])
            
        elif i < (369+92):
            # put in val
            splits['val'].append(label_combos[i])
        else:
            # put in test
            splits['test'].append(label_combos[i])
            
    return splits
    

class DoubleDVSSignClassDataset(NeuromorphicDataset):
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
        
        lu = lu.split(".")
        self.labels_left = int(lu[0])
        self.labels_right = int(lu[1])
#         self.labels_left = lu // 10
#         self.labels_right = lu % 10 

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

        data_l, label_l =  self.data_orig[key_l] # This is a hack because for some reason it's trying to find keys outside of where it's supposed to (all train and test instead of just train for some reason)
        data_r, label_r =  self.data_orig[key_r]

        size_x, size_y = data_r.shape[2:4]
        data = torch.zeros(data_r.shape[:2]+(size_x*2,size_y))
        data[:, :, :size_x, :] = data_l
        data[:, :, size_x:, :] = data_r
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
    def __init__(self, root = 'data/ASL-DVS/dvssign.hdf5', chunk_size=100, meta_train=False, meta_val=False, meta_test=False, meta_split='', transform=None, target_transform=None, download=False, class_augmentations=None):
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
        
        if meta_split: 
            # load the splits from file i.e. meta_split is a json file that contains the split that will be saved with the model and config file in logs
            # importing the module
            import json

            # Opening JSON file
            with open(meta_split) as json_file:
                splits = json.load(json_file)
                
                # print for now just to make sure that it works
                # Print the type of data variable
                print("Type:", type(splits))

                # Print the data of dictionary
                print(f"\n{split_name}:", splits[split_name])
        else:
            splits = split_generator()
            
            import json
            
            # save the splits for future use
            json = json.dumps(splits)
            f = open("doubledvssign_splits_full.json","w")
            f.write(json)
            f.close()


        self._labels = [s for s in splits[split_name]]
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



