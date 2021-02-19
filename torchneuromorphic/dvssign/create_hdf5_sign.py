#!/bin/python
#-----------------------------------------------------------------------------
# File Name : create_hdf5.py
# Author: Emre Neftci
#
# Creation Date : Tue Nov  5 13:15:54 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
from tqdm import tqdm
import scipy.misc
import h5py
import glob
import torch.utils.data
from ..events_timeslices import *
from ..utils import *
import os

from collections import namedtuple, defaultdict
import torch
import torch.utils.data
from ..utils import load_ATIS_bin, load_jaer

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

def sign_load_events_from_mat(file_path, max_duration=None):
    timestamps, xaddr, yaddr, pol = load_mat(file_path)#load_ATIS_bin(file_path)
    return np.column_stack([
        np.array(timestamps, dtype=np.uint32),
        np.array(pol, dtype=np.uint8),
        np.array(xaddr, dtype=np.uint16), 
        np.array(yaddr, dtype=np.uint16)])

def sign_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("DVSSign Dataset not found, looked at: {}".format(dataset_path))

    sign_dict_train = {}
    
    sign_dict_test = {}
    
    # need to create train test split
    # simple 80/20 split based on number of gestures per class on a fraction of the dataset
    num_train = 400 #3360
    num_test = 100 #840
    os.chdir(dataset_path+'/a')
    for key in mapping.keys():
        os.chdir('../'+key)
        curr_dir = os.getcwd()
        # each class has 4200 samples.
        # if doing 80/20 split, 3360 train 840 test
        num_samples = 1
        for file in glob.glob("*.mat"):
            label = file.split('_')[0]

            if num_samples < num_train:
                if label in sign_dict_train.keys():
                    sign_dict_train[label].append(curr_dir+'/'+file)
                else:
                    sign_dict_train[label] = []
                    sign_dict_train[label].append(curr_dir+'/'+file)
            if num_samples < num_train+num_test:
                if label in sign_dict_test.keys():
                    sign_dict_test[label].append(curr_dir+'/'+file)
                else:
                    sign_dict_test[label] = []
                    sign_dict_test[label].append(curr_dir+'/'+file)
            num_samples += 1

    # We need the same number of train and test samples for each digit, let's compute the minimum
    #max_n_train = min(map(lambda l: len(l), train_files))
    #max_n_test = min(map(lambda l: len(l), test_files))
    #n_train = max_n_train # we could take max_n_train, but my memory on the shared drive is full
    #n_test = max_n_test # we test on the whole test set - lets only take 100*10 samples
    #assert((n_train <= max_n_train) and (n_test <= max_n_test)), 'Requested more samples than present in dataset'

    print("DVSSign: {} train samples and {} test samples per digit (max: {} train and {} test)".format(num_train*NUM_CLASSES, num_test*NUM_CLASSES, num_train*NUM_CLASSES, num_test*NUM_CLASSES))
    # Crop extra samples of each digits
    #train_files = map(lambda l: l[:n_train], train_files)
    #test_files = map(lambda l: l[:n_test], test_files)

    return sign_dict_train, sign_dict_test

def create_events_hdf5(directory, hdf5_filename):
    fns_train, fns_test = sign_get_file_names(directory)
    #fns_train = [val for sublist in fns_train for val in sublist]
    #fns_test = [val for sublist in fns_test for val in sublist]
    test_keys = []
    train_keys = []
    train_label_list = [[] for i in range(NUM_CLASSES)]
    test_label_list = [[] for i in range(NUM_CLASSES)]

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        #key = 0
        num = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        for key in tqdm(fns_train.keys()):
            for file_d in fns_train.values():
                for i in range(len(file_d)):
                    data = sign_load_events_from_mat(file_d[i])
                    times = data[:,0]
                    addrs = data[:,1:]
                    label = mapping[key] #int(file_d.split('/')[-2]) 

                    train_keys.append(num)

                    train_label_list[mapping[key]].append(num)

                    metas.append({'key':str(num), 'training sample':True}) 
                    subgrp = data_grp.create_group(str(num))
                    tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
                    ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
                    lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
                    subgrp.attrs['meta_info']= str(metas[-1])
                    num += 1
                
            for file_d in fns_test.values():
                for i in range(len(file_d)):
                    data = sign_load_events_from_mat(file_d[i])
                    times = data[:,0]
                    addrs = data[:,1:]
                    label = mapping[key]

                    test_keys.append(num)

                    test_label_list[mapping[key]].append(num)

                    metas.append({'key':str(num), 'testing sample':True}) 
                    subgrp = data_grp.create_group(str(num))
                    tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
                    ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
                    lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
                    subgrp.attrs['meta_info']= str(metas[-1])
                    num += 1
                
        extra_grp.create_dataset('train_keys', data = train_keys)
        extra_grp.create_dataset('train_keys_by_label', data = train_label_list)
        extra_grp.create_dataset('test_keys_by_label', data = test_label_list)
        extra_grp.create_dataset('test_keys', data = test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
