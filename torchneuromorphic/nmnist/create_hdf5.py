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

def nmnist_load_events_from_bin(file_path, max_duration=None):
    timestamps, xaddr, yaddr, pol = load_ATIS_bin(file_path)
    return np.column_stack([
        np.array(timestamps, dtype=np.uint32),
        np.array(pol, dtype=np.uint8),
        np.array(xaddr, dtype=np.uint16),
        np.array(yaddr, dtype=np.uint16)])

def nmnist_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("N-MNIST Dataset not found, looked at: {}".format(dataset_path))

    train_files = []
    test_files = []
    for digit in range(10):
        digit_train = glob.glob(os.path.join(dataset_path, 'Train/{}/*.bin'.format(digit)))
        digit_test = glob.glob(os.path.join(dataset_path, 'Test/{}/*.bin'.format(digit)))
        train_files.append(digit_train)
        test_files.append(digit_test)

    # We need the same number of train and test samples for each digit, let's compute the minimum
    max_n_train = min(map(lambda l: len(l), train_files))
    max_n_test = min(map(lambda l: len(l), test_files))
    n_train = max_n_train # we could take max_n_train, but my memory on the shared drive is full
    n_test = max_n_test # we test on the whole test set - lets only take 100*10 samples
    assert((n_train <= max_n_train) and (n_test <= max_n_test)), 'Requested more samples than present in dataset'

    print("N-MNIST: {} train samples and {} test samples per digit (max: {} train and {} test)".format(n_train, n_test, max_n_train, max_n_test))
    # Crop extra samples of each digits
    train_files = map(lambda l: l[:n_train], train_files)
    test_files = map(lambda l: l[:n_test], test_files)

    return list(train_files), list(test_files)

def create_events_hdf5(directory, hdf5_filename):
    fns_train, fns_test = nmnist_get_file_names(directory)
    fns_train = [val for sublist in fns_train for val in sublist]
    fns_test = [val for sublist in fns_test for val in sublist]
    test_keys = []
    train_keys = []
    label_list = [[] for i in range(10)]

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        for file_d in tqdm(fns_train+fns_test):
            istrain = file_d in fns_train
            data = nmnist_load_events_from_bin(file_d)
            times = data[:,0]
            addrs = data[:,1:]
            label = int(file_d.split('/')[-2])
            out = []

            if istrain: 
                train_keys.append(key)
            else:
                test_keys.append(key)
            metas.append({'key':str(key), 'training sample':istrain}) 
            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset('times' , data=times, dtype=np.uint32)
            ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint8)
            lbl_dset= subgrp.create_dataset('labels', data=label, dtype=np.uint8)
            subgrp.attrs['meta_info']= str(metas[-1])
            assert label in range(10)
            label_list[label].append(key)
            key += 1
        extra_grp.create_dataset('train_keys', data=train_keys)
        extra_grp.create_dataset('keys_by_label', data=label_list)
        extra_grp.create_dataset('test_keys', data=test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
