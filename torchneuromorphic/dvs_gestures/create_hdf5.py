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

def create_events_hdf5(directory, hdf5_filename):
    fns_train = gather_aedat(directory,1,24)
    fns_test = gather_aedat (directory,24,30)
    test_keys = []
    train_keys = []

    assert len(fns_train)==98

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()

        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        for file_d in tqdm(fns_train+fns_test):
            istrain = file_d in fns_train
            data, labels_starttime = aedat_to_events(file_d)
            tms = data[:,0]
            ads = data[:,1:]
            lbls = labels_starttime[:,0]
            start_tms = labels_starttime[:,1]
            end_tms = labels_starttime[:,2]
            out = []

            for i, v in enumerate(lbls):
                if istrain: 
                    train_keys.append(key)
                else:
                    test_keys.append(key)
                s_ = get_slice(tms, ads, start_tms[i], end_tms[i])
                times = s_[0]
                addrs = s_[1]
                subj, light = file_d.split('/')[-1].split('.')[0].split('_')[:2]
                metas.append({'key':str(key), 'subject':subj,'light condition':light, 'training sample':istrain}) 
                subgrp = data_grp.create_group(str(key))
                tm_dset = subgrp.create_dataset('times' , data=times, dtype=np.uint32)
                ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint8)
                lbl_dset= subgrp.create_dataset('labels', data=lbls[i]-1, dtype=np.uint8)
                subgrp.attrs['meta_info']= str(metas[-1])
                assert lbls[i]-1 in range(11)
                key += 1
        extra_grp.create_dataset('train_keys', data=train_keys)
        extra_grp.create_dataset('test_keys', data=test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
            
def gather_aedat(directory, start_id, end_id, filename_prefix = 'user'):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found, looked at: {}".format(directory))
    import glob
    fns = []
    for i in range(start_id,end_id):
        search_mask = directory+'/'+filename_prefix+"{0:02d}".format(i)+'*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out)>0:
            fns+=glob_out
    return fns


