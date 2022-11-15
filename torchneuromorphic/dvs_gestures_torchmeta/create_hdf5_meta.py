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
from torchneuromorphic.events_timeslices import *
from torchneuromorphic.utils import *
import os

import pdb

# there are 29 users
# for a meta version, each user and their gesture class will be a task
# therefore, 29*11 = 319 tasks
# 23 train, 6 test like the original dataset, 23*11=253 train tasks, 6*11=66 test tasks

data_path = '../dvs_gestures/data/DvsGesture/'


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

TEST_OFFSET = 23
NUM_CLASSES = 29

def create_events_hdf5(directory='../dvs_gestures/data/DvsGesture/', hdf5_filename='dvs_gesture_meta.hdf5'):
    fns_train = gather_aedat(directory,1,24)
    fns_test = gather_aedat (directory,24,30)
    
    test_keys = []
    train_keys = []
    train_label_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    test_label_list = [[],[],[],[],[],[]]

    assert len(fns_train)==98

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        train_keys_by_label = f.create_group('train_keys_by_label')
        test_keys_by_label = f.create_group('test_keys_by_label')
        for file_d in tqdm(fns_train+fns_test):
            #pdb.set_trace()
            istrain = file_d in fns_train
            data, labels_starttime = aedat_to_events(file_d)
            tms = data[:,0]
            ads = data[:,1:]
            lbls = labels_starttime[:,0]
            start_tms = labels_starttime[:,1]
            end_tms = labels_starttime[:,2]
            out = []

            for i, v in enumerate(lbls):
                #pdb.set_trace()
                if istrain: 
                    user = train_mapping[file_d.split('/')[-1].split('_')[0]]
                    train_keys.append(key)
                    train_label_list[user].append(key)
                else:
                    user = test_mapping[file_d.split('/')[-1].split('_')[0]]
                    test_keys.append(key)
                    test_label_list[user-TEST_OFFSET].append(key)
                    
                s_ = get_slice(tms, ads, start_tms[i], end_tms[i])
                times = s_[0]
                addrs = s_[1]
                subj, light = file_d.split('/')[-1].split('.')[0].split('_')[:2]
                metas.append({'key':str(key), 'gesture':lbls[i]-1,'light condition':light, 'training sample':istrain}) 
                subgrp = data_grp.create_group(str(key))
                tm_dset = subgrp.create_dataset('times' , data=times, dtype=np.uint32)
                ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint8)
                lbl_dset= subgrp.create_dataset('labels', data=user,dtype=np.uint8)#lbls[i]-1, dtype=np.uint8)
                subgrp.attrs['meta_info']= str(metas[-1])
                assert lbls[i]-1 in range(11)
                key += 1

        pdb.set_trace()
        extra_grp.create_dataset('train_keys', data = train_keys)
        
        for i in range(len(train_label_list)):
            train_keys_by_label.create_dataset(str(i), data=train_label_list[i])
        for i in range(len(test_label_list)):
            test_keys_by_label.create_dataset(str(i) , data=test_label_list[i])
        #extra_grp.create_dataset('train_keys_by_label', data = train_label_list)
        #extra_grp.create_dataset('test_keys_by_label', data = test_label_list)
        
        extra_grp.create_dataset('test_keys', data = test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)# + len(validation_keys)
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



if __name__=='__main__':
    create_events_hdf5()

