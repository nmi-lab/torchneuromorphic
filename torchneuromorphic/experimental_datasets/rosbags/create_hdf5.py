#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Jacques Kaiser
#
# Creation Date : Wed 13 May 2020
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------

import numpy as np
import h5py
from ..events_timeslices import *
from ..utils import *
import torch.utils.data
import glob
import os
from tqdm import tqdm

def gather_rosbags_class_folders(directory):
    # expected folder structure:
    # directory
    # ├── test_list.csv
    # ├── label1
    # │__ ├── bag1.bag
    # │__ ├── bag2.bag
    # ├── label2
    # │__ ├── bag1.bag
    # │__ ├── bag2.bag
    all_bags = glob.glob(os.path.join(directory, '**/*.bag'))
    test_list = np.loadtxt(os.path.join(directory, 'test_list.csv'), delimiter=',',dtype=str)

    test_bags = [ bag for bag in all_bags if np.any([ bag.endswith(test) for test in test_list]) ]
    train_bags = [ bag for bag in all_bags if bag not in test_bags ]
    all_classes = np.unique([os.path.basename(os.path.dirname(b)) for b in all_bags])
    return train_bags, test_bags, all_classes


def create_events_hdf5(directory, hdf5_filename, gather_rosbags=gather_rosbags_class_folders):
    train_bags, test_bags, all_classes = gather_rosbags(directory)
    label_mapping = { label: i
                      for i,label in enumerate(all_classes)
    }
    label_mapping_inv = { i: label for label,i in label_mapping.items() }
    label_order = [label_mapping_inv[i] for i in range(len(all_classes)) ]

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        train_keys = []
        test_keys = []
        for file_d in train_bags + test_bags:
            filename=os.path.basename(file_d)
            class_name=os.path.basename(os.path.dirname(file_d))
            label=label_mapping[class_name]
            events=rosbag_to_events(file_d)
            addrs = np.array([events['pol'], events['x'], events['y']]).T
            istrain = file_d in train_bags
            if istrain:
                train_keys.append(key)
            else:
                test_keys.append(key)

            metas.append({'key':str(key), 'training sample':istrain})

            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset('times' , data=events['ts'], dtype=np.uint32)
            ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint8)
            lbl_dset= subgrp.create_dataset('labels', data=label, dtype=np.uint8)
            subgrp.attrs['meta_info']= str(metas[-1])
            key+=1
        extra_grp.create_dataset('train_keys', data=train_keys)
        extra_grp.create_dataset('test_keys', data=test_keys)
        extra_grp.create_dataset('label_order', data=np.array(label_order, dtype='S10'))
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
