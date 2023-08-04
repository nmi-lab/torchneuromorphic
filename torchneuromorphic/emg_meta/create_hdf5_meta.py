#!/bin/python
#-----------------------------------------------------------------------------
# File Name : create_hdf5.py
# Author: Kenneth Stewart
#
# Copyright : (c) UC Regents, Kenneth Stewart
# Licence : Apache v2
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

import importlib
from training_utils.params import Params
from training_utils.import_helpers import get_input_names, filter_dict, filter_kwargs, import_module_from_string

# for emg, the gesture and the day recorded are a task
# precompute the spikes, save to hdf5 that should work right???
# then making meta dataloader should be easy
# try with CS first, have most data for CS and it will prove if it can work
# only drawback with this method, like dvsgesture is the split is set, no cross validation with splitting easily


data_path = '/Users/k_stewart/chouti/emg_processing/CS_new/'
param_file = '/Users/k_stewart/chouti/emg_processing/Decolle/decolle/scripts/CS_new_slayer_94acc/params.yml'


train_mapping = {
    '01_03_2023': 0,
    '07_03_2023': 1,
    '04_04_2023': 2,
    '28_03_2023': 3,
    '21_02_2023': 4,
}

test_mapping = {
    '18_04_2023': 5,
    '28_03_2023_2': 6,
    '28_03_2023_3': 7,
    '14_03_2023': 8,
    '21_03_2023': 9,
}

TEST_OFFSET = 5
NUM_CLASSES = 10 # num days really, num classes/tasks total is technically 100 like double nmnist

def create_events_hdf5(directory='/Users/k_stewart/chouti/emg_processing/CS_new/', hdf5_filename='emg_meta.hdf5'):
    train_days = list(train_mapping.keys())
    test_days = list(test_mapping.keys())
    
    gen_train_0 = create_loader(data_path, train_days[0], param_file)
    gen_train_1 = create_loader(data_path, train_days[1], param_file)
    gen_train_2 = create_loader(data_path, train_days[2], param_file)
    gen_train_3 = create_loader(data_path, train_days[3], param_file)
    gen_train_4 = create_loader(data_path, train_days[4], param_file)
    
    gen_test_0 = create_loader(data_path, test_days[0], param_file)
    gen_test_1 = create_loader(data_path, test_days[1], param_file)
    gen_test_2 = create_loader(data_path, test_days[2], param_file)
    gen_test_3 = create_loader(data_path, test_days[3], param_file)
    gen_test_4 = create_loader(data_path, test_days[4], param_file)
    
    test_keys = []
    train_keys = []
    train_label_list = [[],[],[],[],[]]
    test_label_list = [[],[],[],[],[]]
    
    gens_ = [gen_train_0, gen_train_1, gen_train_2, gen_train_3, gen_train_4,
            gen_test_0, gen_test_1, gen_test_2, gen_test_3, gen_test_4]

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        train_keys_by_label = f.create_group('train_keys_by_label')
        test_keys_by_label = f.create_group('test_keys_by_label')
        g = 0
        
        for gen in tqdm(gens_):
            #pdb.set_trace()
            if g>4:
                istrain=False
            else:
                istrain=True
            
            for data, lbls in tqdm(gen):
                #pdb.set_trace()
                if istrain: 
                    day = g 
                    train_keys.append(key)
                    train_label_list[day].append(key)
                else:
                    day = g 
                    test_keys.append(key)
                    test_label_list[day-TEST_OFFSET].append(key)

                metas.append({'key':str(key), 'gesture':lbls, 'training sample':istrain}) 
                subgrp = data_grp.create_group(str(key))
                
                ad_dset = subgrp.create_dataset('samples' , data=data, dtype=np.uint8)
                lbl_dset= subgrp.create_dataset('labels', data=day,dtype=np.uint8)#lbls[i]-1, dtype=np.uint8)
                subgrp.attrs['meta_info']= str(metas[-1])
                #assert lbls[i]-1 in range(11)
                key += 1
                
            g += 1

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
            
            
            
def create_loader(directory, day='', param_file=''):
    if not os.path.isdir(directory):
        raise FileNotFoundError("EMG Data not found, looked at: {}".format(directory))
    
    
    params = Params(param_file)
    
    dataset = importlib.import_module(params['dataset'])
    try:
        create_data = dataset.create_data
    except AttributeError:
        create_data = dataset.create_dataloader
        
    data_loader_kwargs = filter_kwargs(params, create_data)
    
    window=int(params['deltat'] * params['sampling_rate'])
    
    gen_ = create_data(window=window,
                                  increment=int(params['increment_features']),# * params['sampling_rate']),
                                  window_features=int(params['window_features']),# * params['sampling_rate']),
                                  increment_features=int(params['increment_features']),# * params['sampling_rate']),
                                  #sample_length=int((window - (params['window_features'] - params['increment_features'])) / params['increment_features']),
                                  file_name=[directory+day+'/training_all.txt'],
                                  batch_size=1,
                                  features=params['features'],
                                  spiking=True,
                                  log_scale=True,
                                  one_dl=True)
    
    return gen_ 



if __name__=='__main__':
    
#     train_days = list(train_mapping.keys())
#     test_days = list(test_mapping.keys())
    
#     gen_ = create_loader(data_path, train_days[0], param_file)
    
    create_events_hdf5()

