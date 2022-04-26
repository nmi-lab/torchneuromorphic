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

from collections import namedtuple, defaultdict
import torch
import torch.utils.data
from dv import AedatFile
import pandas
import itertools

NUM_CLASSES = 50 # each language included in the complete set

train_dir = 'data/nomniglot/dvs_background_1'
validation_dir = 'data/nomniglot/dvs_background_2'
test_dir = 'data/nomniglot/dvs_evaluation'

VAL_OFFSET = 16
TEST_OFFSET = 30

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

def nomniglot_load_events_from_aedat(aedat_file_path, csv_file_path):
    # each aedat has 20 samples so this will give a list of 20 samples
    # instead of just 1
    
    print(aedat_file_path)
    
    timestamp, polarity, x, y = [], [], [], []
    samples_per_character = 20
    lst = []
    
    # readout and store the information from the aedat file
    with AedatFile(aedat_file_path) as f:  # read aedat4
        for e in f['events']:
            timestamp.append(e.timestamp)
            polarity.append(e.polarity)
            x.append(e.x)
            y.append(e.y)
            
    # each aedat has 20 samples, deliniated by timestamps in the csv file
    start_end_timestamp = pandas.read_csv(csv_file_path).values
    
    for i in range(samples_per_character):
        start_index = timestamp.index(start_end_timestamp[i][1])
        end_index = timestamp.index(start_end_timestamp[i][2])
        tmp = np.column_stack([
            np.array(timestamp[start_index:end_index], dtype=np.uint32),
            np.array(polarity[start_index:end_index], dtype=np.uint8),
            np.array(x[start_index:end_index], dtype=np.uint16),
            np.array(y[start_index:end_index], dtype=np.uint16)])
        
        print(tmp.size)
        if tmp.size!=0:
            lst.append(tmp)
        else:
            print("empty sample")
        
    return lst


def get_file_names(dataset_path, num_per_class=0):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("NOmniglot Dataset not found, looked at: {}".format(dataset_path))

    dict_files = {}
#     dict_validation = {}
#     dict_test = {}
    
    # in case fewer samples are needed for memory reasons or something
    # num_train = 
    # num_validation = 
    # num_test = 
    
    samples = 0 # samples are really multiplied by 20 because each file has 20 samples
    
    for root, subdirectories, files in os.walk(dataset_path):
        for subdirectory in subdirectories:
            if num_per_class and samples >= num_per_class:
                continue
            
            if "character" in os.path.join(root, subdirectory) and ".ipynb" not in os.path.join(root, subdirectory):
                dir_path = os.path.join(root, subdirectory).split('/')
                
                if dir_path[3] in dict_files.keys():
                    dict_files[dir_path[3]].append(os.path.join(root, subdirectory))
                else:
                    dict_files[dir_path[3]] = [os.path.join(root, subdirectory)]
                    
                samples += 1
        samples = 0
                
                
                    

    return dict_files #train, dict_validation, dict_test

def create_events_hdf5(directory='', hdf5_filename=''):#, num_instances=):
    
    directory = 'data/nomniglot/'
    hdf5_filename = 'nomniglot_one.hdf5'
    dict_train = get_file_names(directory+'dvs_background_1',1)
    dict_validation = get_file_names(directory+'dvs_background_2',1)
    dict_test = get_file_names(directory+'dvs_evaluation',1)
    
    print(dict_train['Alphabet_of_the_Magi'])
    
    test_keys = []
    validation_keys = []
    train_keys = []
    train_label_list = [[] for i in range(16)]
    validation_label_list = [[] for i in range(14)]
    test_label_list = [[] for i in range(20)]
    
    # test to make sure an hdf5 can be made ok 
    # this looks like it will take a long time for all the data so try just one alphabets
    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        
        for k, v in dict_train.items():
            #if k == "Braille":
            for path in v:
                for f in os.listdir(path):
                    if os.path.isfile(os.path.join(path, f)):
                        if ".aedat4" in f:
                            aedat_path = os.path.join(path, f)
                        elif ".csv" in f:
                            csv_path = os.path.join(path,f)

                samples = nomniglot_load_events_from_aedat(aedat_path, csv_path)

                for data in samples:
                    times = data[:,0]
                    addrs = data[:,1:]
                    label = train_mapping[k]
                    
                    print("K", k)
                    print("LABEL", label)
                    print("KEY", key)

                    train_keys.append(key)

                    train_label_list[label].append(key)

                    metas.append({'key':str(key), 'training sample':True}) 
                    subgrp = data_grp.create_group(str(key))
                    tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
                    ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
                    lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
                    subgrp.attrs['meta_info']= str(metas[-1])
                    key += 1
                
                        
        print(len(train_keys))
        
        for k, v in dict_validation.items():
            #if k == "Braille":
            for path in v:
                for f in os.listdir(path):
                    if os.path.isfile(os.path.join(path, f)):
                        if ".aedat4" in f:
                            aedat_path = os.path.join(path, f)
                        elif ".csv" in f:
                            csv_path = os.path.join(path,f)

                samples = nomniglot_load_events_from_aedat(aedat_path, csv_path)

                for data in samples:
                    times = data[:,0]
                    addrs = data[:,1:]
                    label = validation_mapping[k]

                    validation_keys.append(key)

                    validation_label_list[label-VAL_OFFSET].append(key)

                    metas.append({'key':str(key), 'validation sample':True}) 
                    subgrp = data_grp.create_group(str(key))
                    tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
                    ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
                    lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
                    subgrp.attrs['meta_info']= str(metas[-1])
                    key += 1
                    
        print(len(validation_keys))
        
        for k, v in dict_test.items():
            #if k == "Braille":
            for path in v:
                for f in os.listdir(path):
                    if os.path.isfile(os.path.join(path, f)):
                        if ".aedat4" in f:
                            aedat_path = os.path.join(path, f)
                        elif ".csv" in f:
                            csv_path = os.path.join(path,f)

                samples = nomniglot_load_events_from_aedat(aedat_path, csv_path)

                for data in samples:
                    times = data[:,0]
                    addrs = data[:,1:]
                    label = test_mapping[k]

                    test_keys.append(key)

                    test_label_list[label-TEST_OFFSET].append(key)

                    metas.append({'key':str(key), 'test sample':True}) 
                    subgrp = data_grp.create_group(str(key))
                    tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
                    ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
                    lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
                    subgrp.attrs['meta_info']= str(metas[-1])
                    key += 1
                    
        print(len(test_keys))
        
#         pad_token = -1
        
#         train_label_list = list(zip(*itertools.zip_longest(*train_label_list, fillvalue=pad_token)))
        
#         validation_label_list = list(zip(*itertools.zip_longest(*validation_label_list, fillvalue=pad_token)))
        
#         test_label_list = list(zip(*itertools.zip_longest(*test_label_list, fillvalue=pad_token)))
        
        print(train_label_list)
                
        extra_grp.create_dataset('train_keys', data = train_keys)
        extra_grp.create_dataset('train_keys_by_label', data = train_label_list)
        extra_grp.create_dataset('validation_keys_by_label', data = validation_label_list)
        extra_grp.create_dataset('validation_keys', data = validation_keys)
        extra_grp.create_dataset('test_keys_by_label', data = test_label_list)
        extra_grp.create_dataset('test_keys', data = test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(validation_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
        extra_grp.attrs['Nvalidation'] = len(validation_keys)

if __name__=="__main__":
    
    create_events_hdf5()
#     directory = 'data/nomniglot/'
#     hdf5_filename = 'nomniglot.hdf5'
#     dict_train = get_file_names(directory+'dvs_background_1',1)
#     dict_validation = get_file_names(directory+'dvs_background_2',1)
#     dict_test = get_file_names(directory+'dvs_evaluation',1)
    
#     print(dict_train['Alphabet_of_the_Magi'])
    
#     test_keys = []
#     validation_keys = []
#     train_keys = []
#     train_label_list = [[] for i in range(16)]
#     validation_label_list = [[] for i in range(14)]
#     test_label_list = [[] for i in range(20)]
    
#     # test to make sure an hdf5 can be made ok 
#     # this looks like it will take a long time for all the data so try just one alphabets
#     with h5py.File(hdf5_filename, 'w') as f:
#         f.clear()
#         key = 0
#         metas = []
#         data_grp = f.create_group('data')
#         extra_grp = f.create_group('extra')
        
#         for k, v in dict_train.items():
#             #if k == "Braille":
#             for path in v:
#                 for f in os.listdir(path):
#                     if os.path.isfile(os.path.join(path, f)):
#                         if ".aedat4" in f:
#                             aedat_path = os.path.join(path, f)
#                         elif ".csv" in f:
#                             csv_path = os.path.join(path,f)

#                 samples = nomniglot_load_events_from_aedat(aedat_path, csv_path)

#                 for data in samples:
#                     times = data[:,0]
#                     addrs = data[:,1:]
#                     label = train_mapping[k]
                    
#                     print("K", k)
#                     print("LABEL", label)

#                     train_keys.append(key)

#                     train_label_list[label].append(key)

#                     metas.append({'key':str(key), 'training sample':True}) 
#                     subgrp = data_grp.create_group(str(key))
#                     tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
#                     ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
#                     lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
#                     subgrp.attrs['meta_info']= str(metas[-1])
#                     key += 1
                
                        
#         print(len(train_keys))
        
#         for k, v in dict_validation.items():
#             #if k == "Braille":
#             for path in v:
#                 for f in os.listdir(path):
#                     if os.path.isfile(os.path.join(path, f)):
#                         if ".aedat4" in f:
#                             aedat_path = os.path.join(path, f)
#                         elif ".csv" in f:
#                             csv_path = os.path.join(path,f)

#                 samples = nomniglot_load_events_from_aedat(aedat_path, csv_path)

#                 for data in samples:
#                     times = data[:,0]
#                     addrs = data[:,1:]
#                     label = validation_mapping[k]

#                     validation_keys.append(key)

#                     validation_label_list[label].append(key)

#                     metas.append({'key':str(key), 'validation sample':True}) 
#                     subgrp = data_grp.create_group(str(key))
#                     tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
#                     ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
#                     lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
#                     subgrp.attrs['meta_info']= str(metas[-1])
#                     key += 1
                    
#         print(len(validation_keys))
        
#         for k, v in dict_test.items():
#             #if k == "Braille":
#             for path in v:
#                 for f in os.listdir(path):
#                     if os.path.isfile(os.path.join(path, f)):
#                         if ".aedat4" in f:
#                             aedat_path = os.path.join(path, f)
#                         elif ".csv" in f:
#                             csv_path = os.path.join(path,f)

#                 samples = nomniglot_load_events_from_aedat(aedat_path, csv_path)

#                 for data in samples:
#                     times = data[:,0]
#                     addrs = data[:,1:]
#                     label = test_mapping[k]

#                     test_keys.append(key)

#                     test_label_list[label].append(key)

#                     metas.append({'key':str(key), 'test sample':True}) 
#                     subgrp = data_grp.create_group(str(key))
#                     tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
#                     ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
#                     lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
#                     subgrp.attrs['meta_info']= str(metas[-1])
#                     key += 1
                    
#         print(len(test_keys))
        
# #         pad_token = -1
        
# #         train_label_list = list(zip(*itertools.zip_longest(*train_label_list, fillvalue=pad_token)))
        
# #         validation_label_list = list(zip(*itertools.zip_longest(*validation_label_list, fillvalue=pad_token)))
        
# #         test_label_list = list(zip(*itertools.zip_longest(*test_label_list, fillvalue=pad_token)))
        
#         #print(train_label_list)
                
#         extra_grp.create_dataset('train_keys', data = train_keys)
#         extra_grp.create_dataset('train_keys_by_label', data = train_label_list)
#         extra_grp.create_dataset('validation_keys_by_label', data = validation_label_list)
#         extra_grp.create_dataset('validation_keys', data = validation_keys)
#         extra_grp.create_dataset('test_keys_by_label', data = test_label_list)
#         extra_grp.create_dataset('test_keys', data = test_keys)
#         extra_grp.attrs['N'] = len(train_keys) + len(validation_keys) + len(test_keys)
#         extra_grp.attrs['Ntrain'] = len(train_keys)
#         extra_grp.attrs['Ntest'] = len(test_keys)
#         extra_grp.attrs['Nvalidation'] = len(validation_keys)
