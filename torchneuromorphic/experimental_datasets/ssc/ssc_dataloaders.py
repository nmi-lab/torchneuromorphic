import struct
import time
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from ..transforms import *
import os

mapping = range(35)

def one_hot1d(mbt, num_classes):
    out = np.zeros([num_classes], dtype='float32')
    out[int(mbt)] = 1
    return out

def create_events_hdf5(directory, hdf5_filename):
    train_evs, train_labels_isolated = load_shd_hdf5(directory+'/ssc_train.h5')
    test_evs, test_labels_isolated = load_shd_hdf5(directory+'/ssc_test.h5')
    border = len(train_labels_isolated)

    tmad = train_evs + test_evs
    labels = train_labels_isolated + test_labels_isolated 
    test_keys = []
    train_keys = []

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        for i,data in enumerate(tmad):
            times = data[:,0]
            addrs = data[:,1:]
            label = labels[i]
            out = []
            istrain = i<border
            if istrain: 
                train_keys.append(key)
            else:
                test_keys.append(key)
            metas.append({'key':str(key), 'training sample':istrain}) 
            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset('times' , data=times, dtype=np.uint32)
            ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint16)
            lbl_dset= subgrp.create_dataset('labels', data=label, dtype=np.uint8)
            subgrp.attrs['meta_info']= str(metas[-1])
            assert label in mapping
            key += 1
        extra_grp.create_dataset('train_keys', data=train_keys)
        extra_grp.create_dataset('test_keys', data=test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)

def load_shd_hdf5(filename, train=True):
    with h5py.File(filename, 'r', swmr=True, libver="latest") as f:
        evs = []
        labels = []
        for i,tl in enumerate(f['labels']):
            label_ = tl
            digit = tl
            digit=int(digit)
            labels.append(digit)
            tm = np.int32(f['spikes']['times'][i][:]*1e6)
            ad = np.int32(f['spikes']['units'][i][:])
            evs.append(np.column_stack([tm,ad]))
        return evs, labels

            
class SSCDataset(NeuromorphicDataset):
    resources_url = [['https://compneuro.net/datasets/ssc_test.h5.gz',None, 'ssc_test.h5.gz'], ['https://compneuro.net/datasets/ssc_train.h5.gz',None, 'ssc_train.h5.gz']]
    directory = 'data/shd/'
    resources_local = [directory+'/ssc_train.hdf5',directory+'/ssc_test.hdf5']


    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            chunk_size = 500):

        self.n = 0
        self.download_and_create = download_and_create
        self.root = root
        self.train = train 
        self.chunk_size = chunk_size

        super(SSCDataset, self).__init__(
                root,
                transform=transform,
                target_transform=target_transform )
        
        with h5py.File(root, 'r', swmr=True, libver="latest") as f:
            if train:
                self.n = f['extra'].attrs['Ntrain']
                self.keys = f['extra']['train_keys']
            else:
                self.n = f['extra'].attrs['Ntest']
                self.keys = f['extra']['test_keys']

    def download(self):
        isexisting = super(SSCDataset, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.directory, self.root)


    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if not self.train:
                key = key + f['extra'].attrs['Ntrain']
            data, target = sample(
                    f,
                    key,
                    T = self.chunk_size,
                    shuffle=self.train)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

def sample(hdf5_file,
        key,
        T = 500,
        shuffle = False):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tend = dset['times'][-1] 
    start_time = 0

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad, label

def create_dataloader(
        root = 'data/shd/shd.hdf5',
        batch_size = 72 ,
        chunk_size_train = 1000,
        chunk_size_test = 1000,
        ds = 1,
        dt = 500,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        **dl_kwargs):

    size = [700//ds, 1, 1]

    if transform_train is None:
        transform_train = Compose([
            Downsample(factor=[dt,ds]),
            ToChannelHeightWidth(),
            ToCountFrame(T = chunk_size_train, size = size),
            ToTensor()
            ])
    if transform_test is None:
        transform_test = Compose([
            Downsample(factor=[dt,ds]),
            ToChannelHeightWidth(),
            ToCountFrame(T = chunk_size_test, size = size),
            ToTensor()
            ])
    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train), toOneHot(len(mapping))])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test), toOneHot(len(mapping))])

    collate_fn = None
    #collate_fn = lambda x: zip(*[x[i] for i in range(len(x))])

    train_d = SSCDataset(root,train=True,
                                 transform = transform_train, 
                                 target_transform = target_transform_train, 
                                 chunk_size = chunk_size_train)

    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, collate_fn = collate_fn, **dl_kwargs)

    test_d = SSCDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 chunk_size = chunk_size_test)

    test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, collate_fn = collate_fn, **dl_kwargs)
                                 

    return train_dl, test_dl
