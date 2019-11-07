import struct
import numpy as np
import scipy.misc
import h5py
import glob
import os
import torch.utils.data
from ..events_timeslices import frame_evs, one_hot

def one_hot1d(mbt, num_classes):
    out = np.zeros([num_classes], dtype='float32')
    out[int(mbt)] = 1
    return out

def make_infinite(dataloader):
    while True:
        for x in iter(dataloader):
            yield x

def load_tidigit_hdf5(filename, train=True):
    with h5py.File(filename, 'r', swmr=True, libver="latest") as f:
        if train:
            train_evs = []
            train_labels_isolated = []
            for tl in f['train_labels']:
                label_ = tl.decode()
                label_s = label_.split('-')
                if len(label_s[-1])==1:
                    digit = label_s[-1]
                    if digit is 'o':
                        digit=10
                    if digit is 'z':
                        digit=0
                    else:
                        digit=int(digit)
                    train_labels_isolated.append(digit)
                    tm = np.int32(f['train_timestamps'][label_][:]*1e6)
                    ad = np.int32(f['train_addresses'][label_].value)
                    train_evs.append(np.column_stack([tm,ad]))
            return train_evs, train_labels_isolated

        else:
            test_evs = []
            test_labels_isolated  = []
            for tl in f['test_labels']:
                label_ = tl.decode()
                label_s = label_.split('-')
                if len(label_s[-1])==1:
                    digit = label_s[-1]
                    if digit is 'o':
                        digit=10
                    if digit is 'z':
                        digit=0
                    else:
                        digit=int(digit)
                    test_labels_isolated.append(digit)
                    tm = np.int32(f['test_timestamps'][label_][:]*1e6)
                    ad = np.int32(f['test_addresses'][label_].value)
                    test_evs.append(np.column_stack([tm,ad]))
            
            return test_evs, test_labels_isolated
            
class NTIdigitsDataset(torch.utils.data.Dataset):
    nclasses = 11
    
    def __init__(self, 
            filename,
            train=True,
            ds=[1], #transform
            size=[64], #transform
            dt=1000, #transform
            max_duration=1000, #transform
            download=False
            ):
        super(NTIdigitsDataset).__init__()
        if download:
            raise NotImplementedError()
        evs, labels = load_tidigit_hdf5(filename, train=train)
        self.evs = evs
        self.labels = labels
        self.ds = ds
        self.dt = dt
        self.size = size
        self.max_duration = max_duration

    def __getitem__(self, key):
        tm = self.evs[key][:,0]
        ad = self.evs[key][:,1:2]
        #T = np.minimum(tm[-1]//self.dt,self.max_duration)
        T = self.max_duration
        sample = frame_evs(tm, ad, duration=T, downsample=[self.ds], size=self.size, deltat=self.dt)
        #learn_mask = np.convolve(np.ones(100)/100,sample.reshape(-1,self.size[0]).sum(axis=1))[:T]>1e-3
        learn_mask = sample.reshape(-1,self.size[0]).sum(axis=1)>0
        targets = np.ones(T)*11
        targets[learn_mask] = self.labels[key]
        return np.array(sample, dtype='float32'), one_hot(targets, self.nclasses+1)[:,:self.nclasses].astype('float32')
    
    def __len__(self):
        return len(self.evs)
        

def create_data(filename = 'data/tidigits/n-tidigits.hdf5',
                chunk_size_train=10,
                chunk_size_test=20,
                batch_size=50,
                size=[2, 128, 128],
                dt = 1000,
                ds = 1,
                **dl_kwargs):
    
    """
    chunk_size_train: number of samples in the time axis for the training set
    chunk_size_test: number of samples in the time axis for the testing set
    batch_size: batch_size on each iteration
    size: expected input size, list in the format [channels, dimx]
    dt: framing delta t in microseconds
    ds: downsampling factor in the dimx dimension. 1 means no downsampling
    """
    if not os.path.isfile(filename):
        raise Exception("File {} does not exist".format(filename))

    train_d = NTIdigitsDataset(filename, train=True, ds = ds, size = size, dt = dt, max_duration = chunk_size_train)
    train_dl = torch.utils.data.DataLoader(train_d,
                                           batch_size=batch_size,
                                           shuffle=True, **dl_kwargs)
    
    test_d   = NTIdigitsDataset(filename, train=False, ds = ds, size = size, dt = dt, max_duration = chunk_size_test)
    test_dl = torch.utils.data.DataLoader( test_d,
                                           batch_size=batch_size,
                                           shuffle=True, **dl_kwargs)

    return train_dl, test_dl
