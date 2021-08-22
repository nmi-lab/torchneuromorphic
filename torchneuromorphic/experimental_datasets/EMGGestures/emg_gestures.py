import struct
import numpy as np
import scipy.misc
import h5py
import glob
import os
import torch.utils.data
from ..events_timeslices import frame_evs, one_hot

SAMPLING_RATE = 200
mapping = { 0 :'Pinch1' ,
            1 :'Pinch2' ,
            2 :'Pinch3' ,
            3 :'Pinch4' ,
            -1 :'none' }
inv_mapping = {'Pinch1': 0,
               'Pinch2': 1,
               'Pinch3': 2,
               'Pinch4': 3,
               'none'  :-1}



def one_hot1d(mbt, num_classes):
    out = np.zeros([num_classes], dtype='float32')
    out[int(mbt)] = 1
    return out

def make_infinite(dataloader):
    while True:
        for x in iter(dataloader):
            yield x
            
class EMGGesturesDataset(torch.utils.data.Dataset):
    nclasses = 11
    
    def __init__(self, evs, labels, ds=[1], size=[64], dt=1000, max_duration=1000):
        super(NTIdigitsDataset).__init__()
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
        

def create_data(filename = 'data/pinch.hdf5',
                chunk_size_train=10,
                chunk_size_test=20,
                batch_size=50,
                size=[2, 128, 128],
                dt = 1000,
                ds = 1,
                download = True,
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
    else:
        from torchvision.datasets.utils import download_and_extract_archive
        download_and_extract_archive('https://zenodo.org/record/3194792/files/Pinch.zip?download=1', 'data/', extract_root = 'data/', filename = 'pinch.zip', remove_finished = True)

        with h5py.File(filename, 'w') as f:
            f.clear()
            data = f.create_group('data')
            extras = f.create_group('extras')

            labels = f.create_group('labels')


            emg_data = glob.glob('data/Pinch/*emg.npy')

            lab_data = glob.glob('data/Pinch/*ann.npy')

            key=0
            for ef in emg_data:
                ef_data = np.load(ef)
                ef_labels = np.load(ef[:-8]+"_ann.npy")
                ef_labels_idx = np.zeros([ef_labels.shape[0]])
                for k,v in inv_mapping.items():
                    ef_labels_idx[ef_labels==k]=v

                ef_time = np.linspace(0, ef_data.shape[0]*1000000/SAMPLING_RATE, ef_data.shape[0],dtype='uint32')
                dev  = data.create_group(str(key))
                dtgt = labels.create_group(str(key))
                dset_dt = dev.create_dataset('time', data = ef_time, dtype=np.uint32)
                dset_da = dev.create_dataset('data', data = ef_data, dtype=np.int32)
                dset_l  = dtgt.create_dataset('class', data=ef_labels_idx, dtype=np.int8)
                key = key + 1



            dset_dt = events.create_dataset('time',  [], dtype=np.uint32)
            dset_da = events.create_dataset('data',  [], dtype=np.uint16)
            dset_l  = labels.create_dataset('class', [], dtype=np.uint32)





