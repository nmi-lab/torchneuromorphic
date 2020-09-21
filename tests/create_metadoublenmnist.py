#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_nmnist.py
# Author: Emre Neftci
#
# Creation Date : Thu Nov  7 20:30:14 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from torchneuromorphic.doublenmnist_torchmeta.doublenmnist_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
from matplotlib import pyplot as plt
from torchmeta.utils.data import CombinationMetaDataset
from torchmeta.utils.data import BatchMetaDataLoader, MetaDataLoader
from torchmeta.datasets.helpers import doublemnist
from torchmeta.transforms import Categorical, ClassSplitter




if __name__ == "__main__":
    root = 'data/nmnist/n_mnist.hdf5'
    batch_size = 72 
    chunk_size = 300
    ds = 1
    dt = 1000
    transform = None
    target_transform = None
    nclasses = 5
    ntasks = 3
    samples_per_class = 2
    samples_per_test = 2
    classes_meta = np.arange(100, dtype='int')

    size = [2, 32//ds, 32//ds]


    if transform is None:
        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            #ToEventSum(T = chunk_size, size = size),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()])

    if target_transform is None:
        target_transform = Compose([Repeat(chunk_size), toOneHot(nclasses)])

    cc = DoubleNMNIST(root = root, meta_test=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size,  num_classes_per_task=5)
    #cd = ClassNMNISTDataset(root,meta_train=True, transform = transform, target_transform = target_transform, chunk_size=chunk_size)

    dmnist_it = BatchMetaDataLoader(ClassSplitter(cc, shuffle=True, num_train_per_class=3, num_test_per_class=5), batch_size=16, num_workers=0)
    sample = next(iter(dmnist_it))
    data,targets = sample['train']

    ##Load torchmeta MNIST for comparison
    #from torchmeta.datasets.doublemnist import DoubleMNISTClassDataset
    #dataset = DoubleMNISTClassDataset("data/",meta_train=True)
    #dataset_h = BatchMetaDataLoader(doublemnist("data/",meta_train=True, ways=5, shots=10), batch_size=16, num_workers=0)

