#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_dvsgestures.py
# Author: Emre Neftci
#
# Creation Date : Fri 19 Sep 2019 
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv3
#----------------------------------------------------------------------------- 
from torchneuromorphic.nmnist.nmnist_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
import torchneuromorphic.transforms as transforms
from torch.utils.data import Subset, SubsetRandomSampler

def create_dataloader(
        root = 'data/nmnist/n_mnist.hdf5',
        batch_size = 72 ,
        chunk_size_train = 300,
        chunk_size_test = 300,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        **dl_kwargs):
    
    ds = [1,1] 
    low_crop = [0,0]
    high_crop = [32,32]
    size = [2, np.ceil((high_crop[0]-low_crop[0])/ds[0]).astype('int'), np.ceil((high_crop[1]-low_crop[1])/ds[1]).astype('int')]

    print(size)
    default_transform = lambda chunk_size:  transforms.Compose([
         transforms.CropDims(low_crop,high_crop,[2,3]),
         transforms.Downsample(factor=[dt,1,ds[0],ds[1]]),
         transforms.ToCountFrame(T = chunk_size, size = size),
         transforms.ToTensor()
    ])


    if transform_train is None:
        transform_train = default_transform(chunk_size_train)
    if transform_test is None:
        transform_test = default_transform(chunk_size_test)

    if target_transform_train is None:
        target_transform_train = transforms.Compose([
            transforms.Repeat(chunk_size_train),
            transforms.toOneHot(10)])
    if target_transform_test is None:
        target_transform_test =  transforms.Compose([
            transforms.Repeat(chunk_size_test),
            transforms.toOneHot(10)])

    train_d = NMNISTDataset(root,
                            train=True,
                            transform = transform_train, 
                            target_transform = target_transform_train, 
                            chunk_size = chunk_size_train)


    train_subset_indices = train_d.keys_by_label[:,:100].reshape(-1)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, sampler=SubsetRandomSampler(train_subset_indices), **dl_kwargs)

    test_d = NMNISTDataset(root,
                               transform = transform_test, 
                               target_transform = target_transform_test, 
                               train=False,
                               chunk_size = chunk_size_test,
                               )


    test_subset_indices = test_d.keys_by_label[:,:100].reshape(-1) 
    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, sampler=SubsetRandomSampler(test_subset_indices), **dl_kwargs)
    return train_dl, test_dl

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader()

    import tqdm
    for d,t in tqdm.tqdm(iter(test_dl)): 
        print(d.shape)
