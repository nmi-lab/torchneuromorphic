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
from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *
from torchneuromorphic.utils import plot_frames_imshow
import torchneuromorphic.transforms as transforms

def create_dataloader(
        root = 'data/dvsgesture/dvs_gestures_build19.hdf5',
        batch_size = 72 ,
        chunk_size_train = 500,
        chunk_size_test = 1800,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        **dl_kwargs):
    
    n_events_attention = 100 

    default_transform = lambda chunk_size:  transforms.Compose([
         transforms.CropDims(low_crop,high_crop,[2,3]),
         transforms.Downsample(factor=[dt,1,ds[0],ds[1]]),
         transforms.ToCountFrame(T = chunk_size, size = size),
         transforms.ToTensor()
    ])
    
    size = [2,64,64]
    default_transform = lambda chunk_size: transforms.Compose([
            transforms.Downsample(factor=[dt,1,1,1]),
            transforms.Attention(n_events_attention, size=size),
            transforms.Downsample(factor=[1,1,4,4]),
            transforms.ToCountFrame(T = chunk_size, size = [2,16,16]),
            transforms.ToTensor()
        ])

    if transform_train is None:
        transform_train = default_transform(chunk_size_train)
    if transform_test is None:
        transform_test = default_transform(chunk_size_test)

    if target_transform_train is None:
        target_transform_train = transforms.Compose([
            transforms.Repeat(chunk_size_train),
            transforms.toOneHot(11)])
    if target_transform_test is None:
        target_transform_test =  transforms.Compose([
            transforms.Repeat(chunk_size_test),
            transforms.toOneHot(11)])

    train_d = DVSGestureDataset(root,
                                train=True,
                                transform = transform_train, 
                                target_transform = target_transform_train, 
                                chunk_size = chunk_size_train)

    train_dl = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True, **dl_kwargs)

    test_d = DVSGestureDataset(root,
                               transform = transform_test, 
                               target_transform = target_transform_test, 
                               train=False,
                               chunk_size = chunk_size_test)

    test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader(
            batch_size=32,
            num_workers=0)
    ho = iter(train_dl)
    frames, labels = next(ho)
    plot_frames_imshow(frames, labels)

