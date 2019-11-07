#!/bin/python
#-----------------------------------------------------------------------------
# File Name : 
# Author: Emre Neftci
#
# Creation Date : Tue Nov  5 16:26:06 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
import torch
from torchvision.transforms import Compose,ToTensor,Normalize,Lambda

def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)

class Downsample(object):
    """Resize the address event Tensor to the given size.

    Args:
        factor: : Desired resize factor. Applied to all dimensions including time
    """
    def __init__(self, factor):
        assert isinstance(size, int) or (isinstance(size, Iterable))
        self.factor = factor
        self.interpolation = interpolation

    def __call__(self, tmad):
        return tmad//s

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Crop(object):
    def __init__(self, low_crop, high_crop):
        '''
        Crop all dimensions
        '''
        self.low = low_crop
        self.high = high_crop

    def __call__(self, tmad):
        idx = np.where(np.any(tmad>high_crop, axis=1))
        tmad = np.delete(tmad,idx,0)
        idx = np.where(np.any(tmad<high_crop, axis=1))
        tmad = np.delete(tmad,idx,0)
        return tmad

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CropDim(object):
    def __init__(self, low_crop, high_crop, dim):
        self.low = low_crop
        self.high = high_crop
        self.dim = dim

    def __call__(self, tmad):
        idx = np.where(tmad[:,self.dim]>high_crop)
        tmad = np.delete(tmad,idx,0)
        idx = np.where(tmad[:,self.dim]<low_crop)
        tmad = np.delete(tmad,idx,0)
        return tmad

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToCountFrame(object):
    """Convert Address Events to Binary tensor.

    Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (T x C x H x W) in the range [0., 1., ...] 
    """
    def __init__(self, size=[2, 32, 32],deltat=1000):
        self.size = size
        self.dt = deltat

    def __call__(self, tmad):
        times = tmad[:,0]
        t_start = times[0]
        t_end = times[-1]
        addrs = tmad[:,1:]

        ts = range(t_start, t_start-t_end, self.dt)
        chunks = np.zeros([len(ts)] + self.size, dtype='int8')
        idx_start = 0
        idx_end = 0
        for i, t in enumerate(ts):
            idx_end += find_first(times[idx_end:], t)
            if idx_end > idx_start:
                ee = addrs[idx_start:idx_end]
                pol, x, y = ee[:, 2], ee[:, 0], ee[:, 1]
                np.add.at(chunks, (i, pol, x, y), 1)
            idx_start = idx_end
        return torch.Tensor(chunks.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Repeat(object):
    '''
    Replicate np.array (C) as (n_repeat X C). This is useful to transform sample labels into sequences
    '''
    def __init__(self, n_repeat):
        self.n_repeat = n_repeat

    def __call__(self, target):
        return np.tile(np.expand_dims(target,0),[self.n_repeat,1])

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (T X H x W x C)
    """

    def __call__(self, frame):
        """
        Args:
            frame (numpy.ndarray): numpy array of frames

        Returns:
            Tensor: Converted data.
        """
        return torch.FloatTensor(frame)

    def __repr__(self):
        return self.__class__.__name__ + '()'



