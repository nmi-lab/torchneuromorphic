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
from torchneuromorphic.nmnist.create_nmnist_small_noxtgt_masked import *
from torchneuromorphic.utils import plot_frames_imshow
import torchneuromorphic.transforms as transforms
from torch.utils.data import Subset, SubsetRandomSampler
from pylab import *

if __name__ == "__main__":
    train_dl, test_dl = create_dataloader()

    d,t = next(iter(test_dl))
    plot_frames_imshow(d[0:2], nim=1, avg=50, transpose=True)

