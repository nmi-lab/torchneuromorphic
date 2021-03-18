#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_aedat_legacy_timesurface.py
# Author: Emre Neftci
#
# Creation Date : Tue 16 Mar 2021 01:28:22 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv3
#----------------------------------------------------------------------------- 
from torchneuromorphic.utils import plot_frames_imshow, legacy_aedat_to_events
import torchneuromorphic.transforms as transforms
import pylab as plt
import numpy as np
import sys

device = 'cuda'
events = legacy_aedat_to_events(sys.argv[1])
dt = 1000
size = [2, 346, 260]
process_events = transforms.Compose([
            transforms.Downsample(factor=[dt,1,1,1]),
            transforms.ToCountFrame(T = 1000, size = size),
            transforms.ToTensor(),
            transforms.ExpFilterEvents(tau=100, length=500, device=device)
            ])
frames =  process_events(events)


plot_frames_imshow(np.array([frames.detach().cpu().numpy()]), nim=1)
plt.show()
