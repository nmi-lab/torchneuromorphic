#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_aedat2.py
# Author: Emre Neftci
#
# Creation Date : Tue 16 Mar 2021 07:44:09 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv3
#----------------------------------------------------------------------------- 
from torchneuromorphic.utils import plot_frames_imshow, legacy_aedat_to_events
import torchneuromorphic.transforms as transforms
import sys 

events = legacy_aedat_to_events(sys.argv[1])
dt = 1000
size = [2, 240, 240]
process_to_countframe = transforms.Compose([
            transforms.Downsample(factor=[dt,1,1,1]),
            transforms.ToCountFrame(T = 500, size = size),
            #transforms.ToTensor()
            ])
frames = process_to_countframe(events)



