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
import unittest
import importlib

class TestNMNIST(unittest.TestCase):
    dataset_name = 'nmnist'

    #def test_create(self):
    #    create_mod= importlib.import_module(
    #            'torchneuromorphic.datasets.'+self.dataset_name+'.download_create')
    #    create_mod.create_events_hdf5('data/{nm}/'.format(nm=self.dataset_name), 'data/{nm}/{nm}.hdf5'.format(nm=self.dataset_name))
    #    self.assertTrue()

    def test_dataloader(self):
        create_dl= importlib.import_module('torchneuromorphic.datasets.'+self.dataset_name+'.create_dl')
        train_dl, test_dl = create_dl.create_dataloader(
            root='data/{nm}/{nm}.hdf5'.format(nm = self.dataset_name),
            batch_size=32,
            ds=1,
            num_workers=0)
        tr_it = iter(train_dl)
        frames, labels = next(tr_it)
        self.assertEqual(list(frames.shape), [32,300,2,32,32])



class TestDVSGestures(unittest.TestCase):
    dataset_name = 'dvsgestures'

    #def test_create(self):
    #    create_mod= importlib.import_module(
    #            'torchneuromorphic.datasets.'+self.dataset_name+'.download_create')
    #    create_mod.create_events_hdf5(
    #            'data/{nm}/'.format(nm=self.dataset_name),
    #            'data/{nm}/{nm}.hdf5'.format(nm=self.dataset_name))
    #    self.assertTrue()

    def test_dataloader(self):
        create_dl= importlib.import_module('torchneuromorphic.datasets.'+self.dataset_name+'.create_dl')
        train_dl, test_dl = create_dl.create_dataloader(
            root='data/{nm}/{nm}.hdf5'.format(nm = self.dataset_name),
            batch_size=32,
            ds=1,
            num_workers=0)
        tr_it = iter(train_dl)
        frames, labels = next(tr_it)
        self.assertEqual(list(frames.shape), [32,300,2,32,32])




if __name__ == '__main__':
    unittest.main()


    

