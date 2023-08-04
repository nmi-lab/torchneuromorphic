#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Author: Massimiliano Iacono
#         Dario Di Domenico
#
# Creation Date : Fri 01 Dec 2017 10:05:17 PM PST
# Last Modified : Sun 29 Jul 2018 01:39:06 PM PDT
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
# -----------------------------------------------------------------------------

import logging

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, Sampler

from training_utils.features_computation import compute_features_window, convert_nparray_to_spikes


class EMGDataset(Dataset):

    def __init__(self, data, labels, window=20, increment=10, features_list=None, log_scale=False, spiking=False):
        self.data = data
        self.labels = labels
        self.window = window
        self.increment = increment
        self.features_list = features_list
        self.norm_range = None
        self.log_scale = log_scale
        self.spiking = spiking

    def __getitem__(self, index):
        features = None
        try:
            feat_win = []
            labels = self.labels[index]
            raw = self.data[index]
            for i in range(0, len(raw) - (self.window - self.increment), self.increment):
                if self.features_list is not None:
                    features = compute_features_window(
                        raw[i: i + self.window], features_list=self.features_list)
                    if self.norm_range is not None:
                        features = normalization(features, self.norm_range)
                    feat_win.append(features)
        except IndexError:
            raise StopIteration
        out_data = np.array(feat_win).swapaxes(0, 1)
        if self.log_scale:
            out_data = np.log(1 + out_data)
        if self.spiking:
            out_data = convert_nparray_to_spikes(out_data)
        return out_data, labels[0].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def set_norm_range(self, norm_range):
        self.norm_range = norm_range


class SlidingWindowSampler(Sampler):

    def __init__(self, data_source, slide_step=1, window_size=1, shuffle=True, rep_duration=2, sampling_rate=2000):
        super().__init__(data_source)
        self.data_source = data_source
        self.slide_step = slide_step
        self.window_size = window_size
        self.shuffle = shuffle
        self.rep_break_ids = np.arange(
            0, len(self), sampling_rate * rep_duration)[1:]

    def __iter__(self):
        ids_list = list(range(0, len(self) - self.window_size +
                        self.slide_step, self.slide_step))
        windows = np.array([slice(x, x + self.window_size) for x in ids_list])

        is_unique_label = [
            len(np.unique(self.data_source.labels[x])) == 1 for x in windows]
        windows = windows[is_unique_label]
        is_window_across_rep = [not np.any(
            [x.start < i < x.stop for i in self.rep_break_ids]) for x in windows]
        windows = windows[is_window_across_rep]

        if self.shuffle:
            np.random.shuffle(windows)

        for win in windows:
            assert (len(np.unique(self.data_source.labels[win])) == 1)
            assert [not np.any([win.start < i < win.stop for i in self.rep_break_ids])]
            yield win

    def __len__(self):
        return len(self.data_source)


def calibrate(dataloader, iterations=50):
    features_boundaries = None
    assert iterations > 1
    for i in range(iterations):
        try:
            features, _ = next(dataloader)
            features = np.array(features)
        except StopIteration:
            break
        if features_boundaries is None:
            features_boundaries = {
                'max': features.max(0).max(0),
                'min': features.min(0).min(0)
            }
            continue
        features_boundaries['min'] = np.minimum(
            features_boundaries['min'], features.min(0).min(0))
        features_boundaries['max'] = np.maximum(
            features_boundaries['max'], features.max(0).max(0))
    return features_boundaries


def normalization(features_dict, norm_ranges):
    return (features_dict - norm_ranges['min']) / (norm_ranges['max'] - norm_ranges['min'])


def convert_to_xhz(self, data_slice):
    new_slice = torch.zeros(
        (data_slice.shape[0], data_slice.shape[1] // self.hz_factor, data_slice.shape[2]))

    for i in range(data_slice.shape[1] - (self.hz_factor - 1)):
        if i % self.hz_factor == 0:
            new_slice[:][i // self.hz_factor,
                         :] = torch.mean(data_slice[:, i:i + self.hz_factor])

    return new_slice


def clean_data(data):
    # Clean data from unwanted peaks
    # When the acquisition starts there is a peak in the data that we want to get rid of

    samples_to_cut = 200
    data = data[200:, :]
    acq_start_indexes = np.where(
        np.abs((np.diff(data, axis=0)).mean(1)) > 1000)

    for c, i in enumerate(acq_start_indexes[0]):
        data = np.delete(data, slice(i - c * samples_to_cut - 2,
                         i - c * samples_to_cut + samples_to_cut - 2), 0)

    return data


def read_header(file_name):
    header = []
    with open(file_name) as f:
        while True:
            line = f.readline()

            if not line.startswith('#'):
                break
            header.append(line[1:])
    # TODO add label_map to header
    header_dict = yaml.safe_load(''.join(header))
    return header_dict


def create_dataloader(batch_size, file_name, window, increment, window_features, increment_features, test_file_names=None, features=('all',), shuffle=True, log_scale=False, spiking=False, one_dl=False):
    logging.info('Loading...')
    # Load raw data from file + cleanup
    if test_file_names is None:
        test_file_names = []
    if not (isinstance(file_name, list) or isinstance(file_name, tuple)):
        file_name = (file_name, )
    if not (isinstance(test_file_names, list) or isinstance(test_file_names, tuple)):
        test_file_names = (test_file_names, )

    header_dict = read_header(file_name[0])
    data = []
    for f in file_name + test_file_names:
        new_header = read_header(f)
        compared_header = {
            k: header_dict[k] for k in header_dict if k in new_header and header_dict[k] == new_header[k]}
        if len(compared_header) != len(header_dict):
            raise ValueError(
                'Headers in all data file must contain the same parameters.')

    data.append(np.loadtxt(f).astype(np.float32))
    data = np.row_stack([np.loadtxt(f).astype(np.float32) for f in file_name])
    
    if one_dl:
        labels = data[:, -1]
        data = data[:, :64]
        
        emg_dataset = EMGDataset(data=data,
                                   labels=labels,
                                   features_list=features,
                                   window=window_features,
                                   increment=increment_features,
                                   spiking=spiking,
                                   log_scale=log_scale)
        
        dl = DataLoader(dataset=emg_dataset,
                          batch_size=batch_size,
                          sampler=SlidingWindowSampler(emg_dataset,
                                                       slide_step=increment,
                                                       window_size=window,
                                                       shuffle=shuffle,
                                                       rep_duration=header_dict['rep_duration'],
                                                       sampling_rate=header_dict['sampling_rate']))
        
        return dl


    if not test_file_names:
        # Splitting training and test set
        labels = data[:, -1]
        data = data[:, :64]  # TODO Parametrize 64 as nChann
        _, indices, counts = np.unique(
            labels, return_index=True, return_counts=True)
        test_indices = []
        train_indices = []
        for i, c in zip(indices, counts):
            split_idx = int(i + c * 0.2)
            test_indices.append(slice(i, split_idx))
            train_indices.append(slice(split_idx, i + c))

        train_data = np.concatenate([data[idx] for idx in train_indices])
        train_labels = np.concatenate([labels[idx] for idx in train_indices])
        test_data = np.concatenate([data[idx] for idx in test_indices])
        test_labels = np.concatenate([labels[idx] for idx in test_indices])
    else:
        train_data = data[:, :64]
        train_labels = data[:, -1]
        test_data_and_labels = np.row_stack(
            [np.loadtxt(f).astype(np.float32) for f in test_file_names])
        test_data = test_data_and_labels[:, :64]
        test_labels = test_data_and_labels[:, -1]

    emg_dataset_train = EMGDataset(data=train_data,
                                   labels=train_labels,
                                   features_list=features,
                                   window=window_features,
                                   increment=increment_features,
                                   spiking=spiking,
                                   log_scale=log_scale)

    emg_dataset_test = EMGDataset(data=test_data,
                                  labels=test_labels,
                                  features_list=features,
                                  window=window_features,
                                  increment=increment_features,
                                  spiking=spiking,
                                  log_scale=log_scale)

    train_dl = DataLoader(dataset=emg_dataset_train,
                          batch_size=batch_size,
                          sampler=SlidingWindowSampler(emg_dataset_train,
                                                       slide_step=increment,
                                                       window_size=window,
                                                       shuffle=shuffle,
                                                       rep_duration=header_dict['rep_duration'],
                                                       sampling_rate=header_dict['sampling_rate']))
    test_dl = DataLoader(dataset=emg_dataset_test,
                         batch_size=batch_size,
                         sampler=SlidingWindowSampler(emg_dataset_test,
                                                      slide_step=increment,
                                                      window_size=window,
                                                      shuffle=shuffle,
                                                      rep_duration=header_dict['rep_duration'],
                                                      sampling_rate=header_dict['sampling_rate']))

    return train_dl, test_dl
