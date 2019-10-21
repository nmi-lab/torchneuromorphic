#!/usr/bin/env python
"""Train a CNN for Google speech commands. Modified from """
__author__ = 'Yuan Xu, Erdene_Ochir Tuguldur'

import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision
from torchvision.transforms import *
from transforms import *
from speech_commands_dataset import *

def create_dataloaders(data_location         = 'data/projects_data_dir/',
                       batch_size            = 128,
                       dataload_workers_nums = 6,
                       input_type            = 'mel32'):
     
    train_dataset         = data_location + 'speech_commands/train'
    valid_dataset         = data_location + 'speech_commands/valid'
    background_noise      = data_location + 'speech_commands/train/_background_noise_'

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    n_mels = 32
    if input_type == 'mel40':
        n_mels = 40

    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(background_noise, data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_dataset = SpeechCommandsDataset(train_dataset,
                                    Compose([LoadAudio(),
                                             data_aug_transform,
                                             add_bg_noise,
                                             train_feature_transform]))

    valid_feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    valid_dataset = SpeechCommandsDataset(valid_dataset,
                                    Compose([LoadAudio(),
                                             FixAudioLength(),
                                             valid_feature_transform]))

    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  pin_memory=use_gpu, num_workers=dataload_workers_nums)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  pin_memory=use_gpu, num_workers=dataload_workers_nums)

    return train_dataloader, valid_dataloader

if __name__ == "__main__":
    train_dataloader, valid_dataloader = create_dataloaders()