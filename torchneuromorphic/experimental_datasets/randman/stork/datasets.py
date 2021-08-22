#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import tables

import os, sys 
import string
import re
import hashlib

import torch
import torch.utils.data

from . import utils

import threading
lock = threading.Lock()

def synchronized_open_file(*args, **kwargs):
    with lock:
        return tables.open_file(*args, **kwargs)

def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)


# Load Randman library
# sys.path.append(os.path.expanduser("/tungstenfs/scratch/gzenke/zenkfrie/projects/"))
import randman


def standardize(x,eps=1e-7):
    mi,_ = x.min(0)
    ma,_ = x.max(0)
    return (x-mi)/(ma-mi+eps)


def split_dataset(X,y,splits=[0.8,0.2], shuffle=True):
    """ Splits a dataset into training, validation and test set etc..

    Args:
        X: The data
        y: The labels
        splits: The split ratios (default=0.8,0.2)
    
    Returns:
        Tuple of x_train, y_train, x_test, y_test, etc ...
    """

    splits = np.array(splits)

    if (splits<=0).any():
        raise AssertionError("Split requires positive splitting ratios greater than zero.")
    splits /= splits.sum()

    if shuffle:
        idx = np.arange(len(X),dtype=np.int)
        np.random.shuffle(idx)
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]

    start = 0
    sets = []
    for split in splits:
        idx_split = int(split*len(X))
        end = start+idx_split
        sets.append( (X[start:end], y[start:end]) )
        start = end
    return sets

        

def current2firing_time(x, tau=50e-3, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value 
    tmax -- The maximum time returned 
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x>thr
    T = torch.ones_like(x)*tmax
    T[idx] = tau*torch.log(x[idx]/(x[idx]-thr))
    return T



def make_tempo_randman(nb_classes=10, nb_units=100, nb_steps=100, step_frac=1.0, dim_manifold=2, nb_spikes=2, nb_samples=1000, alpha=2.0, shuffle=True, classification=True, seed=None):
    """ Generates event based generalized tempo randman classification dataset. 

    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding wont work. 
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.

    Args: 
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
        seed: The random seed (default: None)

    Returns: 
        A tuple of data,labels. The data is structured as numpy array 
        (sample x event x 2 ) where the last dimension contains 
        the relative [0,1] (time,unit) coordinates and labels.
    """
  
    data = []
    labels = []
    targets = []

    if seed is not None:
        np.random.seed(seed)

    rng_state = torch.random.get_rng_state()
    
    max_value = np.iinfo(np.int).max
    randman_seeds = np.random.randint(max_value, size=(nb_classes,nb_spikes) )

    for k in range(nb_classes):
        x = np.random.rand(nb_samples,dim_manifold)
        submans = [ randman.Randman(nb_units, dim_manifold, alpha=alpha, seed=randman_seeds[k,i]) for i in range(nb_spikes) ]
        units = []
        times = []
        for i,rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(np.repeat(np.arange(nb_units).reshape(1,-1),nb_samples,axis=0))
            times.append(y.numpy())

        units = np.concatenate(units,axis=1)
        times = np.concatenate(times,axis=1)
        events = np.stack([times,units],axis=2)
        data.append(events)
        labels.append(k*np.ones(len(units)))
        targets.append(x)

    data = np.concatenate(data, axis=0)
    labels = np.array(np.concatenate(labels, axis=0), dtype=np.int)
    targets = np.concatenate(targets, axis=0)

    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]
        targets = targets[idx]

    data[:,:,0] *= nb_steps*step_frac
    data = np.array(data, dtype=int)


    # restore torch.random state
    torch.random.set_rng_state(rng_state)

    # TODO improve code efficiency. 
    # Should build the format directly in the code above instead of converting it here.
    # QnD fix: Convert back to a list of 2-tuples
    data = [ (torch.from_numpy(d[:,0]),torch.from_numpy(d[:,1])) for d in data ]

    if classification:
        return data, labels
    else:
        return data, targets




# Class definitions


class SpikingDataset(torch.utils.data.Dataset):
    """
    Provides a base class for all spiking dataset objects.
    """
    def __init__(self, nb_steps, nb_units, p_drop=0.0, p_insert=0.0, sigma_t=0.0, sigma_u=0.0, sigma_u_uniform=0.0, time_scale=1 ):
        """
        This converter provides an interface for standard spiking datasets 

        Args:
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
            sigma_u: Amplitude of unit jitter added to each spike (default 0). The jitter is applied *after* unit scaling.
            sigma_u_uniform: Uniform noise amplitude added to all units (default 0). The jitter is applied *after* unit scaling.
            time_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
        """
        super().__init__()
        self.p_drop  = p_drop
        self.p_insert = p_insert
        self.sigma_t = sigma_t
        self.sigma_u = sigma_u
        self.sigma_u_uniform = sigma_u_uniform
        self.time_scale = time_scale
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.nb_insert = int(self.p_insert*self.nb_steps*self.nb_units)
        self.data_augmentation = True


    def add_noise(self, times, units):
        """
        Expects lists of times and units as arguments and then adds spike noise to them.
        """

        if self.sigma_t:
            dt = torch.randn(len(times))*self.sigma_t
            times = times + dt

        if self.sigma_u or self.sigma_u_uniform:
            if self.sigma_u:
                du = (torch.rand(len(units))-0.5)*self.sigma_u
                units = units + du
            if self.sigma_u_uniform:
                du = torch.randn(1)*self.sigma_u_uniform
                units = units + du

        if self.p_drop:
            rnd=torch.rand(len(times))
            idx=rnd>self.p_drop
            times=times[idx]
            units=units[idx]
            
        if self.p_insert: # insert spurious extra spikes
            insert_times = (torch.rand(self.nb_insert)*self.nb_steps).long()
            insert_units = (torch.rand(self.nb_insert)*self.nb_units).long()
            times = torch.cat((times, insert_times))
            units = torch.cat((units, insert_units))

        return times, units


    def get_valid(self, times, units):
        """ Return only the events that fall inside the input specs. """

        # Tag spikes which would otherwise fall outside of our nb_steps
        idx = (times>=0)&(times<self.nb_steps)

        idxu = (units>=0)&(units<self.nb_units)
        idx = idx & idxu

        # Remove spikes which would fall outside of our nb_steps or nb_units
        times = times[idx]
        units = units[idx]

        return times, units


    def preprocess_events(self, times, units):
        """ Apply data augmentation and filter out invalid events. """

        if self.data_augmentation: 
            times, units = self.add_noise(times, units)

        times, units = self.get_valid(times, units)
        return times.long(), units.long()


class RasDataset(SpikingDataset):
    def __init__(self, dataset, nb_steps, nb_units, p_drop=0.0, p_insert=0.0, sigma_t=0.0, time_scale=1 ):
        """
        This converter provides an interface for standard Ras datasets to dense tensor format. 

        Args:
            dataset: (data,labels) tuple where data is in RAS format
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
            time_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
        """
        super().__init__(nb_steps, nb_units, p_drop=p_drop, p_insert=p_insert, sigma_t=sigma_t, time_scale=time_scale)

        data, labels = dataset

        if self.time_scale==1:
            Xscaled = data
        else:
            Xscaled = []
            for times,units in data:
                times = self.time_scale*times
                idx = times<self.nb_steps
                Xscaled.append((times[idx], units[idx]))

        self.data = Xscaled 
        self.labels = labels
        if type(self.labels)==torch.tensor:
            self.labels = torch.cast(labels,dtype=torch.long)


    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.data)


    def __getitem__(self, index):
        "Returns one sample of data"

        times, units = self.data[index]
        times, units = self.preprocess_events(times, units)

        times = times.long() 
        # units = units.long() # since torch cannot handle uint16
        
        X = torch.zeros( (self.nb_steps, self.nb_units) )
        X[times,units] = 1.0
        y = self.labels[index]

        return X, y


class SpikeLatencyDataset(RasDataset):
    def __init__(self, data, nb_steps, nb_units, time_step=1e-3, tau=50e-3, thr=0.1, p_drop=0.0, p_insert=0.0, sigma_t=0.0 ):
        """ This dataset takes standard (vision) datasets as input and provides a time to first spike dataset.

        Args:
            tau: Membrane time constant (default=50ms)
            thr: Firing threshold (default=0.1)
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
        """

        self.time_step = time_step
        self.thr = thr
        self.tau = tau
        ras_data = self.prepare_data(data,tau_eff=tau/time_step,thr=thr,tmax=nb_steps)
        super().__init__(ras_data, nb_steps, nb_units, p_drop=p_drop, p_insert=p_insert, sigma_t=sigma_t, time_scale=1)


    def prepare_data(self, data, tau_eff, thr, tmax):
        X,y = data
        nb_units = X.shape[1]

        # compute discrete firing times
        times = current2firing_time(X, tau=tau_eff,  thr=self.thr, tmax=tmax).long()
        units = torch.arange(nb_units,dtype=torch.long)

        labels = y.long()
        ras = [  ]
        for i in range(len(X)):
            idx = times[i]<tmax
            ras.append( (times[i,idx], units[idx]) ) 
        return (ras,labels)






class HDF5Dataset(SpikingDataset):
    def __init__(self, h5filepath, nb_steps, nb_units, p_drop=0.0, p_insert=0.0, sigma_t=0.0, sigma_u=0.0, sigma_u_uniform=0.0, time_scale=1.0, unit_scale=1.0, 
            unit_permutation=None, add_random_delay=0, preload=False, precompute_dense=False, sparse_output=False, coalesced=False ):
        """
        This dataset acts as an interface for HDF5 datasets to dense tensor format. 
        Per default this dataset class is not thread-safe unless used with the preload option. 

        Args:
            h5filepath: The path and filename of the HDF5 file containing the data.
            p_drop: Probability of dropping a spike (default 0).
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0). The jitter is applied *after* time scaling.
            sigma_u: Amplitude of unit jitter added to each spike (default 0). The jitter is applied *after* unit scaling.
            sigma_u_uniform: Uniform noise amplitude added to channels
            time_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
            unit_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
            permute_units: Permute order of units before scaling
            add_random_delay: Add random delay for each units
            preload: If set to true the datasets are first loaded into RAM instead of read from the HDF5 file directly.
            precompute_dense: If set to true the dense dataset is computed and stored in RAM (Warning! This may use a lot of RAM).
            sparse_output: If set to True, return sparse output tensor.
        """
        super().__init__(nb_steps, nb_units, p_drop=p_drop, p_insert=p_insert, sigma_t=sigma_t, sigma_u=sigma_u, sigma_u_uniform=sigma_u_uniform, time_scale=time_scale)
        self.unit_scale = unit_scale
        self.sparse_output = sparse_output
        self.coalesced = coalesced
        self.precompute_dense = precompute_dense
        self.add_random_delay = add_random_delay
        self.permutation = unit_permutation

        if add_random_delay:
            self.delays = np.random.randint(add_random_delay,size=int(nb_units/unit_scale))

        if preload:
            self.h5file = fileh = synchronized_open_file(h5filepath, mode='r')
            self.units  = [ x for x in fileh.root.spikes.units ]
            self.times  = [ x for x in fileh.root.spikes.times ]
            self.labels = [ x for x in torch.tensor(fileh.root.labels, dtype=torch.long) ]
            synchronized_close_file(fileh)
        else:
            self.h5file = fileh = tables.open_file(h5filepath, mode='r')
            self.units = fileh.root.spikes.units
            self.times = fileh.root.spikes.times
            self.labels = torch.tensor(fileh.root.labels, dtype=torch.long)

        if precompute_dense:
            self.dataset  = [ self.get_dense(i) for i in range(len(self.labels)) ]


    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.labels)


    def get_dense(self, index):
        "Convert a single sample from event-based format to dense."

        times = (torch.from_numpy(self.time_scale*self.times[index])).float() 

        # TODO Implement the random delay

        if self.permutation is None:
            units = np.array(self.unit_scale*self.units[index],dtype=int)
        else:
            units = np.array(self.unit_scale*self.permutation[self.units[index]],dtype=int)
        units = torch.from_numpy(units)

        times, units = self.preprocess_events(times, units)

        if self.sparse_output:
            y = self.labels[index]
            return (times, units), y
        else:
            if self.coalesced:
                # Slow but coalesced
                indices = torch.LongTensor(torch.stack([times,units],axis=1).T)
                values = torch.FloatTensor(torch.ones(len(times)))
                X = torch.sparse.FloatTensor(indices, values, torch.Size([self.nb_steps,self.nb_units])).to_dense()
            else:
                # Fast but not coalesced
                X = torch.zeros( (self.nb_steps, self.nb_units) )
                X[times,units] = 1.0

            # Really slow and stupid
            # X = np.zeros( (self.nb_steps, self.nb_units) )
            # for t,u in zip(times,units):
            #     X[int(t),int(u)] += 1
            # X = torch.from_numpy(X).float()
            
            y = self.labels[index]
            return X, y


    def __getitem__(self, index):
        "Returns one sample of data"
        if self.precompute_dense:
            return self.dataset[index]
        else:
            return self.get_dense(index)




class DatasetView(torch.utils.data.Dataset):
    def __init__(self, dataset, elements):
        """
        This meta dataset provides a view onto an underlying Dataset by selecting a subset of elements specified in an index list.

        Args:
            dataset: The mother dataset instance
            elements: A list with indices of the data points in the mother dataset
        """
        super().__init__()
        self.dataset = dataset
        self.elements = elements

    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.elements)

    def __getitem__(self, index):
        "Returns one sample of data"
        return self.dataset[self.elements[index]]



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, nb_steps, nb_units, p_drop=0.0, p_insert=0.0, sigma_t=0.0, time_scale=1 ):
        """
        This converter provides an interface for text datasets to dense tensor character-prediction datasets. 

        Args:
            data: The data in ras format
        """
        super().__init__()
        self.nb_steps = nb_steps
        self.nb_units = nb_units

        chars = tuple(set(text))
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}

        # Encode the text
        encoded = np.array([char2int[ch] for ch in text], dtype=np.int)
        nb_samples = int(len(encoded)//nb_steps)
        encoded = encoded[:nb_samples*nb_steps] # truncate
        self.data = encoded.reshape((nb_samples,nb_steps))
        self.times = np.arange(nb_steps)


    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.data)


    def __getitem__(self, index):
        "Returns one sample of data"

        units = self.data[index]
        X = torch.zeros( (self.nb_steps, self.nb_units) )
        y = torch.zeros( (self.nb_steps, self.nb_units) )
        X[self.times,units] = 1.0
        y[:-1] = X[1:]

        return X, y


# Experimental datasets 
import soundfile as sf

class RawAudioDataset:
    """ Provides a baseclass for audio datasets. 

    Args:
        nb_steps: The number of time steps to return
        time_step: The size of the assumed time step in seconds
        nb_fft: The number of FFT channels to compute
        nb_filt: The number of mel-spaced filters to compute in the filter bank
        frame_size: The window size in seconds to compute FFT over
        pre_emphasis: A pre-emphasis parameter for preprocessing
        standardize (bool): Whether to standardize the filter banks
        diffcode (bool): Whether to return differences of consecutive frames
        binarize (bool): Whether to return binary values 0 and 1 
        resize (bool): Whether to resize the data to all have the same shape for easy stacking
        repeat_last_frame (bool): Whether to repeat the last frame instead of zero padding

    This baseclass implements a rudimentary conversion of audio data to a mel-spaced spectrum to be used as raw
    input to network models.
    """
    def __init__(self, nb_steps, time_step=10e-3, nb_fft=512, nb_filt=40, frame_size=25e-3, pre_emphasis=0.95, 
            standardize=True, diffcode=False, binarize=False, resize=True, repeat_last_frame=True):

        self.nb_fft = nb_fft
        self.nb_filt = nb_filt
        self.frame_stride = time_step
        self.frame_size   = frame_size
        self.pre_emphasis = pre_emphasis

        self.standardize = standardize
        self.diffcode = diffcode
        self.binarize = binarize
        self.resize = resize
        self.repeat_last_frame = repeat_last_frame

        self.length = nb_steps
        self.data = []
        self.signal_lengths = []



    def recursive_walk(self, rootdir):
        """
        Yields:
            str: All filnames in rootdir, recursively.
        """
        for r, dirs, files in os.walk(rootdir):
            for f in files:
                yield os.path.join(r, f)
        
    def get_signal(self, fname):
        """ Returns audio data and sampling rate from audio file. """
        signal, sample_rate = sf.read(fname)
        signal = signal.astype('float32')
        return signal, sample_rate
        
    def get_feature(self, fname):
        """ Compute mel-spaced feature from audio data file. 

        Args:
            fname (str): The file name of the audio file to operate on

        Returns: 
            An audio feature
        """

        # Code mostly taken from
        # https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
        
        # Load from file
        signal, sample_rate = self.get_signal(fname)
        
        # Pre-Emphasis
        emphasized_signal = np.append(signal[0], signal[1:] - self.pre_emphasis * signal[:-1])

        # Framing
        frame_length, frame_step = self.frame_size * sample_rate, self.frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # Hamming window
        frames *= np.hamming(frame_length)

        # Fourier transform
        mag_frames = np.absolute(np.fft.rfft(frames, self.nb_fft))  # Magnitude of the FFT
        pow_frames = ((1.0 / self.nb_fft) * ((mag_frames) ** 2))  # Power Spectrum

        # Apply filter banks
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.nb_filt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        fbin = np.floor((self.nb_fft + 1) * hz_points / sample_rate)

        fbank = np.zeros((self.nb_filt, int(np.floor(self.nb_fft / 2 + 1))))
        for m in range(1, self.nb_filt + 1):
            f_m_minus = int(fbin[m - 1])   # left
            f_m = int(fbin[m])             # center
            f_m_plus = int(fbin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - fbin[m - 1]) / (fbin[m] - fbin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (fbin[m + 1] - k) / (fbin[m + 1] - fbin[m])

                
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Ensure numerical Stability

        self.signal_lengths.append(len(filter_banks))
        if self.resize:
            fillstart = len(filter_banks)
            filter_banks.resize(self.length, self.nb_filt) # TODO find better solution for this

            if self.repeat_last_frame and fillstart<=self.length:
                    filter_banks[fillstart:]=filter_banks[fillstart-1]

        
        # filter_banks = 20 * np.log10(filter_banks)  # dB
        filter_banks = np.log10(filter_banks)  # dB

        if self.standardize:
            mean, std = np.mean(filter_banks), np.std(filter_banks)
            filter_banks = (filter_banks-mean)/(std+np.finfo(float).eps)

        if self.diffcode:
            diff = np.diff(filter_banks, axis=0)
            filter_banks = diff

        if self.binarize:
            filter_banks = 1.0*(filter_banks>0.0)

        filter_banks = torch.from_numpy(filter_banks).float()
        return filter_banks

    def __len__(self):
        return len(self.data)

    def prepare_item(self, index):
        raise NotImplemented 

    def __getitem__(self, index):
        return self.data[index]
        
    def cache(self, fname):
        """ Save current data to cache file. """
        utils.write_to_file(self.data, fname)

    def load_cached(self, fname):
        """ Load from cached data file. """
        self.data = utils.load_from_file(fname)



class RawHeidelbergDigits(RawAudioDataset):
    """ Dataset adapter to work with the Heidelberg dataset 

    Args:
        dirname (str): The root directory where the raw Heidelberg digits dataset can be found
        nb_steps (int): The number of steps to return
        time_step (float): The time step size in seconds (default 10ms)
        subset (list): List with file basnames belonging to the set
        label (str): Which labels to use digit, speaker, or language
        diffcode (bool): Whether to return difference between frames or frames
        binarize (bool): Whether to return binary values 0 and 1 
    """

    def __init__(self, dirname, nb_steps, time_step=10e-3, subset=None, label='digit', diffcode=False, binarize=False, cache_fname=None):

        super().__init__(nb_steps, time_step=time_step, diffcode=diffcode, binarize=binarize)

        self.dirname = dirname
        assert os.path.isdir(dirname), dirname
        self.filelist = [k for k in self.recursive_walk(self.dirname)
                          if k.endswith('.flac')]
        print("Found {} flac files ...".format(len(self.filelist)))
        assert len(self.filelist), "Found no '.flac' files!"
        assert label in ["digit", "speaker", "language"], label
        self.label = label
        
        self.length = nb_steps
        if diffcode:
            self.length = nb_steps+1

        if subset is not None:
            self.filelist = [ fn for fn in self.filelist if os.path.basename(fn) in subset ]
            print("Selected {} flac files ...".format(len(self.filelist)))

        if cache_fname is None:
            self.load_data()
        else:
            try: 
                self.load_cached(cache_fname)
                print("Finished loading {} cached data ...".format(len(self.data)))
            except FileNotFoundError:
                self.load_data()
                self.cache(cache_fname)


    def load_data(self):
        self.data = [ self.prepare_item(i) for i in range(len(self.filelist)) ]
        print("Loaded %i data..."%(len(self.data)))

                
    def parse_fname(self,fname):
        bn = os.path.basename(fname)
        bn = os.path.splitext(bn)[0]
        tokens = bn.split("_")
        lang, speaker, trial, digit = [ t.split('-')[1] for t in tokens ] 
        return lang, speaker, trial, digit

    def prepare_item(self, index):
        fn = self.filelist[index]
        feat = self.get_feature(fn)
        lang, speaker, trial, digit = self.parse_fname(fn)
        if self.label == 'digit':
            label = int(digit)
            if lang=="english": label += 10
        elif self.label == 'speaker':
            raise NotImplemented
        elif self.label == 'language':
            raise NotImplemented
        return feat, label
    

class RawSpeechCommands(RawAudioDataset):
    """ Dataset adapter for the Google Speech Commands dataset by Pete Warden

    Args:
        dirname (str): The root directory where the raw audio data can be found
        nb_steps (int): The number of steps to return (default 100)
        time_step (float): The time step size in seconds (default 10ms)
        subset (str): Which subset to return. Either "all", "training", "validation", or "testing"
        label_mode (str): How to assign labels. Either "word prediction" or "keyword spotting"
        diffcode (bool): Whether to return difference between frames or frames
    """
    def __init__(self, dirname, nb_steps=100, time_step=10e-3, subset="all", label_mode='word prediction', shuffle=False, diffcode=False, binarize=False, cache_fname=None):

        super().__init__(nb_steps, time_step=time_step, diffcode=diffcode, binarize=binarize)

        self.shuffle = shuffle
        self.dirname = dirname
        assert os.path.isdir(dirname), dirname
        self.filelist = [k for k in self.recursive_walk(self.dirname)
                          if k.endswith('.wav')]
        print("Found {} wav files ...".format(len(self.filelist)))
        assert len(self.filelist), "Found no '.wav' files!"
        assert label_mode in ["word prediction", "keyword spotting"], label_mode
        self.label_mode = label_mode
        assert subset in ["all", "training", "validation", "testing"], subset
        self.subset = subset
        # TODO have to still implement the subset logic

        if label_mode=="word prediction":
            word_list = ["backward", "bed", "bird", "cat", "dog", "down",
                    "eight", "five", "follow", "forward", "four", "go",
                    "happy", "house", "learn", "left", "marvin", "nine", "no",
                    "off", "on", "one", "right", "seven", "sheila", "six",
                    "stop", "three", "tree", "two", "up", "visual", "wow",
                    "yes", "zero"]
            self.skip_list = ["_background_noise_",]
        elif label_mode=="keyword spotting":
            word_list = ["yes", "no", "up", "down", "left",
                    "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
                    "five", "six", "seven", "eight", "nine"]
            self.skip_list = []

        self.word_dic = {v: k for k, v in enumerate(word_list)}
        self.unknown_label = len(word_list)
        
        self.length = nb_steps
        if diffcode:
            self.length = nb_steps+1

        if cache_fname is None:
            self.load_data()
        else:
            try: 
                self.load_cached(cache_fname)
                print("Finished loading {} cached data ...".format(len(self.data)))
            except FileNotFoundError:
                self.load_data()
                self.cache(cache_fname)


    def load_data(self):
        data = []
        for fn in self.filelist:
            tokens = fn.split("/")
            key  = tokens[-2]
            if key in self.skip_list: continue
            if self.subset=="all" or self.which_set(os.path.basename(fn))==self.subset: 
                data.append(self.prepare_item(fn))
        print("Finished loading {} data ...".format(len(data)))

        if self.shuffle:
            print("Shuffling ...")
            np.random.shuffle(data)

        self.data = data 
                

    def parse_fname(self,fname):
        tokens = fname.split("/")
        key  = tokens[-2]
        if key in self.word_dic.keys():
            return self.word_dic[key]
        else:
            return self.unknown_label


    def prepare_item(self, fn):
        feat = self.get_feature(fn)
        label = self.parse_fname(fn)
        return feat, label

    def which_set(self, filename, validation_percentage=0.1, testing_percentage=0.1):
        """Determines which data partition the file should belong to.

        We want to keep files in the same training, validation, or testing sets even
        if new ones are added over time. This makes it less likely that testing
        samples will accidentally be reused in training when long runs are restarted
        for example. To keep this stability, a hash of the filename is taken and used
        to determine which set it should belong to. This determination only depends on
        the name and the set proportions, so it won't change as other files are added.

        It's also useful to associate particular files as related (for example words
        spoken by the same person), so anything after '_nohash_' in a filename is
        ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
        'bobby_nohash_1.wav' are always in the same set, for example.

        Args:
          filename: File path of the data sample.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          String, one of 'training', 'validation', or 'testing'.
        """
        MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

        base_name = os.path.basename(filename)
        # We want to ignore anything after '_nohash_' in the file name when
        # deciding which set to put a wav in, so the data set creator has a way of
        # grouping wavs that are close variations of each other.
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        # This looks a bit magical, but we need to decide whether this file should
        # go into the training, testing, or validation sets, and we want to keep
        # existing files in the same set even if more files are subsequently
        # added.
        # To do that, we need a stable way of deciding based on just the file name
        # itself, so we do a hash of that and then use that to generate a
        # probability value that we use to assign it.
        hash_name_hashed = hashlib.sha1(hash_name.encode("utf8")).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (1.0 / MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < validation_percentage:
            result = 'validation'
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = 'testing'
        else:
            result = 'training'
        return result





