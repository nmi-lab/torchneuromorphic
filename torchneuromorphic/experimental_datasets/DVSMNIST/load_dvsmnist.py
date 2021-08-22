import h5py
from collections import namedtuple
import glob
import numpy as np
import os
import struct
import time
import torch
import torch.utils.data
from ..utils improt load_ATIS_bin, load_jaer

dcll_folder = ''
dvsmnist_hdf5_filename = os.path.join(dcll_folder, 'data/dvsmnist_events.hdf5')
nmnist_hdf5_filename = os.path.join(dcll_folder, 'data/nmnist_events.hdf5')
dvsmnist_dataset_path = os.path.join(dcll_folder, 'data/dvsmnist')
nmnist_dataset_path = os.path.join(dcll_folder, 'data/N-MNIST')

Stream = namedtuple('Stream', ['timestamps', 'x', 'y', 'pol'])

def nmnist_load_events_from_bin(file_path, max_duration=None):
    start_time = time.time()
    timestamps, xaddr, yaddr, pol = load_ATIS_bin(file_path)
    timestamps = timestamps.astype(float)
    timestamps *= 1e-6
    assert(timestamps[0] <= timestamps[-1]), 'ts overflow not handled in read'
    min_ts = min(timestamps)
    timestamps -= min_ts
    if max_duration is not None:
        keep_indices = timestamps <= max_duration
    else:
        keep_indices = np.ones_like(timestamps).astype(bool)

    stream = Stream(timestamps=timestamps[keep_indices],
                    x=np.array(xaddr)[keep_indices],
                    y=np.array(yaddr)[keep_indices],
                    pol=np.array(pol)[keep_indices])
    return stream

def dvsmnist_load_events_from_aedat(file_path, max_duration=None):
    start_time = time.time()
    timestamps, xaddr, yaddr, pol = load_jaer(file_path, version='aedat', debug=0)
    timestamps = timestamps.astype(float)
    timestamps *= 1e-6
    if timestamps[0] > timestamps[-1]:
        print('HAD TO RESTORE TS ORDER')
        ordered_indices = np.argsort(timestamps)
        timestamps = timestamps[ordered_indices]
        xaddr = xaddr[ordered_indices]
        yaddr = yaddr[ordered_indices]
        pol = pol[ordered_indices]
    min_ts = min(timestamps)
    timestamps -= min_ts
    if max_duration is not None:
        keep_indices = timestamps <= max_duration
    else:
        keep_indices = np.ones_like(timestamps).astype(bool)

    stream = Stream(timestamps=timestamps[keep_indices],
                    x=np.array(xaddr)[keep_indices],
                    y=np.array(yaddr)[keep_indices],
                    pol=np.array(pol)[keep_indices])
    return stream

def dvsmnist_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("DVS MNIST Dataset not found, looked at: {}".format(dataset_path))

    all_path = glob.glob(os.path.join(dataset_path, 'grabbed_data*/scale16/*.aedat'))
    return sorted(all_path)

def nmnist_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("N-MNIST Dataset not found, looked at: {}".format(dataset_path))

    train_files = []
    test_files = []
    for digit in range(10):
        digit_train = glob.glob(os.path.join(dataset_path, 'Train/{}/*.bin'.format(digit)))
        digit_test = glob.glob(os.path.join(dataset_path, 'Test/{}/*.bin'.format(digit)))
        train_files.append(digit_train)
        test_files.append(digit_test)

    # We need the same number of train and test samples for each digit, let's compute the minimum
    max_n_train = min(map(lambda l: len(l), train_files))
    max_n_test = min(map(lambda l: len(l), test_files))
    n_train = 2000 # we could take max_n_train, but my memory on the shared drive is full
    n_test = 100 # we test on the whole test set - lets only take 100*10 samples
    assert((n_train <= max_n_train) and (n_test <= max_n_test)), 'Requested more samples than present in dataset'

    print("N-MNIST: {} train samples and {} test samples per digit (max: {} train and {} test)".format(n_train, n_test, max_n_train, max_n_test))
    # Crop extra samples of each digits
    train_files = map(lambda l: l[:n_train], train_files)
    test_files = map(lambda l: l[:n_test], test_files)

    return list(train_files), list(test_files)

def split_train_test(n_samples, randomize, train_or_test):
    ids = []
    for i in range(0, 10000, 1000):
        if train_or_test == 'train':
            ids += range(i, i + 900)
        elif train_or_test == 'test':
            ids += range(i + 900, i + 1000)
    if randomize:
        sample_ids = random.sample(ids, n_samples)
    else:
        sample_ids = ids[:n_samples]
    labels = np.zeros(10000, dtype=int)
    for i in range(10):
        labels[i * 1000:(i + 1) * 1000] = i
    labels = [int(labels[i]) for i in sample_ids]
    return sample_ids, labels

def dvsmnist_split_train_test_for_digit(digit, train_or_test, n_train=900, n_test=100):
    assert (n_train+n_test<=1000), "Requested number of samples exceed dataset size"
    i = digit * 1000 # digits are separated by 1000 in the dataset
    if train_or_test == 'train':
        sample_ids = range(i, i + n_train)
    elif train_or_test == 'test':
        sample_ids = range(i + n_train, i + n_train + n_test)
    return sample_ids

def split_train_test(n_samples, randomize, train_or_test):
    ids = []
    for i in range(0, 10000, 1000):
        if train_or_test == 'train':
            ids += range(i, i + 900)
        elif train_or_test == 'test':
            ids += range(i + 900, i + 1000)
    if randomize:
        sample_ids = random.sample(ids, n_samples)
    else:
        sample_ids = ids[:n_samples]
    labels = np.zeros(10000, dtype=int)
    for i in range(10):
        labels[i * 1000:(i + 1) * 1000] = i
    labels = [int(labels[i]) for i in sample_ids]
    return sample_ids, labels

def events_to_tensor(im_dims, events, chunk_size=200, chunk_dt=0.001, dims=[128, 128], crop=False):
    # format: (time, channels, y, x)
    ret = np.zeros((chunk_size, 2, im_dims[0], im_dims[1]), dtype='uint16')

    h_ratio = float(im_dims[0]) / dims[0]
    w_ratio = float(im_dims[1]) / dims[1]
    crop_delta = ((np.array(dims) - np.array(im_dims)) / 2.).astype('uint16')

    for t, x, y, pol in zip(events.timestamps, events.x, events.y, events.pol):
        # determine time index in tensor with respect to t
        # t_idx = np.clip(int(t / chunk_dt), 0, chunk_size-1)
        t_idx = int(t / chunk_dt)
        if (t_idx < 0) or (t_idx >= chunk_size):
            continue
        if crop:
            # add a spike if cropped pixel is within the center
            cropped_pixel = np.array((y, x), dtype='uint16') - crop_delta
            if np.all( ([0, 0] <= cropped_pixel) & (cropped_pixel < im_dims) ):
                ret[t_idx, pol, cropped_pixel[0], cropped_pixel[1]] += 1
        else:
            # add a spike at the resized pixel, either +=1 or =1 make sense
            ret[t_idx, pol, int(y * h_ratio), int(x * w_ratio)] += 1
    return ret

def nmnist_read_train_test_for_digit(digit, dataset_files, train_or_test='train', sample_duration=None, chunk_size=1500, im_dims=[32,32]):
    # indices = split_train_test_for_digit(digit, train_or_test, n_train=50, n_test=50) # DEBUG
    train_files, test_files = dataset_files
    files = train_files[digit] if train_or_test is 'train' else test_files[digit]

    start_time = time.time()
    streams = list(map(lambda f: nmnist_load_events_from_bin(f, max_duration=sample_duration),
                  files))
    stream_time = time.time()
    print("Took {}s to read events for digit {} ({})".format(stream_time - start_time, digit, train_or_test))

    tensors = [events_to_tensor(im_dims, st, chunk_size=chunk_size, dims=[34, 34], crop=True) for st in streams]
    print("Took {}s to convert events".format(time.time() - stream_time))
    return tensors

def dvsmnist_read_train_test_for_digit(digit, dataset_files, train_or_test='train', sample_duration=None, chunk_size=1500, im_dims=[32,32]):
    # indices = split_train_test_for_digit(digit, train_or_test, n_train=50, n_test=50) # DEBUG
    indices = dvsmnist_split_train_test_for_digit(digit, train_or_test)
    start_time = time.time()
    streams = [dvsmnist_load_events_from_aedat(dataset_files[i], max_duration=sample_duration) for i in indices]
    stream_time = time.time()
    print("Took {}s to read events for digit {} ({})".format(stream_time - start_time, digit, train_or_test))

    tensors = [events_to_tensor(im_dims, st, chunk_size=chunk_size) for st in streams]
    print("Took {}s to convert events".format(time.time() - stream_time))
    return tensors

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_sample_temporal_histogram(input_spikes, label_spikes):
    import matplotlib.pyplot as plt
    events = torch.sum(input_spikes, dim=(1,2,3)) # sum over polarity, x and y
    fig, ax = plt.subplots()
    digit = np.nonzero(label_spikes).item()
    ax.set_title('Sample for digit {}'.format(digit))
    ax.plot(smooth(events, 20)) # smooth the signal with temporal convolution
    plt.show()

def plot_sample_spatial_histogram(input_spikes, label_spikes):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    image = torch.sum(input_spikes, dim=(0,1)) # sum over time and polarity
    fig, ax = plt.subplots()

    im = ax.imshow(image)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    digit = np.nonzero(label_spikes).item()
    ax.set_title('Sample for digit {}'.format(digit))
    plt.show()

def create_events_hdf5(dataset_files, hdf5_filename, sample_duration=None, max_chunk_size=1500,
                       digit_reader=dvsmnist_read_train_test_for_digit):
    print("Creating hdf5 file from dataset")

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        print("processing training data...")
        group = f.create_group('train')
        for digit in range(0, 10):
            digit_group = group.create_group(str(digit))
            tensors = digit_reader(digit, dataset_files, train_or_test='train',
                                   sample_duration=sample_duration, chunk_size=max_chunk_size)
            digit_group.create_dataset(name='default', data=tensors, chunks=True)

        print("processing testing data...")
        group = f.create_group('test')
        for digit in range(0, 10):
            digit_group = group.create_group(str(digit))
            tensors = digit_reader(digit, dataset_files, train_or_test='test',
                                   sample_duration=sample_duration, chunk_size=max_chunk_size)
            digit_group.create_dataset(name='default', data=tensors, chunks=True)

class DVSMNIST(torch.utils.data.Dataset):
    def __init__(self, filename=dvsmnist_hdf5_filename, chunk_size=500,
                 train_or_test='train', make_hdf5=False):
        super().__init__()
        if not os.path.isfile(filename) or make_hdf5:
            dataset_files = dvsmnist_get_file_names(dvsmnist_dataset_path)
            create_events_hdf5(dataset_files, filename)
        self.chunk_size=chunk_size
        f = h5py.File(filename, 'r', swmr=True, libver="latest")
        self.group = f[train_or_test]

    def __getitem__(self, index):
        n_rows = len(self.group[str(0)]['default'])

        # get data
        digit = int(index / n_rows)
        sub_idx = index % n_rows

        x = self.group[str(digit)]['default'][sub_idx][:self.chunk_size].astype(np.float32)
        x = torch.from_numpy(x)

        # get label in one-hot encoding
        y = torch.zeros(10, dtype=torch.float32)
        y[digit] = 1.
        return (x, y)

    def __len__(self):
        # total = 0
        # for digit in range(10):
        #     total += len(self.group[str(digit)]['default'])
        # return total
        return len(self.group[str(0)]['default']) * 10

class NMNIST(torch.utils.data.Dataset):
    def __init__(self, filename=nmnist_hdf5_filename, chunk_size=500,
                 train_or_test='train', make_hdf5=False, batchsize=1):
        super().__init__()
        if not os.path.isfile(filename) or make_hdf5:
            dataset_files = nmnist_get_file_names(nmnist_dataset_path)
            create_events_hdf5(dataset_files, filename,
                               sample_duration=0.3, max_chunk_size=300,
                               digit_reader=nmnist_read_train_test_for_digit)
        self.chunk_size=chunk_size
        f = h5py.File(filename, 'r', swmr=True, libver="latest")
        self.group = f[train_or_test]

    def __getitem__(self, index):
        n_rows = len(self.group[str(0)]['default'])

        # get data
        digit = int(index / n_rows)
        sub_idx = index % n_rows

        x = self.group[str(digit)]['default'][sub_idx][:self.chunk_size].astype(np.float32)
        x = torch.from_numpy(x)

        # get label in one-hot encoding
        y = torch.zeros(10, dtype=torch.float32)
        y[digit] = 1.
        return (x, y)

    def __len__(self):
        # total = 0
        # for digit in range(10):
        #     total += len(self.group[str(digit)]['default'])
        # return total
        return len(self.group[str(0)]['default']) * 10

class SequenceGeneratorWrapper(object):
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
    def next(self):
        try:
            (input_spikes, labels_spikes) = next(self.iterator)
        except StopIteration:
            # reset the iterator
            self.iterator = iter(self.loader)
            (input_spikes, labels_spikes) = next(self.iterator)
        return (input_spikes.numpy(), labels_spikes.numpy())
    
    def __len__(self):
        return len(self.loader)
        
    def __iter__(self):
        for i in range(len(self)):
            yield self.next()

def create_data(batch_size = 64 ,
                chunk_size_train = 300,
                chunk_size_test = 300,
                size = [2, 32, 32],
                ds = 4,
                dt = 1000):
    if chunk_size_train != 300 or chunk_size_test != 300:
        raise NotImplementedError('N-MNIST only supports chunk size 300')

    train_loader = torch.utils.data.DataLoader(
        NMNIST(train_or_test='train', make_hdf5=False,
                 chunk_size=chunk_size_train),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        NMNIST(train_or_test='test', make_hdf5=False,
                 chunk_size=chunk_size_test),
        batch_size=batch_size, shuffle=False)
    # return SequenceGeneratorWrapper(train_loader), SequenceGeneratorWrapper(test_loader)
    return train_loader, test_loader
