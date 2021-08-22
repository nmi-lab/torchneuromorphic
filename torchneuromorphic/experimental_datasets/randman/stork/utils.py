import gzip
import pickle
import torch
import string
import random
import time


def get_random_string(string_length=5):
    """ Generates a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


def get_basepath(dir=".",prefix="default", salt_length=5):
    """ Returns pre-formatted and time stamped basepath given a base directory and file prefix. """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if salt_length:
        salt = get_random_string(salt_length)
        basepath = "%s/%s-%s-%s"%(dir,prefix,timestr,salt)
    else:
        basepath = "%s/%s-%s"%(dir,prefix,timestr)
    return basepath


def write_to_file(data, filename):
    """ Writes an object/dataset to zipped pickle.

    Arguments:
    data -- the (data) object
    filename -- the filename to write to
    """
    fp = gzip.open("%s"%filename,'wb')
    pickle.dump(data, fp)
    fp.close()


def load_from_file(filename):
    """ Loads an object/dataset from a zipped pickle. """
    fp = gzip.open("%s"%filename,'r')
    data = pickle.load(fp)
    fp.close()
    return data


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    # x_typename = torch.typename(x).split('.')[-1]
    # sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    # if len(indices.shape) == 0:  # if all elements are zeros
    #     return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse.FloatTensor(indices, values, x.size(),device=x.device)
