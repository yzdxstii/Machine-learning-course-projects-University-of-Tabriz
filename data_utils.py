from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError('invalid python version: {}'.format(version))

def load_CIFAR_batch(filename):
    '''load dingle batch of cifar'''
    with open(filename , 'rb') as f:
        datadict = load_pickle(f)
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000 , 3 , 32 , 32).transpose(0,2,3,1).astype('float')
        y = np.array(y)
        return x, y

def load_CIFAR10(ROOT):
    '''load all of cifar'''
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT , 'data_batch_%d' %(b, ))
        x , y = load_CIFAR_batch(f)
        xs.append(x)
        ys.append(y)
    xtr = np.concatenate(xs)
    ytr = np.concatenate(ys)
    del x , y
    xte , yte = load_CIFAR_batch(os.path.join(ROOT , 'test_batch'))
    return xtr , ytr , xte , yte
    

def get_CIFAR10_data(cifar10_dir , num_training= 49000 , num_validation=1000 , num_test= 1000 , subtract_mean= True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    x_train , y_train , x_test , y_test = load_CIFAR10(cifar10_dir)

    #subsample the data
    mask = list(range(num_training , num_training + num_validation))
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    x_test = x_test[mask]
    y_test = y_test[mask]

    #nurmalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(x_train , axis= 0)
        x_train -= mean_image
        x_val -= mean_image
        x_test -= mean_image

    #transpose so that channels come first
    x_train = x_train.transpose(0 , 3 , 1 , 2).copy()
    x_val = x_val.transpose(0 , 3 , 1 , 2).copy()
    x_test = x_test.traspose(0 , 3 , 1 , 2).copy()

    #package data into a dictionary
    return {
        'x_train': x_train , 'y_train': y_train,
        'x_val': x_val , 'y_val': y_test,
        'x_test': x_test , 'y_test': y_test,
    }