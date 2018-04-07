#Author : Ashwin de Silva
#Last Updated : 2018 Mar 26

import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    data = sio.loadmat('dataset512N.mat')

    X_orig = data['train_set']
    Y_orig = data['mask_set']
    
    X_orig = np.reshape(X_orig,[40,512,512,1])
    Y_orig = np.reshape(Y_orig,[40,512,512,1])
    
    X_orig = X_orig.astype('float32')
    Y_orig = Y_orig.astype('float32')
    
    
    return X_orig, Y_orig