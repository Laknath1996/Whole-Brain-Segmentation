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
    data = sio.loadmat('dataset-aug.mat')

    X_orig = data['train_images_aug']
    Y_orig = data['mask_images_aug']

    X_orig = np.reshape(X_orig,[320,512,512,1])
    Y_orig = np.reshape(Y_orig,[320,512,512,1])

    X_orig = X_orig.astype('float32')
    Y_orig = Y_orig.astype('float32')


    return X_orig, Y_orig

def load_train_dataset():
    data = sio.loadmat('dataset-aug.mat')

    X_test = data['train_images_aug']

    X_test = np.reshape(X_test,[320,512,512,1])

    X_test = X_test.astype('float32')

    X_test = X_test[300:320,:,:,:]

    X_test = X_test/255 #Normalize the test images

    return X_test

