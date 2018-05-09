#Author : Ashwin de Silva
#Last Updated : 2018 May 3

from utilities import *
from BRAIN_unet_utilities import BRAIN_unet
import scipy.io as sio
import numpy as np
    
#test_images = load_train_dataset()

dict = sio.loadmat('test_images.mat')
test_images = dict['test_images']
test_images = np.reshape(test_images,(21,512,512,1))

model = BRAIN_unet((512,512,1))
model.load_weights('unet.hdf5')

mask_images = model.predict(test_images,batch_size = 1,verbose = 1)
dict = {}
dict['mask_images'] = mask_images

sio.savemat('mask_images.mat',dict)
