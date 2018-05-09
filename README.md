# MRI-Brain-Segmentation
The objective of this project is to use a deep learning approach to segment the brain from the MRI images. A U-Net based architecture was used for the segmentation task. A dataset of 40 512 by 512 MRI images were used and using data augmentation the dataset was expanded up to 320 images. The augmentation procedure was to take one of the above mentioned 40 images, rotate it in 90, 180, 270 degrees and do the same for a flipped version of the same image. Hence, from one image it was possible to generate 8 images altogether. 240 images were used for the training, 60 images were used for the vaidation and 20 images were used as the test set.

The following image is one of the images used to train the network. 

![alt text](https://github.com/Laknath1996/MRI-Brain-Segmentation/blob/master/BRAIN_unet/Original.jpg?raw=true)

The pixel wise labelled mask for the above image is state below. 

![alt text](https://github.com/Laknath1996/MRI-Brain-Segmentation/blob/master/BRAIN_unet/mask.jpg?raw=true)

The U-Net architecture that was used is stated below. 
U-Net paper can be found here : https://arxiv.org/abs/1505.04597

![alt text](https://github.com/Laknath1996/MRI-Brain-Segmentation/blob/master/BRAIN_unet/BRAINunet_Archicture.png?raw=true)

The following hyperparameters were used. 

mini-batch size  : 4,
number of epochs : 40,
learning rate    : 10^(-4),
optimizer        : Adam

The model was trained with the use of a GPU.

A test image with its predicted brain mask is stated below. 

![alt text](https://raw.githubusercontent.com/Laknath1996/MRI-Brain-Segmentation/master/BRAIN_unet/img_with_mask%2021.bmp)







