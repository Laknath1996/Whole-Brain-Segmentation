# MRI-Brain-Segmentation
The objective of this project is to use a deep learning approach to segment the brain from the MRI images. A U-Net based architecture was used for the segmentation task. A dataset of 40 512 by 512 MRI images were used and using data augmentation the dataset was expanded up to 320 images. The augmentation procedure was to take one of the above mentioned 40 images, rotate it in 90, 180, 270 degrees and do the same for a flipped version of the same image. Hence, from one image it was possible to generate 8 images altogether. 240 images were used for the training, 60 images were used for the vaidation and 20 images were used as the test set.

The following image is one of the images used to train the network. 


