%Author : Ashwin de Silva
%Last Updated : 2018 Apr 4

%This code block performs the data augmentation for the given raw data set.
%each image is taken, and rotated in 0,90,180 and 360 and the same is
%performed for the flipped image as well.

function augment

%open the matfile
uiopen;

train_images = train_set;
mask_images = mask_set;

%define the augmented data sets

train_images_aug = zeros(320,1,512,512);
mask_images_aug = zeros(320,1,512,512);

for i = 1 : 40 
    im = reshape(train_images(i,:,:,:),[512,512]);
    train_images_aug(i*8,1,:,:) = im;
    train_images_aug(i*8+1,1,:,:) = imrotate(im,90);
    train_images_aug(i*8+2,1,:,:) = imrotate(im,180);
    train_images_aug(i*8+3,1,:,:) = imrotate(im,270);
    train_images_aug(i*8+4,1,:,:) = flip(im,2);
    train_images_aug(i*8+5,1,:,:) = imrotate(im,90);
    train_images_aug(i*8+6,1,:,:) = imrotate(im,180);
    train_images_aug(i*8+7,1,:,:) = imrotate(im,270);
    
    mk = reshape(mask_images(i,:,:,:),[512,512]);
    mask_images_aug(i*8,1,:,:) = mk;
    mask_images_aug(i*8+1,1,:,:) = imrotate(mk,90);
    mask_images_aug(i*8+2,1,:,:) = imrotate(mk,180);
    mask_images_aug(i*8+3,1,:,:) = imrotate(mk,270);
    mask_images_aug(i*8+4,1,:,:) = flip(mk,2);
    mask_images_aug(i*8+5,1,:,:) = imrotate(mk,90);
    mask_images_aug(i*8+6,1,:,:) = imrotate(mk,180);
    mask_images_aug(i*8+7,1,:,:) = imrotate(mk,270);
end

train_images_aug = train_images_aug([8:327],:,:,:);
mask_images_aug = mask_images_aug([8:327],:,:,:);

save('dataset-aug.mat','train_images_aug','mask_images_aug');
    


