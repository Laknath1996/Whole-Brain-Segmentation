%Author : Ashwin de Silva
%Last Updated : 2018 Mar 26

%Extract the 2D images from the .img MRI files 
%save them in the specified folder

close all;
clear all;

databaseDir = '/Users/ashwin/Research Work /MRI Brain Extraction/LPBA40/native_space_radio';
cd(databaseDir);
folder_contents = dir(databaseDir);
folder_contents = folder_contents(4:end);

%headerFileName = '/%s/%s.native_radio.mri.hdr';
%mask_imageFileNameZip = '/%s/%s.native_radio.brain.mask.img.gz';
train_imageFileName = '%s.native_radio.mri.img';
mask_imageFileName = '%s.native_radio.brain.mask.img'

train_set = zeros(numel(folder_contents),1,512,512);
mask_set =  zeros(numel(folder_contents),1,512,512);

for i = 1 : numel(folder_contents)
    train_imageFileName = '%s.native_radio.mri.img';
    mask_imageFileName = '%s.native_radio.brain.mask.img';
    subject_name = folder_contents(i).name;
    disp(sprintf('%s is extracting..',subject_name));
    cd(subject_name);
    train_imageFileName = sprintf(train_imageFileName,subject_name);
    mask_imageFileNam = sprintf(mask_imageFileName,subject_name);
    
    im = analyze75read(train_imageFileName);
    imageSlice = im(25,:,:);
    imageSlice = reshape(imageSlice,[256,256]);
    imageSlice = imresize(imageSlice,[512,512]);
    train_set(i,:,:,:) = imageSlice;
    
    im = analyze75read(mask_imageFileNam);
    imageSlice = im(25,:,:);
    imageSlice = reshape(imageSlice,[256,256]);
    imageSlice = imresize(imageSlice,[512,512]);
    mask_set(i,:,:,:) = imageSlice;
    
    cd ..
end


    