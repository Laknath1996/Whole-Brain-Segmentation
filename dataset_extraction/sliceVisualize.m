%Author : Ashwin de Silva
%Last Updated : 2018 Mar 26

%Visualize the image slice from the test_set and mask_set volumes

function sliceVisualize(set,subject_ID)
    im = set(subject_ID,:,:);
    im = reshape(im,[512,512]);
    
    figure;
    imshow(mat2gray(im));
    title(sprintf('Image Slice of Subject %d',subject_ID));
    
end

