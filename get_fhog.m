%使用piotrs toolbox提取fhog特征 

function [ feature_image ] = get_fhog( im, fparam, gparam )

if ~isfield(fparam, 'nOrients')
    fparam.nOrients = 9;  %方向划分为九个区间，方向不敏感
end
[im_height, im_width, num_im_chan, num_images] = size(im);
feature_image = zeros(floor(im_height/gparam.cell_size), floor(im_width/gparam.cell_size), fparam.nDim, num_images, 'single');
for k = 1:num_images
    hog_image = fhog(single(im(:,:,:,k)), gparam.cell_size, fparam.nOrients);
    %the last dimension is all 0 so we can discard it
    feature_image(:,:,:,k) = hog_image(:,:,1:end-1);
end
end