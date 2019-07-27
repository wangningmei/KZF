%  KZF tracker 主函数.

function [results] = trackerMain(params)

% parameters
padding = params.padding;
sigma = params.sigma;
lambda = params.lambda;
compression_learning_rate = params.compression_learning_rate;
non_compressed_features = params.non_compressed_features;
compressed_features = params.compressed_features;
num_compressed_dim = params.num_compressed_dim;
pos_cn = floor(params.init_pos);
target_sz_cn = floor(params.wsize);


output_sigma_factor = params.output_sigma_factor;

% load the normalized Color Name matrix
temp = load('w2crs');
w2c = temp.w2crs;

use_dimensionality_reduction = ~isempty(compressed_features);

% window size, taking padding into account
sz_cn = floor(target_sz_cn * (1 + padding));

% desired output (gaussian shaped), bandwidth proportional to target size
output_sigma = sqrt(prod(target_sz_cn)) * output_sigma_factor;
[rs, cs] = ndgrid((1:sz_cn(1)) - floor(sz_cn(1)/2), (1:sz_cn(2)) - floor(sz_cn(2)/2));
y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
yf_cn = single(fft2(y));

cos_window_cn = single(hann(sz_cn(1)) * hann(sz_cn(2))');

% initialize the projection matrix
projection_matrix = [];

%   局部参数设置
search_area_scale   = params.search_area_scale;
learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
nScales             = params.number_of_scales;
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response;

features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);
target_sz_cn= floor(params.wsize);

lambda_tp   = params.lambda_tp;
lambda_cn   = params.lambda_cn;

visualization  = params.visualization;
num_frames     = params.no_fram;
init_target_sz = target_sz;

%set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;
search_area = prod(init_target_sz / featureRatio * search_area_scale);

% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end

global_feat_params = params.t_global;

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
use_sz = floor(sz/featureRatio);

% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% construct cosine window
cos_window = single(hann(use_sz(1))*hann(use_sz(2))');

% 计算特征维数
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
end
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% compute feature dimensionality
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;

% allocate memory for multi-scale tracking
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8');
small_filter_sz = floor(base_target_sz/featureRatio);

%第一帧位置使用标记的位置，初始化tracker
loop_frame = 1;
for frame = 1:numel(s_frames)
    %load image
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    
    %计时开始
    tic();
    
    %BACF求解pos与target_sz
    %从第二帧开始，估计目标位置和尺度
    %copy from bacf
    if frame > 1
        for scale_ind = 1:nScales
            multires_pixel_template(:,:,:,scale_ind) = ...
                get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
        end
        xtf = fft2(bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params),cos_window));
        responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
        
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        if interpolate_response == 2
            % use dynamic interp size
            interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
        end
        responsef_padded = resizeDFT2(responsef, interp_sz);
        
        % 空间域响应
        response = ifft2(responsef_padded, 'symmetric');
        
        % 寻找最大峰值
        if interpolate_response == 3
            error('Invalid parameter value for interpolate_response');
        elseif interpolate_response == 4
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);
        else
            [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
        end
        % calculate translation
        switch interpolate_response
            case 0
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 1
                translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
            case 2
                translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
            case 3
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 4
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
        end
        
        % 设置尺度
        currentScaleFactor = currentScaleFactor * scaleFactors(sind);

        % 选取适中的尺度（不至于太小，也不至于太大
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % update position
        old_pos = pos;
        pos = pos + translation_vec;
    end
    
    % 提取训练样本图像区域
    pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
    
    % 提取特征
    xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window));
    
    if (frame == 1)
        model_xf = xf;
    else
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
    end
    
    g_f = single(zeros(size(xf)));
    h_f = g_f;
    l_f = g_f;
    mu    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    
    T = prod(use_sz);
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    params.admm_iterations = 2;

    %   ADMM 迭代优化
    while (i <= params.admm_iterations)
        %   分别求解俩个子问题g，h
        B = S_xx + (T * mu);
        S_lx = sum(conj(model_xf) .* l_f, 3);
        S_hx = sum(conj(model_xf) .* h_f, 3);
        g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf)) - ((1/mu) * l_f) + h_f) - ...
            bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);
        
        %   子问题h
        h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
        [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
        t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
        t(sx,sy,:) = h;
        h_f = fft2(t);
        
        %   update L（拉格朗日向量
        l_f = l_f + (mu * (g_f - h_f));
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
    end
    
    %floor向负无穷方向取整
    target_sz = floor(base_target_sz * currentScaleFactor);
    

    %CN求解目标位置
    if frame > 1
        % compute the compressed learnt appearance
        zp = feature_projection(z_npca, z_pca, projection_matrix, cos_window_cn);
        
        % extract the feature map of the local image patch
        [xo_npca, xo_pca] = get_subwindow(im, pos_cn, sz_cn, non_compressed_features, compressed_features, w2c);
        
        % do the dimensionality reduction and windowing
        x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window_cn);
        
        % calculate the response of the classifier
        kf = fft2(dense_gauss_kernel(sigma, x, zp));
        response = real(ifft2(alphaf_num .* kf ./ alphaf_den));
        
        % target location is at the maximum response
        [row, col] = find(response == max(response(:)), 1);
        pos_cn = pos_cn - floor(sz_cn/2) + [row, col];
    end
    
    % extract the feature map of the local image patch to train the classifer
    [xo_npca, xo_pca] = get_subwindow(im, pos_cn, sz_cn, non_compressed_features, compressed_features, w2c);
    
    if frame == 1
        % initialize the appearance
        z_npca = xo_npca;
        z_pca = xo_pca;
        
        % set number of compressed dimensions to maximum if too many
        num_compressed_dim = min(num_compressed_dim, size(xo_pca, 2));
    else
        % update the appearance
        z_npca = (1 - learning_rate) * z_npca + learning_rate * xo_npca;
        z_pca = (1 - learning_rate) * z_pca + learning_rate * xo_pca;
    end
    
    % if dimensionality reduction is used: update the projection matrix
    if use_dimensionality_reduction
        % compute the mean appearance
        data_mean = mean(z_pca, 1);
        
        % substract the mean from the appearance to get the data matrix
        data_matrix = bsxfun(@minus, z_pca, data_mean);
        
        % calculate the covariance matrix
        cov_matrix = 1/(prod(sz_cn) - 1) * (data_matrix' * data_matrix);
        
        % calculate the principal components (pca_basis) and corresponding variances
        if frame == 1
            [pca_basis, pca_variances, ~] = svd(cov_matrix);
        else
            [pca_basis, pca_variances, ~] = svd((1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * cov_matrix);
        end
        
        % calculate the projection matrix as the first principal
        % components and extract their corresponding variances
        projection_matrix = pca_basis(:, 1:num_compressed_dim);
        projection_variances = pca_variances(1:num_compressed_dim, 1:num_compressed_dim);
        
        if frame == 1
            % initialize the old covariance matrix using the computed
            % projection matrix and variances
            old_cov_matrix = projection_matrix * projection_variances * projection_matrix';
        else
            % update the old covariance matrix using the computed
            % projection matrix and variances
            old_cov_matrix = (1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * (projection_matrix * projection_variances * projection_matrix');
        end
    end
    
    % project the features of the new appearance example using the new
    % projection matrix
    x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window_cn);
    
    % calculate the new classifier coefficients
    kf = fft2(dense_gauss_kernel(sigma, x));
    new_alphaf_num = yf_cn .* kf;
    new_alphaf_den = kf .* (kf + lambda);
    
    if frame == 1
        % first frame, train with a single image
        alphaf_num = new_alphaf_num;
        alphaf_den = new_alphaf_den;
    else
        % subsequent frames, update the model
        alphaf_num = (1 - learning_rate) * alphaf_num + learning_rate * new_alphaf_num;
        alphaf_den = (1 - learning_rate) * alphaf_den + learning_rate * new_alphaf_den;
    end
    

    %开始进行融合
    %pos[2]表示x轴，pos[1]表示y轴，后一项表示目标尺度（target_sz[1]表示长（宽度），target_sc[2]表示宽（高度）
    %得到最终的位置pos([2,1])，以及最终的尺度target_sc([2,1])
    pos([2,1])= lambda_tp*pos([2,1]) + lambda_cn*pos_cn([2,1]);
    pos_cn([2,1]) = lambda_tp*pos([2,1]) + lambda_cn*pos_cn([2,1]);
    target_sz= lambda_tp*target_sz([2,1]) + lambda_cn*target_sz_cn([2,1]);
    target_sz_cn= lambda_tp*target_sz([2,1]) + lambda_cn*target_sz_cn([2,1]);
    rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    %toc表示计时结束
    time = time + toc();
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([1,2])/2, target_sz([1,2])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;

            %画矩形框，rect_position_vis中包含四个参数，即（x,y,w,h）
            %从左下角，宽为w，高为h
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            sc_ind = floor((nScales - 1)/2) + 1;
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
          %  resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind)));colormap hsv;
          %  alpha(resp_handle, 0.2);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(20, 35, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [0 1 1], 'fontsize', 12);
                      
            hold off;
        end
        drawnow
    end
    loop_frame = loop_frame + 1;
end



%   save resutls
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
