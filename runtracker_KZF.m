%   运行BACF tracker  视频序列为"sequence".

function results = runtracker_KZF(seq, video_path)


%   HOG特征维数
hog_params.nDim   = 31;
params.video_path = video_path;
grayscale_params.colorspace='gray';
grayscale_params.nDim = 1;

%   全局特征参数
params.t_features = {
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...
    struct('getFeature',@get_fhog,'fparams',hog_params),...
};
params.t_global.cell_size = 4;                  % 特征单元（cell）尺寸
params.t_global.cell_selection_thresh = 0.75^2; % 在低分辨率情况下减少cell尺寸的门限Thelta

params.padding = 1.0;         			   % extra area surrounding the target
params.output_sigma_factor = 1/16;		   % spatial bandwidth (proportional to target)
params.sigma = 0.2;         			   % gaussian kernel bandwidth
params.lambda = 1e-2;					   % regularization (denoted "lambda" in the paper)
params.learning_rate = 0.075;			   % learning rate for appearance model update scheme (denoted "gamma" in the paper)
params.compression_learning_rate = 0.15;   % learning rate for the adaptive dimensionality reduction (denoted "mu" in the paper)
params.non_compressed_features = {'gray'}; % features that are not compressed, a cell with strings (possible choices: 'gray', 'cn')
params.compressed_features = {'cn'};       % features that are compressed, a cell with strings (possible choices: 'gray', 'cn')
params.num_compressed_dim = 2;             % the dimensionality of the compressed features

params.alph = 0.75;
params.lambda_tp = params.alph;
params.lambda_cn = 1-params.alph;
%   搜索区域  扩展背景参数
params.search_area_shape = 'square';    
params.search_area_scale = 5;           % 训练检测区域与目标大小成比例
params.filter_max_area   = 50^2;        % 特征网格单元格中训练检测区域的大小

%params.learning_rate       = 0.013;        % 学习率
%params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.interpolate_response  = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations     = 50;           % 最大化检测分数的牛顿迭代次数
%   Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.01;

%   size, position, frames initialization
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram  = seq.en_frame - seq.st_frame + 1;
params.seq_st_frame = seq.st_frame;
params.seq_en_frame = seq.en_frame;

%   ADMM 参数、lambd 参数设置
params.admm_iterations = 2;
params.admm_lambda = 0.01;

%   Debug and visualization
params.visualization = 1;


%   主函数 参见 trackerMain.m
results = trackerMain(params);

