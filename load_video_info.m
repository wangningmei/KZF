%%读取图片序列的格式为，.img图片序列以及groundtruth标记数据
%   加载视频序列信息
%base_path  = './sequence';
%video      = 'Bolt';

%video_path = [base_path '/' video];

function [seq, ground_truth] = load_video_info(video_path)

ground_truth = dlmread([video_path '/groundtruth_rect.txt']);

seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

img_path = [video_path '/img/'];

img_files = dir(fullfile(img_path, '*.jpg'));
img_files = {img_files.name};

seq.s_frames = cellstr(img_files);

end

