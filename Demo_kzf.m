

clear;clc;
close all;
% Load video information


base_path  = './sequence';
video      = 'Walking';

video_path = [base_path '/' video];

OPs = zeros(numel(video),1);
FPSs = zeros(numel(video),1);
OPs_OTB50 = [];
FPSs_OTB50 = [];

%for vid = 1:numel(video)
    close all;
    video_path = [base_path '/' video];
    [seq, ground_truth] = load_video_info(video_path);
    seq.VidName = video;
    st_frame = 1;
    en_frame = seq.len;
   % if (strcmp(videos{vid}, 'David'))
   %     st_frame = 300;
   %     en_frame = 770;
   % elseif (strcmp(videos{vid}, 'Football1'))
   %     st_frame = 1;
   %     en_frame = 74;
   % elseif (strcmp(videos{vid}, 'Freeman3'))
   %     st_frame = 1;
   %     en_frame = 460;
   % elseif (strcmp(videos{vid}, 'Freeman4'))
   %     st_frame = 1;
   %     en_frame = 283;
   % end
    seq.st_frame = st_frame;
    seq.en_frame = en_frame;     %设置起始帧
    gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)];
    
    % Run BACF- main function
    %learning_rate = 0.013;  %   you can use different learning rate for different benchmarks.
    
    results = runtracker_KZF(seq, video_path);
   
