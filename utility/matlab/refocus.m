%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% REFOCUS on LF using PHASE-SHIFT ALGORITHM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Add path to DepthEstimation(Ch4)
% download source code from https://sites.google.com/site/hgjeoncv/home/depthfromlf_cvpr15
addpath('<PATH_TO>/DepthEstimation(Ch4)')

load('../../caldata/lfcalib/IntParamLF.mat');
K2              = IntParamLF(2);  % milimeters
fxy             = IntParamLF(3:4);
flens           = max(fxy);
fsubaperture    = 521.4052;       % pixel
baseline        = K2/flens*1e-3;  % meters

SS              = 10;             % stack size
depth_range     = [0.5 7];
disparity_range = (baseline*fsubaperture ./ depth_range);
disparities     = linspace(disparity_range(1), disparity_range(2), SS);

% path to output focalstack folder
outfolder       = './';

% load light-field
load 'LF_0001.mat'

LF = im2double(LF);
for inx=1:length(disparities)
    lfsize=[size(LF,3) size(LF,4)];
    uvcenter=([size(LF,1),size(LF,2)]+1)/2;
    image=zeros([lfsize 3],'double');
    for u=1:size(LF,1)
        for v=1:size(LF,2)
            shift= (uvcenter'-[u;v]) * disparities(inx);
            shifted = fn_SubpixelShift( fft2(squeeze(LF(u,v,:,:,:))), shift, lfsize(1), lfsize(2),1);
            image = image + shifted;
        end
    end
    image = image ./ prod([size(LF,1) size(LF,2)]);
    imwrite(uint8(image*255), [outfolder '/' sprintf('%02d', inx) '.png']);
end
