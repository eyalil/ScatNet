addpath './scatnet-0.2/';
addpath_scatnet;

addpath './libsvm-compact-0.1/matlab/';
%addpath './liblinear-1.94/matlab/';

global mpath;
mpath = 'C:\Dataset\';
if ~exist(mpath, 'file')
    mpath = 'D:\Dataset\';
    if ~exist(mpath, 'file')
        mpath = '~/Dataset/';
    end
end


