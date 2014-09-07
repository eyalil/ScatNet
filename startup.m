addpath './scatnet-0.2/';
addpath_scatnet;

addpath '.\libsvm-compact-0.1\matlab\';

global mpath;
mpath = '~/Dataset/';
if ~exist(mpath, 'file')
	mpath = 'C:\Dataset\';
	if ~exist(mpath, 'file')
		mpath = 'D:\Dataset\';
	end
end
