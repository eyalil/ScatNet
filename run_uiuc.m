startup;
src = uiuc_src('C:\Dataset\UIUC');

%Eyal: do random
%filt_opt.filter_type = 'EyalRandom';
filt_opt.filter_type = 'morlet';
%Random: 0.51 / 0.5680 / 0.5720 / 0.5440
%Non-Random:  0.3920 / 0.4180 / 0.3940 / 0.4160

%Define 'wavelet' transform
filt_opt.J = 5;
scat_opt.M = 1; %One layer for now
scat_opt.oversampling = 0;
Wop = wavelet_factory_2d([480, 640], filt_opt, scat_opt);
features{1} = @(x)(sum(sum(format_scat(scat(x,Wop)),2),3));

%% Prepare database
%matlabpool('open', 2);

options.parallel = 0;
db = prepare_database(src, features, options);

%matlabpool('close');

%% Classify
% proportion of training example
prop = 0.5;
% split between training and testing
[train_set, test_set] = create_partition(src, prop);
% dimension of the affine pca classifier
train_opt.dim = 20;
% training
model = affine_train(db, train_set, train_opt);
% testing
labels = affine_test(db, model, test_set);
% compute the error
error = classif_err(labels, test_set, src)
