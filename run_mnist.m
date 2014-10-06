function [train_err, test_err, indices_out] = run_mnist(dataset, filter_type, filter_count)

    %% Meta - Params
    N_train_possibilities = [30, 100, 200, 500, 1000]; %[30, 100, 200, 500, 1000, 2000, 4000, 99999];
    train_err = zeros(1, length(N_train_possibilities));
    test_err = zeros(1, length(N_train_possibilities));
        
    %% Params
    N_train = 1000; %Number of samples to use, PER DIGIT (so overall 10N samples are used)
    N_test = 1000; %CHANGE THIS; Number of samples to use for test (generally we just use all of them)
    
    global PARALLELISM;
    PARALLELISM = 4;
    
    classifier = 'LinearSVM';
    
    filt_opt.filter_type = filter_type; %Options: 'EyalRandom', 'OrderedRandom', 'UniformRandom', 'morlet', 'Hebbian', 'PCA'

    hebbian_sparsity = 0.5;
    
    WHITENING = 0;
    
	if strcmp(filt_opt.filter_type, 'PCA')
        filt_opt.PCA_Type = 0; %0 - whole image; 1 - 8x8 patches
        
    elseif strcmp(filt_opt.filter_type, 'UniformRandom')
		filt_opt.UniformRandom_Count = filter_count;
        
	end

    %Define 'wavelet' transform
    filt_opt.J = 3 %Was: 3/5
    filt_opt.L = 8;

    scat_opt.M = 1; %Number of layers is set here
    scat_opt.oversampling = 0; %Was: 0
    
    %% Startup
    fprintf('Startup...\n');
    startup;
    tic;
        
    %% Load dataset Data
    if strcmp(dataset, 'MNIST')
        fprintf('Loading MNIST data...\n');
        [train, test] = LoadMNIST(N_train, N_test);
    elseif strcmp(dataset, 'NORB')
        fprintf('Loading NORB data...\n');
        [train, test] = LoadNORB(N_train, N_test);
        
    else
        fprintf('Error: Bad datset\n');
        
    end

    %% Preprocessing
    if strcmp(dataset, 'MNIST') %What to do with NORB?
        [train, test] = ApplyNormalizeImages(train, test); %Normalize images (should always help)   
    end
    if (WHITENING)
        [train, test] = ApplyWhitening(train, test, 0.1); %Whitening?
    end

    [distance_indices_before, class_centers] = CalcDistances(train);
    [fisher_before, S_MC_before] = CalcFisher(train);
    fprintf('S_MC = %g\n', S_MC_before);
    
    %% Create wavelet filters / transition matrix
    %If PCA, calc principle components now (Assume M=1)
    if strcmp(filt_opt.filter_type, 'PCA')
        filt_opt.PCA_Filters = CalcPCAfilters(train, filt_opt.PCA_Type, filter_count);
    end
    
    %Apply filters (unless we're Hebbian)
    if strcmp(filt_opt.filter_type, 'Hebbian')
        feature_gen = CalcHebbianFeatureGen(class_centers, hebbian_sparsity);
        
    else
        fprintf('Creating wavelets...\n');
        [Wop, ~] = wavelet_factory_2d([32, 32], filt_opt, scat_opt);

        %feature_gen = @(x)(sum(sum(format_scat(scat(x, Wop)),2),3));
        feature_gen = @(x) (FeatureGen(x, Wop));
    end
    
    %Ignore the feature gen and do nothing
    %feature_gen = @(x)(x);
    
    
    %% Calculate features
    fprintf('Calculating features...\n');
    
    [train_features test_features] = Eyal_CalculateFeatures(train, test, feature_gen);
    
    for i = 1:length(N_train_possibilities)
        
        N_train = N_train_possibilities(i);

        
        %% Prepare Database
        fprintf('Normalize features...\n');
        [ train_features_norm, test_features_norm ] = Eyal_NormalizeFeatures(train_features, test_features, N_train, classifier);
    
        %Now calculate distance (of normalized features)
        distance_indices_after = CalcDistances(train_features_norm);
        [fisher_after, S_MC_after] = CalcFisher(train_features_norm);
        fprintf('S_MC = %g\n', S_MC_after);
    
        fprintf('Preparing database...\n');
        [ db, train_set, test_set ] = Eyal_PrepareDatabase( train, test, train_features_norm, test_features_norm, N_train );



        %% Classify
        fprintf('Classifying...\n');


        
        if strcmp(classifier, 'PCA')
            %% PCA classifier
        
            %Get the best PCA classifier (allow dimensions of 1:80, do 5 iterations)
            %   (The function must only use the first 10N_train instances!)
            [error, best_dim, best_model] = mnist_pca_classifier(db, N_train, 1:80, 5, length(train_set));

            cur_train_err = min(mean(error)); %??? Check this - not sure if ok
            fprintf('Min dim was %d. Train error is %g.\n', best_dim, cur_train_err);

            % testing
            labels = affine_test(db, best_model, test_set);
            % compute the error
            cur_test_err = classif_err(labels, test_set, db.src); %Might need *100 here (?)


        
        
        elseif strcmp(classifier, 'SVM')
            %% SVM classifier
            
            gamma_options = 2 .^ (-15:2:3); %(-14:2:0);
            C_options = 2 .^ (-5:2:15); %(0:2:10); 

            [best_C, best_gamma, ~] = CV_SVM_Classifier(db, N_train, C_options, gamma_options, 5, 0, length(train_set));

            %Cheating - using the best C and gamma to save time. Note this should be removed
            %later on.
            %C = 2.8;
            %gamma = 0.0073;

            fprintf('Final C = %g; gamma = %g\n', best_C, best_gamma);

            cur_train_err = Do_SVM_Classifier( db, train_set, train_set, best_C, best_gamma, 0);
            cur_test_err = Do_SVM_Classifier( db, train_set, test_set, best_C, best_gamma, 0);
            
            
        elseif strcmp(classifier, 'LinearSVM')
            
            fprintf('N_train is %d...\n', N_train);

            C_options = 2 .^ (-5:2:15); %(0:2:10); 
            [best_C, ~, ~] = CV_SVM_Classifier(db, N_train, C_options, [], 5, 1, length(train_set));
            
            fprintf('Best C = %g\n', best_C);
            
            cur_train_err = Do_SVM_Classifier( db, train_set, train_set, best_C, [], 1);
            cur_test_err = Do_SVM_Classifier( db, train_set, test_set, best_C, [], 1);
            
        end


        %% Intermediate result

        %Print intermediate stuff
        fprintf('N_train = %d\n', N_train);
        fprintf('Final train error = %g%%\n', cur_train_err);
        fprintf('Final test error = %g%%\n', cur_test_err*100);
        
        train_err(i) = cur_train_err;
        test_err(i) = cur_test_err*100;
    end
    
    %% Epilog
    fprintf('\n');
    fprintf('Indices before: \n');
    fprintf('Eyal: %g\n', mean(distance_indices_before));
    fprintf('Fisher SC: %g\n', mean(fisher_before));
    fprintf('Fisher MC: %g\n', S_MC_before);
    
    fprintf('\n');
    fprintf('Indices after: \n');
    fprintf('Eyal: %g\n', mean(distance_indices_after));
    fprintf('Fisher SC: %g\n', mean(fisher_after));
    fprintf('Fisher MC: %g\n', S_MC_after);

    fprintf('\n');
    fprintf('Final Accuracy:\n');
    fprintf('N\t\ttrain\t\ttest\n');
    
    for i = 1:length(N_train_possibilities)
        fprintf('%d\t%g%%\t%g%%\n', N_train_possibilities(i), train_err(i), test_err(i));
    end
       
    fprintf('\n');
    
    indices_out = [mean(distance_indices_after), mean(fisher_after), S_MC_after];
    
    toc
end

function Y = FeatureGen(x, Wop)
    %y0 = format_scat(log_scat(renorm_scat(scat(x,Wop))));
    %y0 = format_scat(log_scat(scat(x,Wop)));
    y0 = format_scat(scat(x,Wop));
    
    %dim1 = size(y0, 1);
    %dim2 = size(y0, 2);
    %dim3 = size(y0, 3);
    %Y = reshape(y0, [dim1, dim2*dim3]);
    %Y = dct(Y);
    
    %Y = dct(sum(y0, 3));
    %Y = Y(1:ceil(end/2));
    
    %Y = sum(y0, 3);
    %Y = reshape(y0, [dim1*dim2*dim3 1]);
    
    %Y = sum(sum(y0,2),3);
    %Y = dct(Y);

    Y = y0; %Do nothing for now. Note that Y is 3D.
    
end
