function [train_err, test_err, distance_indices_before, distance_indices_after] = run_mnist(filter_type, filter_count)

    N_train_possibilities = [30, 100, 200, 500, 1000]; %[30, 100, 200, 500, 1000, 2000, 4000, 99999];
    train_err = zeros(1, length(N_train_possibilities));
    test_err = zeros(1, length(N_train_possibilities));

    %% Params
    N_train = 1000; %Number of samples to use, PER DIGIT (so overall 10N samples are used)
    N_test = 99999999; %Number of samples to use for test (generally we just use all of them)
    
    global PARALLELISM;
    PARALLELISM = 0;
    
    classifier = 'LinearSVM';
    
    filt_opt.filter_type = filter_type;
    
    %filt_opt.filter_type = 'EyalRandom';
    %filt_opt.filter_type = 'OrderedRandom';
    %filt_opt.filter_type = 'UniformRandom';
    %filt_opt.filter_type = 'morlet';
	%filt_opt.filter_type = 'Hebbain';
    
    hebbian_sparsity = 0.5;
    

    %Define 'wavelet' transform
    filt_opt.J = 3 %Was: 3/5
    filt_opt.L = 8;
    
	if strcmp(filt_opt.filter_type, 'UniformRandom')
		filt_opt.UniformRandom_Count = filter_count;
	end


    scat_opt.M = 1; %Number of layers is set here
    scat_opt.oversampling = 0; %Was: 0
    
        
    %% Startup
    fprintf('Startup...\n');
    startup;
    tic;
        
    %% Load MNIST Data
    fprintf('Loading MNIST data...\n');

    [train, test] = retrieve_mnist_data(N_train, N_test);

    %Normalize images (should always help)
    pix_sum = 0;
    pix_sq_sum = 0;
    pix_count = 0;
    
    for d = 1:10
        for i = 1:length(train{d})
            pix_sum = pix_sum + sum(sum(train{d}{i}));
            pix_sq_sum = pix_sq_sum + sum(sum(train{d}{i} .^ 2));
            pix_count = pix_count + 32*32;
        end
    end
    
    %pix_avg = pix_sum / pix_count;
    %pix_std = sqrt((pix_sq_sum ./ pix_count ) - (pix_sum ./ pix_count) .^ 2);
    
    train_length = 0;
    
    for d = 1:10
        for i = 1:length(train{d})
            %train{d}{i} = (train{d}{i} - pix_avg) ./ pix_std;
            
            train{d}{i} = (train{d}{i} / 255)*2 - 1;
            train_length = train_length + 1;
        end
        for i = 1:length(test{d})
            %test{d}{i} = (test{d}{i} - pix_avg) ./ pix_std;
            
            test{d}{i}  = (test{d}{i}  / 255)*2 - 1;
        end
    end

    
    [distance_indices_before, class_centers] = CalcDistances(train);
    class_count = size(class_centers, 2);
    
    %% Create wavelet filters / transition matrix
    if strcmp(filt_opt.filter_type, 'Hebbian')
        
        %Create transition matrix
        W = zeros(32*32, 4*4*25);
        pattern = rand([class_count, 4*4*25]);
               
        %TEMP
        %for t = 1:8*4*25
        %    r = randi([1, class_count*4*4*25]);
        %    pattern(r) = 0;
        %end
        
        for d = 1:class_count
            S_m = class_centers(:, d);
            
            for j = 1:4*4*25
                W(:,j) = W(:,j) + (S_m - 0.5) * (pattern(d,j) - hebbian_sparsity);
            end
        end
        
        W = W / class_count;

        %Feature generating function
        feature_gen = @(x) reshape(W'*(x(:)), [4*4*25 1 1]);

    else
        
        fprintf('Creating wavelets...\n');
        Wop = wavelet_factory_2d([32, 32], filt_opt, scat_opt);

        %Feature generating function
        
        %feature_gen = @(x)(sum(sum(format_scat(scat(x, Wop)),2),3));
        feature_gen = @(x) (FeatureGen(x, Wop));
    end
    
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
    
        fprintf('Preparing database...\n');
        [ db, train_set, test_set ] = Eyal_PrepareDatabase( train, test, train_features_norm, test_features_norm, N_train );



        %% Classify
        fprintf('Classifying...\n');


        
        if strcmp(classifier, 'PCA')
            %% PCA classifier
        
            %Get the best PCA classifier (allow dimensions of 1:80, do 5 iterations)
            %   (The function must only use the first 10N_train instances!)
            [error, best_dim, best_model] = mnist_pca_classifier(db, N_train, 1:80, 5);

            cur_train_err = min(mean(error)); %??? Check this - not sure if ok
            fprintf('Min dim was %d. Train error is %g.\n', best_dim, cur_train_err);

            % testing
            labels = affine_test(db, best_model, test_set);
            % compute the error
            cur_test_err = classif_err(labels, test_set, db.src); %Might need *100 here (?)


        
        
        elseif strcmp(classifier, 'SVM')
            %% SVM classifier
            
            gamma_options = 2 .^ (-14:2:0);
            C_options = 2 .^ (0:2:10); 

            [best_C, best_gamma, ~] = CV_SVM_Classifier(db, N_train, C_options, gamma_options, 5, 0);

            %Cheating - using the best C and gamma to save time. Note this should be removed
            %later on.
            %C = 2.8;
            %gamma = 0.0073;

            fprintf('Final C = %g; gamma = %g\n', best_C, best_gamma);

            cur_train_err = Do_SVM_Classifier( db, train_set, train_set, best_C, best_gamma, 0);
            cur_test_err = Do_SVM_Classifier( db, train_set, test_set, best_C, best_gamma, 0);
            
            
        elseif strcmp(classifier, 'LinearSVM')
            
		fprintf('N_train is %d...\n', N_train);

            C_options = 2 .^ (0:2:10);
            [best_C, ~, ~] = CV_SVM_Classifier(db, N_train, C_options, [], 5, 1);
            
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
    fprintf('Distance indices before: ');
    fprintf('%g ', distance_indices_before);
    fprintf('\n');
    
    fprintf('Distance indices after: ');
    fprintf('%g ', distance_indices_after);
    fprintf('\n');
    
    fprintf('Final Accuracy:\n');
    fprintf('N\t\terr\n');
    
    for i = 1:length(N_train_possibilities)
        fprintf('%d\t%g%%\t%g%%\n', N_train_possibilities(i), train_err(i), test_err(i));
    end
       
    fprintf('\n');
    
    
    
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
