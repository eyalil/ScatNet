function [best_C, best_gamma, best_perf] = CV_SVM_Classifier(db, N, C_options, gamma_options, nb_split, is_linear)

    n_images = 10*N; %Use only the first 10N instances, the rest are part of the real 'test_set'

    rng(1); % set the random split generator of matlab to have reproducible results  

    if is_linear
        assert(isempty(gamma_options));
        gamma_options = [ 666 ]; %Just some fake gamma
    else
        assert(length(gamma_options) >= 1);
    end
    
    %Do cross validation over C and gamma
    global PARALLELISM;
    matlabpool('open', PARALLELISM);
    
    error = zeros(nb_split, length(C_options)*length(gamma_options));
    %parfor i_split = 1:nb_split
    for i_split = 1:nb_split
        parfor i_train = 1:(length(C_options)*length(gamma_options))
        %for i_train = 1:(length(C_options)*length(gamma_options))
            [cur_C cur_gamma] = idx_to_C_gamma(i_train, C_options, gamma_options);
            
            [train_set, test_set] = create_partition(db.src, 0.5);
            
            %Do not allow images other than the first n_images
            train_set = train_set(train_set <= n_images);
            test_set  = test_set(test_set <= n_images);
            
            fprintf('Eyal: Train set size is %d; Test set size is %d\n', length(train_set), length(test_set));
            
            %Train SVM
            error(i_split, i_train) = Do_SVM_Classifier( db, train_set, test_set, cur_C, cur_gamma, is_linear );
            %fprintf('split %3d nb train %g %g accuracy %.2f \n', ...
            %    i_split, cur_C, cur_gamma, 100*error(i_split, i_train));
        end
    end
    
    matlabpool('close');

    %% averaged performance
    perf = 100*(1-mean(error));
    perf_std = 100*std(error);

    for i_train = 1:(length(C_options)*length(gamma_options))
       [cur_C cur_gamma] = idx_to_C_gamma(i_train, C_options, gamma_options);
            
        fprintf('%g %g training : %.2f += %.2f \n', ...
            cur_C, cur_gamma, perf(i_train), perf_std(i_train));
    end

    
    %% best model
    best_perf = 0;
    best_i = -1;
    
    for i_train = 1:(length(C_options)*length(gamma_options))
        if perf(i_train) > best_perf
            best_perf = perf(i_train);
            best_i = i_train;
        end
    end
    
    [best_C best_gamma] = idx_to_C_gamma(best_i, C_options, gamma_options);
    
end

function [C gamma] = idx_to_C_gamma(idx, C_options, gamma_options)
    i_C_idx     = fix((idx-1) / length(gamma_options)) + 1;
    i_gamma_idx = mod((idx-1), length(gamma_options)) + 1;

    C     = C_options(i_C_idx);
    gamma = gamma_options(i_gamma_idx);
end
