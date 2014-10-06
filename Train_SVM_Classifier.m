function [error, best_C, best_gamma] = Train_SVM_Classifier(db, N, C_options, gamma_options, nb_split)

    error
    error
    error
    This is old, do not run this file (for the very least, 10 should not be used...)

    n_images = 10*N; %Use only the first 10N instances, the rest are part of the real 'test_set'

    rng(1); % set the random split generator of matlab to have reproducible results  

    %Do cross validation over C and gamma
    for i_split = 1:nb_split
        for i_train = 1:(length(C_options)*length(gamma_options))
            i_C_idx = idivide(i_train, length(gamma_options));
            i_gamma_idx = mod(i_train, length(gamma_options));
            
            cur_C     = C_options(i_C_idx);
            cur_gamma = gamma_options(i_gamma_idx);
            
            [train_set, test_set] = create_partition(db.src, 0.5);
            
            %Do not allow images other than the first n_images
            train_set = train_set(train_set <= n_images);
            test_set  = test_set(test_set <= n_images);
            
            %Train SVM
            error(i_split, i_train) = Do_SVM_Classifier( db, train_set, test_set, cur_C, cur_gamma );
            fprintf('split %3d nb train %g %g accuracy %.2f \n', ...
                i_split, cur_C, cur_gamma, 100*error(i_split, i_train));
        end
    end

    %% averaged performance
    perf = 100*(1-mean(error));
    perf_std = 100*std(error);

    for i_train = 1:numel(grid_train)
        fprintf('%2d training : %.2f += %.2f \n', ...
            grid_train(i_train), perf(i_train), perf_std(i_train));
    end

    
    %% best model
    best_perf = 0;
    best_i = -1;
    
    for i_train = 1:numel(grid_train)
        if perf(i_train) > best_perf
            best_perf = perf(i_train);
            best_i = i_train;
        end
    end
    
    best_i_C_idx = idivide(best_i, length(gamma_options));
    best_i_gamma_idx = mod(best_i, length(gamma_options));

    best_C     = C_options(best_i_C_idx);
    best_gamma = gamma_options(best_i_gamma_idx);
    
end