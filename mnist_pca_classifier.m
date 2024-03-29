function [error, best_dim, best_model] = mnist_pca_classifier(db, N, grid_train, nb_split, num_of_train_images)

    %rng(1); % set the random split generator of matlab to 
    % have reproducible results  

    for i_split = 1:nb_split
        for i_train = 1:numel(grid_train)
            cur_dim = grid_train(i_train);
            [train_set, test_set] = create_partition(db.src, 0.5);
            
            %Do not allow images other than the first n_images
            train_set = train_set(train_set <= num_of_train_images);
            test_set  = test_set(test_set <= num_of_train_images);
            
            train_opt.dim = cur_dim;
            model = affine_train(db, train_set, train_opt);
            labels = affine_test(db, model, test_set);
            error(i_split, i_train) = classif_err(labels, test_set, db.src);
            fprintf('split %3d nb train %2d accuracy %.2f \n', ...
                i_split, cur_dim, 100*(1-error(i_split, i_train)));
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
    
    best_dim = grid_train(best_i);
    train_opt.dim = best_dim;
    train_set = 1:num_of_train_images; %All real 'train images'
    best_model = affine_train(db, train_set, train_opt);
    
end
