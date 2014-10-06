function run_mnist_times(dataset, filter_type, filter_count, times)

    %Change to mean more...
    N_train_possibilities = [30, 100, 200, 500, 1000];

    train_err = zeros(times, length(N_train_possibilities));
    test_err  = zeros(times, length(N_train_possibilities));
    
    dist_indices = zeros(times, 3);
    
    for i = 1:times
        [train_err(i,:), test_err(i,:), dist_indices(i,:)] = run_mnist(dataset, filter_type, filter_count);       
    end
    
    fprintf('Distance indices: ');
    fprintf('%g ', mean(dist_indices, 1));
    fprintf('\n');
    
    fprintf('Std: ');
    fprintf('%g ', std(dist_indices, 1));
    fprintf('\n');
    
    fprintf('Overall Accuracy:\n');
    fprintf('N\ttrain mean\ttrain std\ttest mean\ttest std\n');
    
    for j = 1:length(N_train_possibilities)
        fprintf('%d\t%g%%\t%g%%\t%g%%\t%g%%\n', N_train_possibilities(j), ...
            mean(train_err(:,j)), std(train_err(:,j)), mean(test_err(:,j)), std(test_err(:,j)));
    end
    
    fprintf('\n');
end
