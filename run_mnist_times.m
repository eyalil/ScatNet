function run_mnist_times(filter_type, filter_count, times)

    %Change to mean more...
    N_train_possibilities = [30, 100, 200, 500, 1000];

    train_err = zeros(times, length(N_train_possibilities));
    test_err  = zeros(times, length(N_train_possibilities));
    
    dist_indices_before = zeros(times, 10);
    dist_indices_after  = zeros(times, 10);
    
    for i = 1:times
        [train_err(i,:), test_err(i,:), dist_indices_before(i,:), dist_indices_after(i,:)] = run_mnist(filter_type, filter_count);       
    end
    
    fprintf('Distance indices before: ');
    fprintf('%g ', mean(dist_indices_before, 1));
    fprintf('\n');
    
    fprintf('Distance indices after: ');
    fprintf('%g ', mean(dist_indices_after, 1));
    fprintf('\n');
    
    
    fprintf('Overall Accuracy:\n');
    fprintf('N\ttrain mean\ttrain std\ttest mean\ttest std\n');
    
    for j = 1:length(N_train_possibilities)
        fprintf('%d\t%g%%\t%g%%\t%g%%\t%g%%\n', N_train_possibilities(j), ...
            mean(train_err(:,j)), std(train_err(:,j)), mean(test_err(:,j)), std(test_err(:,j)));
    end
    
    fprintf('\n');
end
