function [train test] = LoadMNIST(N_train, N_test)

    %Load everything
    [train_temp, test_temp] = retrieve_mnist_data(9999999, 9999999);

    %For each digit, choose random N_train and N_test from the pool
    for d = 1:10
        N = length(train_temp{d});
        if N_train >= N
            train{d} = train_temp{d};
        else
            perm = randperm(N);
            indices = perm(1:N_train);
            for i = 1:N_train
                train{d}{i} = train_temp{d}{indices(i)};
            end
        end

        N = length(test_temp{d});
        if N_test >= N
            test{d} = test_temp{d};
        else
            perm = randperm(N);
            indices = perm(1:N_test);
            for i = 1:N_test
                test{d}{i} = test_temp{d}{indices(i)};
            end
        end
    end


end

