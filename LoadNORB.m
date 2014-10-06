function [train test] = LoadNORB(N_train, N_test)

    NORB_TRAIN_DATA = '../SaxeCode/rwrelease/norb_traindata.mat';
    NORB_TEST_DATA  = '../SaxeCode/rwrelease/norb_testdata.mat';
    
    %Load everything
    load(NORB_TRAIN_DATA);
    X_train = X;
    Y_train = Y;
    
    load(NORB_TEST_DATA);
    X_test = X;
    Y_test = Y;

    clear X;
    clear Y;
    
    %For each class, choose random N_train and N_test from the pool
    for d = 1:5
        good_indices = find(Y_train == d);
        N = length(good_indices);
        if N_train < N
            perm = randperm(N);
            good_indices = good_indices(perm(1:N_train)); 
        end
        for i = 1:min(N,N_train)
            cur_idx = good_indices(i);
            train{d}{i} = reshape(X_train(1:1024, cur_idx), [32, 32]);
        end


        good_indices = find(Y_test == d);
        N = length(good_indices);
        if N_test < N
            perm = randperm(N);
            good_indices = good_indices(perm(1:N_test)); 
        end
        for i = 1:min(N,N_test)
            cur_idx = good_indices(i);
            test{d}{i} = reshape(X_test(1:1024, cur_idx), [32, 32]);
        end
    end


end
