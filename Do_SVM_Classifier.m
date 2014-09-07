function err = Do_SVM_Classifier( db, train_set, test_set, C, gamma, is_linear )

    % Extract feature vector indices of the objects in the training set and their
    % respective classes.
    ind_features_train = [];
    feature_class = [];
    for k = 1:length(train_set)
        ind = db.indices{train_set(k)};
        ind_features_train = [ind_features_train ind];
        feature_class = [feature_class ...
            db.src.objects(train_set(k)).class*ones(1,length(ind))];
    end

    ind_features_test = [];
    for k = 1:length(test_set)
        ind = db.indices{test_set(k)};
        ind_features_test = [ind_features_test ind];
    end

    X_train = db.features(:, ind_features_train);
    Y_train = feature_class;

    size(X_train)
    size(Y_train)
    
    if is_linear
        params = ['-q -c ' num2str(C) ' -t 0']; %Linear kernel, Ignore gamma
    else
        params = ['-q -c ' num2str(C) ' -t 2 -g ' num2str(gamma)]; %Gaussian kernel
    end

    X_test = db.features(:, ind_features_test);

    %Note that I use libsvm directly
    model = svmtrain(double(Y_train'), double(X_train'), params);

    %Note - Giving fake labels as input
    labels = svmpredict(zeros(1, length(test_set))', double(X_test'), model);

    err = classif_err(labels,test_set,db.src);

end

