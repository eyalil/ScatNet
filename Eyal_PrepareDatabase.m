function [ db, train_set, test_set ] = Eyal_PrepareDatabase( train, test, train_features, test_features, N_train )

    class_count = length(train);

    %Set object indices and class
    db = struct();

    db.src.classes = 1:class_count;
    db.src.objects.class = [];
    
    train_idx = 1;
    for d = 1:class_count
        max_train = length(train{d});
        
        for i = 1:min(N_train, max_train)
            db.src.objects(train_idx).class = d;
            train_idx = train_idx + 1;
        end
    end
    train_count = train_idx - 1;

    test_idx = train_count + 1;
    for d = 1:class_count
        N_test_d  = length(test{d});

        for i = 1:N_test_d
            db.src.objects(test_idx).class = d;
            test_idx = test_idx + 1;
        end
    end
    test_count = test_idx - train_count - 1;
        
    
    
    





    
    rows = size(train_features{1}{1},1);
    
    cols = 0;
    for d = 1:class_count
        cols = cols + size(train_features{1}{1},2) * (min(N_train,length(train_features{d})) + length(test_features{d}));
    end

    db.features = zeros(rows,cols);

    db.indices = cell(1,length(db.src.objects));

    % Fill in the features array using the pre-calculated features.
    r = 1;
    train_idx = 1;
    test_idx = train_count + 1;

    for d = 1:class_count
        max_train = length(train{d});
        for i = 1:min(N_train, max_train)
            ind = r:r+size(train_features{d}{i},2)-1;
            db.features(:,ind) = train_features{d}{i};
            db.indices{train_idx} = ind;
            train_idx = train_idx + 1;
            r = r+length(ind);
        end
    end

    for d = 1:class_count
        N_test_d  = length(test{d});
        for i = 1:N_test_d
            ind = r:r+size(test_features{d}{i},2)-1;
            db.features(:,ind) = test_features{d}{i};
            db.indices{test_idx} = ind;
            test_idx = test_idx + 1;
            r = r+length(ind);
        end
    end
    
    
    
    %Set up train set, test set
    train_set = 1:train_count;
    test_set  = (train_count+1):(train_count+test_count);
    
end

