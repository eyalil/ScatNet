function [train_features test_features] = Eyal_CalculateFeatures(train_images, test_images, feature_gen)

    
    global PARALLELISM;
    matlabpool('open', PARALLELISM);

    d_length = length(train_images);
    
    train_features = cell(d_length, 1);
    test_features = cell(d_length, 1);
    
    
    parfor d = 1:d_length
    %for d = 1:d_length
        for i = 1:length(train_images{d})
            img = train_images{d}{i};
            train_features{d}{i} = feature_gen(img);
        end
    end
    
    parfor d = 1:d_length
    %for d = 1:d_length
        for i = 1:length(test_images{d})
            img = test_images{d}{i};
            test_features{d}{i} = feature_gen(img);
        end
    end
        
    matlabpool('close');
    
end

