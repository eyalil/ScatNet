function [ train_features_norm, test_features_norm ] = Eyal_NormalizeFeatures(train_features, test_features, N_train, normalization_method)        

    dim_of_features = size(train_features{1}{1});
    d1 = dim_of_features(1); %Number of coeffs
    d2 = dim_of_features(2); %x/8 of orig image
    d3 = dim_of_features(3); %y/8 of orig image
            
    %max_per_feature = zeros(dim_of_features);
    norm_per_coefficient = zeros([d1, 1]);

    %Find norm per scattering coefficient
    for d = 1:10
        max_train = length(train_features{d});
        for i = 1:min(N_train, max_train)
            %max_per_feature = max(max_per_feature, abs(train_features{d}{i}));
            
            features = train_features{d}{i};
            features_per_coeff = reshape(features, [d1, d2*d3]);
            
            if strcmp(normalization_method, 'PCA')
                norm_per_coefficient_for_this_img = sqrt(sum(features_per_coeff .^ 2, 2));
            elseif strcmp(normalization_method, 'SVM') || strcmp(normalization_method, 'LinearSVM')
                norm_per_coefficient_for_this_img = max(abs(features_per_coeff), [], 2);
            else
                fprintf('Error!!!!\n');
            end
            
            norm_per_coefficient = max(norm_per_coefficient, norm_per_coefficient_for_this_img);
        end
    end 

    %Renorm and flatten
    %(Perform DCT?)
    train_features_norm = cell(1, 10);
    test_features_norm  = cell(1, 10);
    
    for d = 1:10
        max_train = length(train_features{d});
        N_test_d  = length(test_features{d});
        for i = 1:min(N_train, max_train)
            for c = 1:d1
                train_features_norm{d}{i}(c,:,:) = train_features{d}{i}(c,:,:) ./ norm_per_coefficient(c);
            end
            train_features_norm{d}{i} = reshape(train_features_norm{d}{i}, [d1*d2*d3, 1]);
            
            %train_features_norm{d}{i} = cut_high_freq(train_features_norm{d}{i});
        end
        for i = 1:N_test_d
            for c = 1:d1
                test_features_norm{d}{i}(c,:,:) = test_features{d}{i}(c,:,:) ./ norm_per_coefficient(c);
            end
            test_features_norm{d}{i} = reshape(test_features_norm{d}{i}, [d1*d2*d3, 1]);
            
            %test_features_norm{d}{i} = cut_high_freq(test_features_norm{d}{i});
        end
    end 

end

function Y = cut_high_freq(X)
    dct_result = dct(X);
    Y = dct_result(1:end/2);
end