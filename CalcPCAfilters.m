function PCA_filters = CalcPCAfilters(train, PCA_Type, filter_count)
    train_size = size(train{1}{1}, 1) * size(train{1}{1}, 2);
    class_count = length(train);
    
    train_length = 0;
    for d = 1:class_count
        train_length = train_length + length(train{d});
    end
    
    if PCA_Type == 1

        patch_size_x = 8; %Might want to make this non-constant...
        patch_size_y = 8;
        target_size = patch_size_x * patch_size_y;

        shrink_factor_x = 4;
        shrink_factor_y = 4;

        X_train = zeros(train_length * (shrink_factor_x * shrink_factor_y), target_size);

        p = 1;
        for d = 1:class_count
            for i = 1:length(train{d})
                I = train{d}{i};
                for x = 1:shrink_factor_x
                    for y = 1:shrink_factor_y
                        I_patch = I( (x-1)*patch_size_x+1 : x*patch_size_x, (y-1)*patch_size_y+1 : y*patch_size_y);

                        X_train(p,:) = reshape(I_patch, [1, target_size]);
                        p = p+1;
                    end
                end
            end
        end
    else
        X_train = zeros(train_length, train_size);

        p = 1;
        for d = 1:class_count
            for i = 1:length(train{d})
                X_train(p,:) = reshape(train{d}{i}, [1 train_size]);
                p = p+1;
            end
        end
    end

    coeff = princomp(X_train);
    filters = coeff(:, 1:filter_count);

    PCA_filters = filters;

end

