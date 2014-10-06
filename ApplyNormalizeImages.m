function [train, test] = ApplyNormalizeImages(train, test)
    class_count = length(train);
    assert(class_count == 10); %Only valid for MNIST
    
    for d = 1:10
        for i = 1:length(train{d})
            train{d}{i} = (train{d}{i} / 255)*2 - 1;
        end
        for i = 1:length(test{d})
            test{d}{i}  = (test{d}{i}  / 255)*2 - 1;
        end
    end
end

