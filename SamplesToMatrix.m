function [X Y] = SamplesToMatrix( samples )
    class_count = length(samples);

    item_count = 0;
    for d = 1:class_count
        for i = 1:length(samples{d})
            item_count = item_count + 1;
        end
    end
    
    sample_size = size(samples{1}{1}, 1) * size(samples{1}{1}, 2) * size(samples{1}{1}, 3);
    X = zeros(sample_size, item_count);
    Y = zeros(1, item_count);
    
    item_idx = 0;
    for d = 1:class_count
        for i = 1:length(samples{d})
            item_idx = item_idx + 1;
            
            I = samples{d}{i};
            X(:, item_idx) = I(:);
            Y(item_idx) = d;
        end
    end
end

