function samples_mod = MatrixToSamples(X, samples)
    sample_size = size(samples{1}{1});
    class_count = length(samples);

    samples_mod = samples;
    
    item_idx = 0;
    for d = 1:class_count
        for i = 1:length(samples{d})
            item_idx = item_idx + 1;
            samples_mod{d}{i} = reshape(X(:, item_idx), sample_size);
        end
    end
end
