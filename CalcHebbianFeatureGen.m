function feature_gen = CalcHebbianFeatureGen( class_centers, hebbian_sparsity )
    class_count = size(class_centers, 2);

    %Create transition matrix
    W = zeros(32*32, 4*4*25);
    pattern = rand([class_count, 4*4*25]);

    %TEMP (sparse?)
    %for t = 1:8*4*25
    %    r = randi([1, class_count*4*4*25]);
    %    pattern(r) = 0;
    %end

    for d = 1:class_count
        S_m = class_centers(:, d);

        for j = 1:4*4*25
            W(:,j) = W(:,j) + (S_m - 0.5) * (pattern(d,j) - hebbian_sparsity);
        end
    end

    W = W / class_count;

    %Feature generating function
    feature_gen = @(x) reshape(W'*(x(:)), [4*4*25 1 1]);
end

