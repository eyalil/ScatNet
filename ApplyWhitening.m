function [train, test] = ApplyWhitening(train, test, fudge_factor)
    [X ~] = SamplesToMatrix( train );

    X = X';
    [X, M, P] = whiten(X, fudge_factor);
    X = bsxfun(@minus, X, M) * P;
    X = X';

    train = MatrixToSamples(X, train);


    [X2 ~] = SamplesToMatrix( test );

    X2 = X2';
    X2 = bsxfun(@minus, X2, M) * P;
    X2 = X2';

    test = MatrixToSamples(X2, test);
end

