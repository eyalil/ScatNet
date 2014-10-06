function [X, M, P] = whiten(X, fudgefactor)

    C = cov(X);

    M = mean(X);

    [V,D] = eig(C);

    P = V * diag(sqrt(1./(diag(D) + fudgefactor))) * V';

end
