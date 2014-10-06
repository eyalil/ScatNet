function [distance_indices, S_multiclass] = CalcFisher(samples)

%Calc average inter-cluster distance for each class
%Calc average distance between classes
%   Input: features and a label for each such feature list

    if iscell(samples)
        class_count = length(samples);
        sample_size = size(samples{1}{1}, 1) * size(samples{1}{1}, 2) * size(samples{1}{1}, 3);
    
        [X Y] = SamplesToMatrix( samples );
        
    else
       X = samples.X;
       Y = samples.Y;
        
       class_count = length(unique(Y));
       sample_size = size(X,1);
    end
    
    
    
    %Project into interesting subspace
    %interesting_features = [];
    %
    %for i = 1:sample_size
    %    if var(X(i,:)) > 0
    %        interesting_features = [interesting_features i];
    %    end
    %end
    %
    %X = X(interesting_features, :);
    %
    %sample_size = size(X,1); %Update sample size
    

    %Get mu and sigma for each class
    class_mu  = zeros(sample_size, class_count);
    class_sig = zeros(sample_size, sample_size, class_count);
        
    for d = 1:class_count
        X_per_cluster = X(:, Y == d);
        
        class_mu(:, d)     = mean(X_per_cluster, 2);
        class_sig(:, :, d) = cov(X_per_cluster');
    end
    
    %Apply shrinkage - to avoid singular variances
    lambda = 1e-6;
    shrink = @(x) (1-lambda)*x + lambda*eye(size(x));
    for d = 1:class_count
        class_sig(:, :, d) = shrink(class_sig(:, :, d));
    end
    
    %Compute "Sigma_b" - total sigma
    sig_b = zeros(sample_size, sample_size);
    mu_of_mus = mean(class_mu, 2);
    for d = 1:class_count
        sig_b = sig_b + (class_mu(:,d) - mu_of_mus) * (class_mu(:,d) - mu_of_mus)';
    end
    
    sig_s = sum(class_sig, 3);
    sig_mc = sig_s^(-1) * sig_b;
    eig_sig_mc = eig(sig_mc);
    
    S_multiclass = abs(eig_sig_mc(1));
    
    %fprintf('S_MC is: %g\n', S_multiclass);
    

    
    %Compute the Fisher factor
	distance_indices = zeros(1, class_count);
    %temp2 = zeros(1, class_count);
	
    for d = 1:class_count       
        for d2 = 1:class_count
            if (d == d2)
                continue;
            end
            
            mu0 = class_mu(:, d);
            mu1 = class_mu(:, d2);
            sig0 = class_sig(:, :, d);
            sig1 = class_sig(:, :, d2);
            
            %w = inv(sig0 + sig1) * (mu1 - mu0);
            %sigma_between = (w' * (mu1 - mu0)) .^ 2;
            %sigma_within = w' * (sig0 + sig1) * w;        
            %S = sigma_between / sigma_within;
            
            %S_between = (mu1 - mu0)' * (mu1 - mu0);
            %S_within  = trace(sig1 + sig0);
            
            S = (mu1 - mu0)' * inv(sig1 + sig0) * (mu1 - mu0);
            
            %S2 = sqrt(trace(sig0)) / (sqrt(trace(sig1) + sum((mu1 - mu0) .^ 2))); %sqrt(trace(sig0)) / (sqrt(trace(sig1)) + sqrt(sum((mu1 - mu0) .^ 2)));
            
            %fprintf('For %d <-> %d we get (%g / %g) => %g; %g\n', d, d2, S_between, S_within, S, S2);
            
            distance_indices(d) = distance_indices(d) + S;
            %temp2(d) = temp2(d) + S2 / 9;
        end
    end

    %fprintf('Temp2 is: ');
    %fprintf('%g ', temp2);
    %fprintf('\n');
end

