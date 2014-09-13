function [distance_indices class_centers] = CalcDistances(samples)

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
    
    

    class_centers = zeros(sample_size, class_count);
    rand_I = rand(sample_size, 1) * 2 - 1;
    norm_dist = 0;
    
    for d = 1:class_count
        X_per_cluster = X(:, Y == d);
        
        class_centers(:, d) = mean(X_per_cluster, 2);
        
        norm_dist = norm_dist + sqrt(sum((rand_I - class_centers(:, d)) .^ 2));
    end
    
    norm_dist = norm_dist / class_count;
    
       
	distance_indices = zeros(1, class_count);
	
    for d = 1:class_count
        inter_distance = 0;
        intra_distance = 0;
        
        for d2 = 1:class_count
            X_per_cluster = X(:, Y == d);

            for idx = 1:size(X_per_cluster, 2)
                X_per_cluster(:,idx) = X_per_cluster(:,idx) - class_centers(:, d2);
            end

            X_per_cluster = X_per_cluster .^ 2;

            dist = mean(sqrt(sum(X_per_cluster, 1))) / norm_dist;

            if (d == d2)
                inter_distance = inter_distance + dist;
            else
                intra_distance = intra_distance + dist;
            end
        end
        
        intra_distance = intra_distance / (class_count - 1);
        distance_indices(d) = inter_distance / intra_distance;
        
        %fprintf('For class %d inter = %g; intra = %g; index = %g\n', d, inter_distance, intra_distance, distance_index);
    end

end

