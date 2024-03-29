function filters = random_uniform_filter_bank_2d(size_in, options)

    %Handle options
    options = fill_struct(options, 'Q',1);	
    options = fill_struct(options, 'L',8);
    options = fill_struct(options, 'J',4);
    J = options.J;
    Q = options.Q;
    L = options.L;
    options = fill_struct(options, 'sigma_phi',  0.8);	
    options = fill_struct(options, 'sigma_psi',  0.8);	
    options = fill_struct(options, 'sigma_psi',  0.8);	
    options = fill_struct(options, 'xi_psi',  1/2*(2^(-1/Q)+1)*pi);	
    options = fill_struct(options, 'slant_psi',  4/L);	
	options = fill_struct(options, 'filter_format', 'fourier_multires');
    options = fill_struct(options, 'min_margin', options.sigma_phi * 2^(J/Q) );
    options = fill_struct(options, 'precision', 'single');
    sigma_phi  = options.sigma_phi;
	sigma_psi  = options.sigma_psi;
	xi_psi     = options.xi_psi;
	slant_psi  = options.slant_psi;
    
    switch options.precision
        case 'single'
            cast = @single;
        case 'double'
            cast = @double;
        otherwise
            error('precision must be either double or single');
    end
    
    
    %Handle size
    res_max = floor(J/Q);
    size_filter = pad_size(size_in, options.min_margin, res_max);
	phi.filter.type = 'fourier_multires';
    
    % Compute all resolution of the filters
	res = 0;
	
	N = size_filter(1);
	M = size_filter(2);
    
    % Compute low-pass filters phi (for now, this is not random at all)
	scale = 2^((J-1) / Q - res);
	filter_spatial =  gabor_2d(N, M, sigma_phi*scale, 1, 0, 0);
	phi.filter = cast(real(fft2(filter_spatial)));
	phi.meta.J = J;
	
	phi.filter = optimize_filter(phi.filter, 1, options);
	
	littlewood_final = zeros(N, M);
    
    % Compute band-pass filters psi
    R = 2^(J-1); %Circle radius
    
    
    
%     figure;
%     hold on;
    
    
    
    p = 1;
    for idx = 1:options.UniformRandom_Count
        x = R;
        y = R;
        while (x^2 + y^2 > R^2)
            x = 2*R*rand() - R; %Choose uniformly between -R and R
            y = 2*R*rand() - R; %Choose uniformly between -R and R
        end

        scale = sqrt(x^2 + y^2);
        angle = atan2(y, x);
        
%         plot(x, y, 'rx');

        %scale = 2^randi([0,2]);
        %angle = randi([0,L-1])  * pi / L;

        %scale = 2^j;
        %angle = theta*pi / L;

        %fprintf('%d) Scale = %g, angle = %g...\n', p, scale, angle);

        filter_spatial = morlet_2d_noDC(N, ...
            M,...
            sigma_psi*scale,...
            slant_psi,...
            xi_psi/scale,...
            angle);

        psi.filter{p} = cast(real(fft2(filter_spatial)));

        littlewood_final = littlewood_final + ...
            abs(realize_filter(psi.filter{p})).^2;

        psi.meta.j(p) = 0; %j;
        psi.meta.theta(p) = 0; %theta;
        p = p + 1;
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
%     res = 0;
%     angles = (0:L-1)  * pi / L;
%     
% 	for j = 0:J-1
% 		for theta = 1:numel(angles)			
% 			angle = angles(theta);
% 			scale = 2^(j/Q - res);
% 
%             x = scale*cos(angle);
%             y = scale*sin(angle);
%             
%             plot(x,y,'bo');
% 		end
%     end
    
    
    %keyboard;
    

	
	% Second pass : renormalize psi by max of littlewood paley to have
	% an almost unitary operator
	% NB : phi must not be renormalized since we want its mean to be 1
	K = max(littlewood_final(:));
	for p = 1:numel(psi.filter)
		psi.filter{p} = psi.filter{p}/sqrt(K/2);
		psi.filter{p} = optimize_filter(psi.filter{p}, 0, options);
	end
	
	filters.phi = phi;
	filters.psi = psi;
	
	filters.meta.Q = Q;
	filters.meta.J = J;
	filters.meta.L = L;
	filters.meta.size_in = size_in;
	filters.meta.size_filter = size_filter;
    
end