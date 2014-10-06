function filters = pca_filter_bank_2d(size_in, options)

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
	N = size_filter(1);
	M = size_filter(2);
        
    % Prepare PCA Filters
    PCA_filters = options.PCA_Filters;
    filter_size = size(PCA_filters, 1);
    filter_count = size(PCA_filters, 2);
    
    filter_image_width = sqrt(filter_size);
    filter_image_height = filter_image_width;
    
    PCA_filters = reshape(PCA_filters, [filter_image_width, filter_image_height, filter_count]);
    
    PCA_resized_filters = zeros(N, M, filter_count);
    for f = 1:filter_count
        PCA_resized_filters(:,:,f) = imresize(PCA_filters(:,:,f), [N,M]);
    end
    

    % Compute low-pass filters phi (for now, this is not random at all)
	filter_spatial = PCA_resized_filters(:,:,1);
	phi.filter = cast(real(fft2(filter_spatial)));
	phi.meta.J = J;
	
	phi.filter = optimize_filter(phi.filter, 1, options);
	
    % Compute band-pass filters psi
    
    p = 1;
    for idx = 2:filter_count

        filter_spatial = PCA_resized_filters(:,:,idx);

        psi.filter{p} = cast(real(fft2(filter_spatial)));

        psi.meta.j(p) = 0; %j;
        psi.meta.theta(p) = 0; %theta;
        p = p + 1;
    end

    

	for p = 1:numel(psi.filter)
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