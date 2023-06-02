function [local_signal, global_signal, FR] = signal_preprocessing(tau_g, tau_l, dt, dx, ...
    ca3_peak_locations, inductions, plateau_duration, velocity)
%{
parameters
* tau_g: decay time for global signal processing in mileseconds
* tau_l: decay time for local signal processing in mileseconds
* dt : scale value in mileseconds, 10
* dx: positions of animal in the treadmil, cm each point.
* ca3_peak_locations: the peak firing rate of each ca3 cells.
* inductions: a list containing induction lap and place
* plateau_duration: scale value indicating the durantion of current injection
in mileseconds to mimic the plateau potential. 300ms in the experiment
* velocity: scale value indicating the velocity of animal


- create filters for global and local signal preprocessing: f_g, f_l
- create local signal which is firing rate for CA3 -> then create expand to
the complete FR
- create global signal
%}
    tau_m = max(tau_g, tau_l);
    
    [~, global_filter] = create_filters(tau_g); % global_filter_t and local_filter_t are used for plots
    [~, local_filter] = create_filters(tau_l);
    
    
    % generate local signal and global signal
    [local_signal, FR] = get_local_signal();
    global_signal = get_global_signal();


    function [filter_t, filter] = create_filters(decay)
        filter_t = 0 : dt : (6 * tau_m);
        filter = exp(- filter_t / decay);
        filter = filter(filter >= 0.001 * max(filter));
        filter = filter / sum(filter);
        
        filter_t = filter_t (1:length(filter));
    end

    function [local_signal, CA3_FR] = get_local_signal()
        % place field properties
        field_width = 90;
        gauss_sigma = field_width / 3 / sqrt(2);
        peak_rate = 40;
        
        % entent the place field
        L = dx(end);
        ext_x = [dx - L, dx, dx + L];
        
        % compute extent FR
        f = length(dx);
        gauss_force = peak_rate * exp(-((ext_x - ca3_peak_locations') / gauss_sigma) .^ 2);
        gauss_force = gauss_force / peak_rate;
        
        % convolve the extent FR with filter
        conv_signal_aux = ext_convolution(local_filter, gauss_force);
        
        CA3_FR = gauss_force(:, 1:f) + gauss_force(:, f+1 : 2*f) + gauss_force(:, 2*f + 1 : end);
        local_signal = conv_signal_aux(:, 1:f) + conv_signal_aux(:, f+1 : 2*f) + conv_signal_aux(:, 2*f + 1 : end);
        local_signal = local_signal / max(local_signal, [], "all");
    end
    
    function global_signal = get_global_signal()        
        n = size(inductions, 1);
        L = dx(end);
        
        n_dx = length(dx);
        
        loc_max = nan;
        
        % create plateau ind indicator
        global_signal = zeros(inductions(n, 1) + 3, n_dx);
        
        laps = unique(inductions(:, 1));
        
        for i = 1:length(laps)
            lap_i = laps(i);
            ind_pos = inductions(inductions(:, 1) == lap_i, 2);
            
            global_signal_i_aux = zeros(1, n_dx * 2);
            for k = 1:length(ind_pos)
                ind_k = ind_pos(k);
                start_pos = ind_k;
                end_pos = start_pos + velocity * plateau_duration / 1000;
                
                if end_pos <= L
                    global_signal_i_aux(1:n_dx) = or(global_signal_i_aux(1:n_dx), dx >= start_pos & dx <= end_pos);
                else
                    global_signal_i_aux(1:n_dx) = or(global_signal_i_aux(1:n_dx), dx >= start_pos & dx <= end_pos);
                    global_signal_i_aux(n_dx + 1 : end) =  or(global_signal_i_aux(n_dx + 1 : end), dx <= end_pos - L);
                end
            end
            
            global_signal_i = conv(global_signal_i_aux, global_filter);
            if isnan(loc_max)
                TF = islocalmax(global_signal_i);
                TF_first_max_ind = find(TF, 1, "first");
                loc_max = global_signal_i(TF_first_max_ind);                
            end
            global_signal_i = global_signal_i(1:n_dx) + global_signal_i(n_dx + 1: n_dx * 2);
            global_signal_i = global_signal_i/ loc_max;
            global_signal(lap_i, :) = global_signal_i;
        end
        
    end
    
    function conv_signal = ext_convolution(filter, signal_matrix)
        [n, m] = size(signal_matrix);
        
        conv_signal = zeros(n, m);
        
        for i = 1:n
            signal = signal_matrix(i, :);
            conv_res = conv(signal, filter);
            conv_signal(i, :) = conv_res(1:m);
        end
    end
end