function [synaptic_weight, list_pot, list_dep, ca3_peak_locations] = Sim_CA3CA1_Romani(velocity, inductions, init_weight, L, n_ca3)
    % get the basic setup parameter as length of track, number of CA3, peak
    % positions, time step, duration of the plateau
    if nargin < 4
        L = 187;
        n_ca3 = 100;
    elseif nargin < 5
        n_ca3 = 100;
    end
    [L, n_ca3, ca3_peak_locations, dt, plateau_duration] = Exp_setup(L, n_ca3);
    
    
    binned_dx = velocity * dt / 1000;
    dx = binned_dx : binned_dx : L; 
    
    
    %{
    parameter set for the draft of article
    % tag_1000, current weight 01/10/2022 
    %
    tau_elig = 1.664103 * 1000  ;           % 1664.103
    tau_inst = 4.370065E-1 * 1000 + 300;    % 737
    alpha_pot = 4.150117E-01;               % 0.415
    beta_pot =  2/(4.541096E-01 +0.1);      % 3.6094
    alpha_dep = 2.618154E-02;               % 0.02618
    beta_dep = 2/(9.977173E-02 +0.045);     % 13.8149
    k_pot = 0.9;
    k_dep = 0.275;
    %}

    %{
    % parameter set in the Milstein paper taking the mean of experimental
    % data
    tau_elig = 863.91;%1500;%863.91 +  113.93;
    tau_inst = 542.76; %750;%542.76 + 95.47;
    alpha_pot = 0.24;% + 0.05;
    beta_pot =  30.32;%10;%30.32 - 6.5;
    alpha_dep = 0.09;
    beta_dep = 2260;
    k_pot = 2.27;
    k_dep = 0.33;
    %}

    %
    % parameter set for the single spike case
    tau_elig = 2500;
    tau_inst = 1500;
    alpha_pot = 0.5;
    beta_pot =  4;
    alpha_dep = 0.01;
    beta_dep = 44.44;
    k_pot = 1.7;
    k_dep = 0.204;
    %}
    
    % get local and global signal
    % local signals are the filtered version of firing rate of CA3
    % global signal is the filtered version of the binary induction (1: there is the induction, 0 otherwise)
    [local_signal, global_signal] = signal_preprocessing(tau_inst, tau_elig, dt, dx, ...
    ca3_peak_locations, inductions, plateau_duration, velocity);
    
    % maximum lap
    max_lap = size(global_signal, 1);
    
    % weight matrix to store the results, also we iniciate the initial
    % condition
    synaptic_weight = zeros(max_lap + 1, n_ca3);  % row: lap   col: ca3
    synaptic_weight(1, :) = ones(1, n_ca3) * init_weight;
   
    % matrix to store the potentiation and depotentiation    
    list_pot = zeros(max_lap, n_ca3);
    list_dep = zeros(max_lap, n_ca3);
    

    % start to run simulation for each lap
    for lap_i = 1:max_lap
        
        % get the corresponding global signal
        global_signal_i = global_signal(lap_i, :);
        % compute the overlaping between the local signal and global signal
        overlapping_signal = local_signal .* global_signal_i;
        
        % the weight are only updated if there is overlaping between two
        % bioquimical signals
        if max(overlapping_signal, [], "all") > 0
            % compute the nonlinear overlapping
            [pot, dep] = pot_dep(overlapping_signal, alpha_pot, beta_pot, alpha_dep, beta_dep, dt);
            % compute the potentiation and depotentiation
            potentiation = pot * k_pot; % synaptic processing
            depression = dep * k_dep;
            
            % compute the delta weight due the plateau potential
            delta_weight = potentiation' .* (1 - synaptic_weight(lap_i, :)) - depression' .* synaptic_weight(lap_i, :);
            
            % update the weight between 0 and 1
            % store the new weight and potentiation and depotentiation            
            synaptic_weight(lap_i + 1, :) = max(0, min(1, synaptic_weight(lap_i, :) + delta_weight));
            list_pot(lap_i, :) = potentiation;
            list_dep(lap_i, :) = depression;
           
            %diy_plot()
            %pause
        else
            % otherwise we copy the previous synaptic weight
            if lap_i > 1
                synaptic_weight(lap_i + 1, :) = synaptic_weight(lap_i, :);
            end
        end
        
    end
        
    function diy_plot()
        % auxiliar function used to plot the weight, potentiation and
        % depression. 
        % to show the plots, uncomment the lines 75 76.
        aux = inductions(inductions(:, 1) == lap_i, 2);
        if not(isnan(aux))
            aux_time = (ca3_peak_locations - aux) / velocity;
            
            
            plot(aux_time, synaptic_weight(lap_i + 1, :), "-k")
            hold on
            plot(aux_time, potentiation, "+y", "LineWidth", 1)
            plot(aux_time, depression, "*r", "LineWidth", 1)
            hold off
            axis tight
            title("lap = " + lap_i)
            xlabel("Relative time to plateau onset")
            xline(0)
        end
    end
    
    
end



function res = scaled_sigmoid(x, alpha, beta)
% auxiliar function to plot the scaled sigmoid function
    
    res = (s_hat(x) - s_hat(0)) ./ (s_hat(1) - s_hat(0));
    
    function aux = s_hat(y)
        aux = 1./(1+exp(-beta*(y-alpha)));
    end
end


function [pot, dep] = pot_dep(overlapping_signal, alpha_pot, beta_pot, alpha_dep, beta_dep, dt)
% auxiliar function to plot the nonlinear overlapping
    q_pot = scaled_sigmoid(overlapping_signal, alpha_pot, beta_pot);
    q_dep = scaled_sigmoid(overlapping_signal, alpha_dep, beta_dep);
    
    pot = trap_sum(q_pot, dt/1000);
    dep = trap_sum(q_dep, dt/1000);
    
    function res = trap_sum(input, dt)
        %input has n m size
        res = sum(input(:, 1:end-1), 2) + sum(input(:, 2:end), 2);
        res = res * dt / 2;
    end
end

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