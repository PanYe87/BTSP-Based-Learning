function FR_sim_Network(load_file, I0, w0, w_max, save_file, w1_range)
    
    load(load_file, "P", "D", "M", "s", "Env_track", "final_weight", "phase", "N")
            % load the weight matrix and necesarry inf
    [tau, d_t, err, check_dynamic, plot_mean, activation_all, tracker, max_step] = FR_sim_setup(); 
            % load the inf for FR sim
    
    % set simulation range
    if nargin < 6
        w1_range = [2, 12];
    end
    
    f = @(x, y, k, Sparsity) 2*x*y./(x+y)*(1-Sparsity^2*(x+y)).^k;          % amplitude equation
    precision = 0.15; % number of dots in diagram
    eta_max = 1000;
    etas = generate_etas(eta_max, P, D, s, w_max, f, precision, w1_range);
    etas_seq = [etas; flip(etas)];  
    
    w1s = w_max * f(P, D, etas_seq, s); 
    
    % saving the result: forward and backward. Forward eta from 0 to end.
    % Backfward indicates the eta from end to 0.
    FR_results = cell(2, 1);
    FR_total = cell(2, 1);
    
    % scale the weight matrix
    % weights_network = w_max * (final_weight - mean(final_weight, "all")) + w0;
    %weights_network = w_max * (final_weight - P/(P+D)) + w0;
    
    if s < 0.5
        weights_network = gpuArray(w_max * (final_weight - P/(P+D)) + w0);
    else
        weights_network = w_max * (final_weight - P/(P+D)) + w0;
    end
    
    % delete the old one to save the memory
    final_weight = [];
    
    disp("    Simulation for Network: " +  length(etas) * 2 + " iterations to run")
    % simulation backward (step == 2) or foward (step == 1, eta from 0 to end)
    for step = 1:2
        
        etas = etas_seq(step, :);
        
        if step == 1 
            r_base = 1.5; % step = 1, etas increase. We want to set the large initial condition 
        else
            r_base = I0 ^ 2;  % step = 2, etas deseas, the initial condition is small
        end
        
        
        % Auxiliar variable to save resulting firing rate in each
        FR_result_aux = zeros(length(etas), N * M * s);
        FR_total_aux = zeros(length(etas), N * M);
        
        % run simulation for each eta 
        for i = 1:length(etas)       
            eta = etas(i);      % the eta we want visualize
            eta_IC = eta;       % the eta for inital condition. Generally we have the same value as eta
           
            % set initial condition. Different for each eta
            ind_evn = Env_track(:, end - eta);                              % the cell indexs for simulated environment
            FR_init_network = zeros(N * M, 1);                
            FR_init_network(ind_evn) = FR_init_network(ind_evn) + (1 + cos(phase)) * r_base;    
                            % bumpy initial condition and its magnitude
                            % depends on the step.
            
            % check_dynamic = true;
            [final_Evn, ~, FR_total_ind, n_step] = FR_simulation(N, M, s, Env_track, phase, weights_network, ...
                eta, eta_IC, I0, d_t, tau, err, check_dynamic, plot_mean, FR_init_network, max_step, activation_all, tracker);
            
            
            fileID = fopen("Files\weight_matrix\Log_network.txt",'a+');
            fprintf(fileID, 'step = %5d, eta = %5d, Max step = %5d \n', step, eta, n_step);
            fclose(fileID);
            
            FR_total_aux(i, :) = FR_total_ind';
            FR_result_aux(i, :) = final_Evn';  % save results in parfor 
        end
        
        FR_total{step} = FR_total_aux;
        FR_results{step} = FR_result_aux;              % save result using a cell variable
    end
    
    weights_network = [];
    include_noise = nan;
    save(save_file, "FR_results", "FR_total", "N", "etas_seq", "w1s", "w0", "I0", "include_noise", "w_max", "P", "D", "M", "s","phase")
end

function [tau, d_t, err, check_dynamic, plot_mean, activation_all, tracker, max_step] = FR_sim_setup()
    % setup
    tau = 20;
    d_t = 0.5;
    err = 1e-12;
    
    check_dynamic = false;
    plot_mean = true;
    activation_all = false;
    tracker = false;
    max_step = nan;
end


function etas = generate_etas(eta_max, P, D, s, w_max, f, precision, w1_range)
    eta_seq = 0:eta_max;
    w1s = w_max * f(P, D, eta_seq, s);
    
    desire_w1s =  [linspace(12, 6, 4), 6:-precision:2, linspace(2, 1, 3)];
    desire_w1s = flip(unique(desire_w1s));
    aux = abs(w1s' - desire_w1s);
    [~, b] = min(aux);
    etas = eta_seq(b);
    w1s = w_max * f(P, D, etas, s);
    etas = etas(w1s >= min(w1_range) & w1s <= max(w1_range));
    etas = unique(etas);
end