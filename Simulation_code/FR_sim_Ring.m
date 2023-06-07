function FR_sim_Ring(load_file, I0, w0, w_max, save_file, include_noise, sigma, w1_range) 

    load(load_file, "P", "D", "M", "s", "phase", "N", "n")

    % setup
    [tau, d_t, err, check_dynamic, plot_mean, activation_all, tracker, max_step] = FR_sim_setup();
    
    % set simulation range
    if nargin < 7
        sigma = 0;
        w1_range = [1, 12];
    elseif nargin < 8
        w1_range = [1, 12];
    end
    
    
    % seed to makes sure to have same intial condition for each one.
    rng(20221213)

    Env_track_ring = repmat((1:N)', 1, n);   % create the coding of environment that are all same.
    
    phase_ring = unique(phase);
    M_ring = 1; s_ring = 1; eta = 1; eta_IC = 1;                            % thoses variables should be set to 1-
    
    f = @(x, y, k, Sparsity) 2*x*y./(x+y)*(1-Sparsity^2*(x+y)).^k;          % amplitude equation
    
    eta_max = 1000;
    if include_noise
        if s == 0.025
            etas = [0:250:1500, 1500: 50: 5200];
            etas = unique(etas);
        elseif s == 0.1
            etas = [0:20:120, 121:eta_max];
        else
            etas = 0:eta_max;
        end
        w1s = w_max * f(P, D, etas, s);
        etas = etas(w1s >= min(w1_range) & w1s <= max(w1_range));
        etas_seq = [etas; flip(etas)];
        w1s = w_max * f(P, D, etas_seq, s);
        loop_set = etas_seq;
    else
        step_w1 = 0.1;
        w1s = min(w1_range) : step_w1 : max(w1_range);
        w1s = [flip(w1s); (w1s)];
        
        loop_set = w1s;
        etas_seq = [];
    end
    
    % saving the result: forward and backward. Forward eta from 0 to end.
    % Backfward indicates the eta from end to 0.
    FR_results = cell(2, 1);
    
    disp("    Simulation for Network: " +  length(loop_set(:)) + " iterations to run")
    for step = 1:2      
        
        loop_set_step = loop_set(step, :);
        
        if step == 1 % forward: eta increase == w1 decrease
            r_base = 1.5;
        else
            r_base = I0^2;
        end
        
        % initial condition
        FR_init_ring = zeros(N , 1);
        FR_init_ring = FR_init_ring + (1 + cos(phase_ring)) * r_base;
        
        FR_result_aux = zeros(length(loop_set_step), N);
       
        for i = 1:length(loop_set_step)    
            % define the connectivity matrix for the ring model simulation
            if include_noise
                eta_i = loop_set_step(i);
                weights_ring = generate_ring_noise_weight(phase_ring, w0, w_max, P, D, M, s, eta_i, include_noise);
            else
                w1 = loop_set_step(i);
                phase_matrix = phase_ring - phase_ring';
                weights_ring = w0 + w1 * cos(phase_matrix) + sigma * normrnd(0, 1, size(phase_matrix)); 
                phase_matrix = [];
            end
            
            if length(phase_ring) >= 2^9
                weights_ring = gpuArray(weights_ring);
            end
            
            
            [final_Evn, ~, ~, n_step] = FR_simulation(N, M_ring, s_ring, Env_track_ring, phase_ring, weights_ring, ...
                    eta, eta_IC, I0, d_t, tau, err, check_dynamic, plot_mean, FR_init_ring, max_step, activation_all, tracker);
            
            fileID = fopen("Files\weight_matrix\Log_ring.txt",'a+');
            if include_noise
                fprintf(fileID, 'step = %5d, eta = %5d, Max step = %5d \n', step, loop_set_step(i), n_step);
            else
                fprintf(fileID, 'step = %5d, w_1 = %5d, Max step = %5d \n', step, loop_set_step(i), n_step);
            end
            fclose(fileID);    
                
                
            %FR_init_ring = FR_final;
            FR_result_aux(i, :) = final_Evn';
            weights_ring = [];  final_Evn =[]; 
        end
        FR_results{step} = FR_result_aux;
    end        

    phase = phase_ring;
    save(save_file, "FR_results", "N", "etas_seq", "w1s", "w0", "I0", "include_noise", "w_max", "P", "D", "M", "s", "phase")
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

function weight_ring = generate_ring_noise_weight(phase, w0, w_max, P, D, M, s, eta, include_noise)
    phase_matrix = phase - phase';

    is_stack = true;

    [weight_ring, V] = Compute_mean_variance(P, D, eta, is_stack, M, s, phase_matrix, w0, w_max);

    if include_noise 
        rng(20221213)
        weight_ring = normrnd(weight_ring, 1*sqrt(V));
    end
end