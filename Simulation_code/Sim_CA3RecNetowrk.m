function [phase, Env_track, final_weight, weight_before, weight_mean, weight_var] = Sim_CA3RecNetowrk(n_ca3_per_track, n, M, s, P, D, plot_figure, seed, use_gpu)
%{
We use Sim_CA3RecNetwork to simulate the weight matrix of N = n_ca3_per_track * M CA3 cells 
over n distinct environments, which mimic the animal exploring novel enviroments.

Parameters:
n_ca3_per_track: number of active CA3 cells over the track. 
    PD:
n: number of distinct enviroments, each environment is defined by the 
order of the actve CA3 cells. 
    PD: We use random permutation to generate n different envronments.
M: number of CA3 neurons that encode the same information.
    PD: according to paper (need to cite), increasing the M leads to the
    improvemnet of storage capacity of net.
s: sparsity.    range of s = (0, 1). 0 indicates a very sparse net.  
    PD: increasing the sparsity will reduce the interference of different
    environment that leads to improve the memory capacity
P & D : learning constant for potentiation and depression.
use_gpu: 
%}

    if nargin < 9
        use_gpu = true; 
    end
    
    if M == 1
        use_gpu = false;
    end

    % set the seed to reproduce the same results
    rng(seed)
    % start with initial weights equal to 0
    init_weight = 0;

    %-----------------------------------------------------------------------%
    % define N: total number of CA3 cells
    N = n_ca3_per_track * M;                            % N indicates the total CA3 cells in the net.

    % define the phase for 1 cells scenario
    phase_dist = 2 * pi / n_ca3_per_track;              % only a part of net is active during learning
    phase = -pi + phase_dist/2 : phase_dist : pi;       % differnt to the paper, we use -pi to pi 
    phase = phase';                                     % transpose the row vector.

    % expand to the multi-neurons scenario
    phase = repelem(phase, M * s);                      % define the phase for multi neurons scenario
    N_active_cells = round(N * s);                      % active cells in the net                

    %-----------------------------------------------------------------------%
    % Rules for CA3 recurrent neurons
    % f_P = 1 + cos(phase)
    % f_D = 1 - cos(phase)
    f_P = @(x) (1+cos(x));
    f_D = @(x) (1-cos(x));

    %-----------------------------------------------------------------------%
    % Generate random environments        
    Env_track = generate_environments();

    %-----------------------------------------------------------------------%
    % set the initial weight with some noise
    % the weight is limited in 0 to 1    
    if use_gpu
        Connectivity = gpuArray(ones(N, N) * init_weight + unifrnd(0, 0.1, N, N));
    else
        Connectivity = ones(N, N) * init_weight + unifrnd(0, 0.1, N, N);
    end
    
    weight_mean = zeros(1, n);
    weight_var = zeros(1, n);
    
    %-----------------------------------------------------------------------%
    % Start to simulation
    
    for env_i = 1:n
        %disp(env_i)
        active_neurons_env = Env_track(:, env_i);      % get the information of the enviroment

        selected_weight = Connectivity(active_neurons_env, active_neurons_env); % select the weight of active neurons

        delta_phase = phase - phase';	
        
        if use_gpu
            delta_weight = P * (1 - selected_weight) .* gpuArray(f_P(delta_phase)) - D * selected_weight .* gpuArray(f_D(delta_phase));
        else
            delta_weight = P * (1 - selected_weight) .* f_P(delta_phase) - D * selected_weight .* f_D(delta_phase);
        end
        
        clear delta_phase
        updated_selected_weight = selected_weight + delta_weight;   % update the selected weights
        clear selected_weight delta_weight
        updated_selected_weight = max(0, min(1, updated_selected_weight)); 

        Connectivity(active_neurons_env, active_neurons_env) = updated_selected_weight;

        if env_i == n - 1
           weight_before = Connectivity; 
        end
        
        weight_mean(env_i) = mean(Connectivity, "all");
        weight_var(env_i) = var(Connectivity(:));
    end
    
    if plot_figure
        [w_m, w_var] = extra_values();
        subplot(2, 1, 1)
        plot(1:n, weight_mean, "k.")
        title("Evolution of Mean weight")
        yline(w_m, "r-")
        box off
        subplot(2, 1, 2)
        plot(1:n, weight_var, "k.")
        yline(w_var, "r-")
        box off
    end
    
    final_weight = Connectivity;

    %% auxiliar functions %%

    function Env_track = generate_environments()
        % for each lap, we permuting the sequence 1:N then we select the first 
        % active_neurons elements as the representation of the enevinroment
        map = cell2mat(arrayfun(@(dummy) randperm(N), 1:n, 'UniformOutput', false)'); % random permutate 1:N n times, then select only first N_active_cells rows.
        map = map';
        Env_track = map(1:N_active_cells, :);
    end

    function [w_m, w_var] = extra_values()
        w_m = P/(P+D);
        w_2 = (3/2*P^3-1/2*P^2*D-2*P^2)/((P+D)*(3/2*P^2+3/2*D^2+P*D-2*P-2*D));
        w_var = w_2 - w_m^2;
    end
end