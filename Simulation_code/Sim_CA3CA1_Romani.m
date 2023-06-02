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
    
    %%% model parameters: Aaron game me in the email
    %{
    % dont modify these parameters, tag_999
    tau_elig = 1.664103 * 1000  ; % timescale for local signal
    tau_inst = 4.370065E-1 * 1000 + 300;  % timescale for global signal
    alpha_pot = 4.150117E-01;
    beta_pot =  2/(4.541096E-01 +0.1);
    alpha_dep = 2.618154E-02;
    beta_dep = 2/(9.977173E-02 +0.045);     
    k_pot = 0.9;
    k_dep = 0.275;
    %}
    
    %
    % tag_1000, current weight 01/10/2022
    %
    tau_elig = 1.664103 * 1000  ; % timescale for local signal
    tau_inst = 4.370065E-1 * 1000 + 300;  % timescale for global signal
    alpha_pot = 4.150117E-01;
    beta_pot =  2/(4.541096E-01 +0.1);
    alpha_dep = 2.618154E-02;
    beta_dep = 2/(9.977173E-02 +0.045);     
    k_pot = 0.9;
    k_dep = 0.275;
    
    
    
    
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
