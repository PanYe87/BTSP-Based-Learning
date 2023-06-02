function [synaptic_weight, list_pot, list_dep, ca3_peak_locations] = Sim_CA3CA1_map(velocity, inductions, init_weight, L, n_ca3)
    % setup
    % the length of track; number of ca3; the peak of input cells.
    if nargin < 5
        n_ca3 = 400;
    end
    peak_dist = L / n_ca3; 
    ca3_peak_locations = peak_dist/2  : peak_dist : L;    
    
    % synpatic processes via skewguassian function
    [f_P, f_D] = pot_dep_func(L, velocity);
    if velocity  == 25
        P = 2.365; 
        D = 2.57;
    elseif velocity == 15
        P = 3.05; 
        D = 3;       
    elseif velocity == 35
        P = 2; 
        D = 2.15;        
    else
        P = 2.625; 
        D = 2.8;        
    end
    
    % maximum lap which is depending the last induction lap, (last one + 3 laps, the latter as postinduction assesment)
    max_lap = inductions(end, 1) + 3;
    
    % weight matrix to store the results, also we iniciate the initial
    % condition
    synaptic_weight = zeros(max_lap + 1, n_ca3);  % row: lap   col: ca3
    synaptic_weight(1, :) = ones(1, n_ca3) * init_weight;
    
    % matrix to store the potentiation and depotentiation
    list_pot = zeros(max_lap, n_ca3);
    list_dep = zeros(max_lap, n_ca3);
    
    % start to run simulation for each lap
    for lap_i = 1:max_lap
        
        %only update if there is induction
        if ismember(lap_i, inductions(:, 1))
            % for induction lap, we first get the position where the
            % plateau occured inside of our output cell, CA1
            ind_pos = inductions(inductions(:, 1) == lap_i, 2);
            
            Pot = zeros(1, n_ca3);
            Dep = zeros(1, n_ca3);
            % Compute the potentiation and depression using the defined
            % plasticity
            for k_ind = 1:length(ind_pos)
                ind_pos_k = ind_pos(k_ind);
                relative_plateau_onset = (ca3_peak_locations - ind_pos_k) / velocity;
                Pot = Pot + f_P(relative_plateau_onset);
                Dep = Dep + f_D(relative_plateau_onset);
            end
            
            
            % we define the potentiation and depression quantities as the
            % product of the corresponding synaptic process and the
            % learning constant, here the learning constants are 1
            potentiation = Pot * P;
            depression = Dep * D;
            
            % compute the delta weight due the plateau potential
            delta_weight = potentiation .* (1 - synaptic_weight(lap_i, :)) - depression .* synaptic_weight(lap_i, :);
            
            % update the weight between 0 and 1
            % store the new weight and potentiation and depotentiation
            synaptic_weight(lap_i + 1, :) = max(0, min(1, synaptic_weight(lap_i, :) + delta_weight));
            list_pot(lap_i, :) = potentiation;
            list_dep(lap_i, :) = depression;
            
        else
            % otherwise we copy the previous synaptic weight
            if lap_i > 1
                synaptic_weight(lap_i + 1, :) = synaptic_weight(lap_i, :);
            end
        end
    end
end
