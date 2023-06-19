function [FR_evn, FR_tracker, FR, n_step] = FR_simulation(n_ca3_per_track, M, s, map, phase, synaptic_weight, eta, eta_IC, I0, d_t, tau, err, ...
    check_dynamic, plot_mean, FR_init, max_step, activation_all, tracker)
%{
Simulation for the firing rate network.

Parameters:
n_ca3_per_track: number of CA3 populationss on the track
M: number of CA3 cells in each population
s: sparsity, P(cell is active) = s
map: coding of the visited environments
phase: phase of each cell
synaptic_weight: the final weight
eta: run simulation in the environment n - eta
eta_IC: run simulation with initial condition of the environment n - eta_IC
I0: external input, here is constant
d_t: time step
tau: time constant
err: the maximum error allowed to stop the simulation
check_dynamic: logic value, allowing the visualization of the firing rate
    throught the simulation
plot_mean: logic value, visualize the mean
activation_all: logic value, if false only take into account actived cells
    in the simulation
max_step: max step to stop the simulation, avoid the endless simulation                                            
%}
                  
    
    if nargin < 16
        activation_all = false;
        tracker = true;
        max_step = 100000;
    elseif nargin < 17
        activation_all = false;
        tracker = true;
    elseif nargin < 18
        tracker = true;
    end
    
    if isnan(max_step)
        max_step = 200000;
    end
    
    N = n_ca3_per_track;
    N_env = size(map, 2);
    W = synaptic_weight;

    rng(20221202)
    
    % the information of environment to make the plot
    ind_evn = map(:, N_env - eta);
    
    ind_evn_Initial_condition = map(:, N_env - eta_IC);
    
    if not(exist("FR_init")) || all(isnan(FR_init))
        FR_init = zeros(N * M, 1);
        FR_init(ind_evn_Initial_condition) = FR_init(ind_evn_Initial_condition) + (1 + cos(phase)) * I0^2;   
        MN = M*N;
    else
        MN = length(FR_init);
    end
    
    rng(9999)
    FR_init = max(0, FR_init + normrnd(0, mean(FR_init(ind_evn_Initial_condition))/10, [MN, 1])); % give the noise
    if check_dynamic
       figure 
    end
    
    
    n_step = 0;

    % run simulation
    %FR_tracker = FR_init(ind_evn);
    FR_tracker = FR_init;
    
    FR = FR_init;
    
    cont = true;
    while cont
        res = update_FR(); % update the FR
        
        stop_controller(); % require res and FR
                           % if the absolute difference between the mean is
                           % smaller than error, stop
        
        FR = res;
        n_step = n_step + 1;
        
        if tracker
           %FR_tracker(:, end + 1) = FR(ind_evn);    % save the fring rate 
           FR_tracker(:, end + 1) = FR;    % save the fring rate 
        end
        %{
        if n_step == 1
            get_insight();
            w = waitforbuttonpress;
        end
        %}
        if mod(n_step, 20) == 1 
            get_insight();
        end
    end
    
    FR_evn = FR(ind_evn);    
    
    %% auxiliar functions %%
    
    function res = update_FR()
        input = input_for_FI_curve();
        FI_value = FI_curve(input, false);
        d_FR = d_t/tau*(-FR + FI_value);
        res = max(0, FR + d_FR); 
        
        
        function input = input_for_FI_curve()
            
            % 1/(NM) * sum W_{ij} *r_j
            % input = (W * FR)/(N*M) + I_0;
            
            % only active neurons in that environment account
            input = zeros(MN, 1);
            aux = W*FR;
            if activation_all
                input = aux / (M * N) + I0;
            else
                input(ind_evn) = aux(ind_evn) / (M*N*s) + I0;
            end
            %input(ind_evn) = aux(ind_evn) / (N*M*S) + I0;  % taking into account number of active neurons
            %input(ind_evn) = aux(ind_evn) / (N) + I0;  % taking into only the number of position
            %input = aux / (N*M) + I0; % all the cells matter.
            %input = aux / (N) + I0; % only group
        end
        
        function FR = FI_curve(x, derivate)
            FR = zeros(1, length(x))';
            if derivate
                FR(x>1) = 1./sqrt(x(x>1) - 3/4);
                FR(x >= 0 & x <= 1) = 2 * x(x >= 0 & x <= 1);
            else
                FR(x>1) = 2*sqrt(x(x>1) -3/4);
                FR(x >= 0 & x <= 1) = x(x >= 0 & x <= 1) .^2;
            end
        end
    end
    
    function stop_controller()
        % MSE = abs(mean(FR) - mean(res));
        MSE = abs(mean(FR(ind_evn)) - mean(res(ind_evn))); % updated on 2022/12/21. the stop criteria should be mean bump at that environment

        if MSE < err
            cont = false;
        end
        
        if n_step + 1 >= max_step
            cont = false;
        end
        
    end
    
    
    function get_insight()
        if check_dynamic %& n_step >= 5000
            FR_act = FR(ind_evn);
            if plot_mean
                FR_mean = grpstats(FR_act, phase, {'mean'});
                plot(unique(phase), FR_mean, "-k")
            else
                plot(phase, FR_act, ".k")
            end
            %yline(I0 ^2, "r")
            xlim([-pi, pi])
            drawnow limitrate
        end
    end
end