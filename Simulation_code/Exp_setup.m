function [L, n_ca3, ca3_peak_locations, dt, plateau_duration] = Exp_setup(L, n_ca3)

    dt = 10;
    % length of track;  number of input ca3;   time step in ms
    
    plateau_duration = 300; 
    % in mileseconds
    
    peak_dist = L / n_ca3; 
    ca3_peak_locations = peak_dist/2  : peak_dist : L;
end

