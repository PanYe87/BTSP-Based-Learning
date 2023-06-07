function [sorted_weight, phase] = Aligment(Synaptic_weight, Env_track, k, phase, ...
    n_ca3_per_track, M, s, use_gpu)
%{
Given a weight matrix, we want to subtract the activated neurons at the
environment n-k and align them according to the \Delta phase.
    
Parameters:
Synaptic_weight: Weight matrix
Env_track: a list of coded environment
k: number of past environment with respect to the final one
phase: the phase of each active neuron
n_ca3_per_track: number of CA3 distributed on the track
M: number of neurons that encode same informaton
s: sparsity
is_stack: average the weight of same information coded neurons.
%}
    if nargin < 8
        use_gpu = false;
    end
        
    if use_gpu
        Synaptic_weight = gpuArray(Synaptic_weight);
    end
    
    selected_env = Env_track(:, end - k);
    selected_weight = Synaptic_weight(selected_env, selected_env);
    
    
    % each col per neuron
    delta_phase = (phase - phase');
    if use_gpu 
        delta_phase = gpuArray(delta_phase);
   end
    delta_phase(delta_phase > pi) = delta_phase(delta_phase > pi) - 2 * pi;
    delta_phase(delta_phase < -pi) = delta_phase(delta_phase < -pi) + 2 * pi;
    
    for i = 1:numel(phase)
        delta_phase_i = delta_phase(i, :);
        [~, aux] = sort(delta_phase_i);
        selected_weight(i, :) = selected_weight(i, aux);
    end
    sorted_weight = selected_weight;
    
    % Corret the phase for even N
    if mod(n_ca3_per_track, 2)== 0
        ind_middle = round(n_ca3_per_track*M*s/2);
        m1 = mean(sorted_weight(1:ind_middle, :));
        m2 = mean(sorted_weight(ind_middle + 1 : end, :));
        [~, m_1] = max(m1);
        [~, m_2] = max(m2);
        
        if m_1 < m_2
            sorted_weight(ind_middle + 1 : end, :) = [sorted_weight(ind_middle + 1 : end, 2:end), sorted_weight(ind_middle + 1 : end, 1)];
            phase = phase + abs(phase(ind_middle));
        elseif m_1 > m_2
            sorted_weight(ind_middle + 1 : end, :) = [sorted_weight(ind_middle + 1 : end, end), sorted_weight(ind_middle + 1 : end, 1:(end-1))];
            phase = phase - abs(phase(ind_middle));
        end
        
        
    end
    
    
    
end