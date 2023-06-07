function [Mean, V, phase, amp, A, B, C] = Compute_mean_variance(P, D, eta, is_stack, M, s, phase, w_mean, w_max)
%{
Compute the theoretical mean and variance curve, also the corresponding
phase. For special case, compute the constant term of the variance A.

Parameters:
P, D: constant of learning
eta: number of past environments
is_stack: individual statistics or mean statistics
M: number of nerons with same statistic properties
s: sparseness
computer_amp_var: logic function, true -> compute only the theoretical amplitude and constant term of variance,
    otherwise compute the mean and variance with phase
w_mean: w0 after the rescalling the weight matrix
w_max: w1 after the rescalling the weight matrix
%}
    

    if nargin < 7
        w_mean = P/(P+D);
        w_max = 1;
        phase = linspace(-pi, pi, 200);
        phase = phase';
    end
    if nargin < 8
        w_mean = P/(P+D);
        w_max = 1;
    end
    if nargin < 9
        w_max = 1; 
    end
    
    if isnan(phase)
        phase = linspace(-pi, pi, 200);
        phase = phase'; 
    end
    
    % amplitude
    f = @(P, D, eta, s) 2*P*D./(P+D)*(1-s.^2*(P+D)).^eta;
    [F, F_2, w_m, w_2] = extra_values();
    
    a_0 = f(P, D, 0, s);
    A0= P.^2 + 2*P*(1-P-D)*w_m+(1-P-D).^2*w_2-w_m.^2;
    B0 = 2*(P-D)/(P+D)*(P.^2+P*(1-2*P-2*D)*w_m - (P+D)*(1-P-D)*w_2);
    C0 = ((P-D)/(P+D)).^2*(P.^2-2*P*(P+D)*w_m + (P+D).^2*w_2);
    
    A = A0 * F_2 .^eta + w_m.^2 * (F_2.^eta - 1) + ...
        3/2 * P.^2 * s.^2 * (1 - F_2.^eta) / (1 - F_2) + ...
        2*P.^2 * s.^4 * (1 - 3/2 * P - 1/2 * D) / (F - F_2) * ( (1-F.^eta) / (1-F) - (1-F_2.^eta) / (1-F_2) ) + ...
        2*P * s.^2 * (1 - 3/2 * P - 1/2 * D) * w_m * ( F.^eta-F_2.^eta ) / ( F-F_2 );
    B = B0 * F_2.^eta + 2*a_0 * ( F_2.^eta - F.^(2*eta) ) * w_m + ...
        2*a_0 * P * s.^2 * ( (1 - 3/2 * P - 1/2 * D) * (F.^eta - F_2.^eta) / (F-F_2) - (F.^eta - F.^(2*eta)) /(1-F));
    C = C0*F_2.^eta +a_0.^2*(F_2.^eta - F.^(2*eta));
    
    amp = w_max * f(P, D, eta, s);
    
    A = A * w_max.^2;
    B = B * w_max.^2;
    C = C * w_max.^2;

    if is_stack
       A = A/(M*s);
       B = B/(M*s);
       C = C/(M*s);
    end
    
    if length(eta) > 1
        Mean = 0;
        V = 0;
    else
        Mean = amp * cos(phase) + w_mean;
        V = A + B * cos(phase) + C * (cos(phase)) .^ 2;
    end
      
    function [F, F_2, w_m, w_2] = extra_values()
        F = 1 - s.^2 * (P+D);
        F_2 = 1 + s.^2 * ( 3/2*P.^2 + 3/2*D.^2 + P*D - 2*P - 2*D );
        w_m = P/(P+D);
        w_2 = (3/2*P.^3-1/2*P.^2*D-2*P.^2)/((P+D)*(3/2*P.^2+3/2*D.^2+P*D-2*P-2*D));
    end
end