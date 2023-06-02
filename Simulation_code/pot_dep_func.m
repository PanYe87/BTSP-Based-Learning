function [f_P, f_D] = pot_dep_func(L, velocity, param_P, param_D)
%{
Define the underlying bioprocesses using skewed t student function  
https://ieeexplore.ieee.org/document/7217164
%}
   
    if nargin < 3
        % parameter are nu, mu, sigma, skew
        % nu: degree of t student
        % mu: location
        % sigma: scale
        % negative skew control the movement of the center toward right
        %{
        tag_999: dont modify these parameters
        param_P = ; 
        param_D = ; 
        %}
        % tag_1000 current parameter 08/10/2022
        if velocity == 25
            param_P = [3.5, 0.685, 1.65, -1.61]; 
            param_D = [5, 1.75, 3.65, -5.35]; 
        elseif velocity == 15
            param_P = [5, 0.75, 1.97, -1.375]; 
            param_D = [3, 2.15, 3.15, -3.5];             
        elseif velocity == 35
            param_P = [2, 0.6, 1.42, -1.625]; 
            param_D = [3, 1.54, 3.85, -5.75]; 
        elseif velocity == 20
            param_P = [5, 0.75, 1.825, -1.55]; 
            param_D = [5, 1.875, 3.45, -4.55]; 
        end
    end
    
    f_P = plasticity(param_P(1), param_P(2), param_P(3), param_P(4));
    f_D = plasticity(param_D(1), param_D(2), param_D(3), param_D(4));
    
    function f = plasticity(nu, mu, sigma, skew)
            
            f = @(x) 0;
            k_max = 10;
            for k = -1*k_max  : k_max
                f = @(x) f(x) + 2/sigma .* tpdf((x + L / velocity *k - mu)/sigma, nu) .* ...
                    tcdf(skew * (x + L / velocity *k - mu)/sigma .* ...
                    sqrt( (nu+1)./(nu+ (x + L / velocity *k - mu).^2/sigma^2) ), nu + 1);
            end
    end
end