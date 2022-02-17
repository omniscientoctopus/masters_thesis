function [store_values] = chaboche_wrapper(X)
	
    % Input random variables to Python
    % X : [N, 6]
    % Returns stress at time 't = 2s' 
    % double(answer) = [N, 1]
    
    shape = size(X); % [N, 6]
    N = shape(1);
    store_values = zeros(N, 1);
    % before yielding begins
    t_max = 0.01; % 0.5 * time before plasticity begins = 0.5 * (250/(200*1000*5*0.01)) = 0.5* (Initial yield limit/(Young_Mod * strain rate))
    
    for i = 1:N 
        
        % each row contains one set of parameters
        answer = py.chaboche.monotonic(X(i, :), t_max);
        store_values(i) = double(answer);
  
    end

end
