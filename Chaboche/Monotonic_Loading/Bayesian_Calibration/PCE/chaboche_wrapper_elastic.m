function [store_values_PCE_surrogate] = chaboche_wrapper(X)
	
    % Input random variables to Python
    % X : [N, 6]
    % Returns stress at time 't = 2s' 
    % double(answer) = [N, 1]
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% PCE Surrogate %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Input: each row contains one set of parameters
    % Output: Stress response [N, n_t+1] 
    % When necessary, a numpy array can be created explicitly from a MATLAB array:
    % x = rand(2,2);           % MATLAB array
    % y = py.numpy.array(x);   % numpy array
    answer = py.chaboche.PCE_surrogate_evaluate(py.numpy.array(X));
	
    % make MATLAB compatible
    temp = double(answer);

    % extract values at required point(s)
    % size: [N, 1]
    store_values_PCE_surrogate = temp;

end
