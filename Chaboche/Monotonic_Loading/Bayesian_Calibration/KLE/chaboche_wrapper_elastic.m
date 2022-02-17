function [store_values] = chaboche_wrapper(X)

    % Input:
    %   X : [N, 6]
    %       random variables to Python
    % 
    % Output:
    %   store_values: [N, n]
    %                 Returns stress at t = [t1, t2, .... tn]
    
    shape = size(X); % [N, 6]
    N = shape(1);
   
    % Time integration
    t_min = 0;
    t_max = 200/((5e-2)*2.1e5); % smallest yield limit / (strain rate * largest Young's Modulus) = 0.019
    delta_t = 1e-3;
    n = floor(t_max/delta_t); % 19
    t_max = n * delta_t; % to ensure time points lie on mutliples of delta_t, else interpolation must be used

    % time points at which model will be calibrated
    t = linspace(t_min, t_max, n+1); % 19 points
    t = t(10+1); % A point in the middle
    
    % index for value t = ti
    % convert to int
    % add one because index starts with 1
    index = int32((t/t_max)*n) + 1;
            
    % Input: each row contains one set of parameters
    % Output: Stress response [N, n_t+1] 
    % When necessary, a numpy array can be created explicitly from a MATLAB array:
    % x = rand(2,2);           % MATLAB array
    % y = py.numpy.array(x);   % numpy array
    answer = py.chaboche.surrogate_evaluate(py.numpy.array(X));
	
    % make MATLAB compatible
    temp = double(answer);

    % extract values at required points
    % size: [N, n+1]
    store_values = temp(:, index);

end
