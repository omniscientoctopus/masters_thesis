function [store_values_KLE_surrogate] = chaboche_wrapper(X)
	
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% KLE Surrogate %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % index for value t = ti
    % convert to int
    % add one because index starts with 1
    index = 10 + 1;
            
    % Input: each row contains one set of parameters
    % Output: Stress response [N, n_t+1] 
    % When necessary, a numpy array can be created explicitly from a MATLAB array:
    % x = rand(2,2);           % MATLAB array
    % y = py.numpy.array(x);   % numpy array
    answer = py.chaboche.KLE_surrogate_evaluate(py.numpy.array(X));
	
    % make MATLAB compatible
    temp = double(answer);

    % extract values at required points
    % size: [N, n+1]
    store_values_KLE_surrogate = temp(:, index);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%% Model %%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    shape = size(X); % [N, 6]
    N = shape(1);
    store_values_model = zeros(N, 1);
    % before yielding begins
    t_max = 0.01; % 0.5 * time before plasticity begins = 0.5 * (250/(200*1000*5*0.01)) = 0.5* (Initial yield limit/(Young_Mod * strain rate))
    
    for i = 1:N 
        
        % each row contains one set of parameters
        answer = py.chaboche.monotonic(X(i, :), t_max);
        store_values_model(i) = double(answer);
  
    end

    sprintf('Model value: %f \nPCE_Surrogate value: %f \nKLE_Surrogate value: %f', store_values_model, store_values_PCE_surrogate, store_values_KLE_surrogate)

end
