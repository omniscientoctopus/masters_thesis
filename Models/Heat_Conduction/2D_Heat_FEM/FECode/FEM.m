function [coord,connect,v,material,Sigma_tilda] = FEM(InputDataSet)
%% Finite Element Method - Matlab Code
%
% Institut fuer Statik | TU Braunschweig
% Beethovenstrasse 51
% 38106 Braunschweig
%
% This function 'FEM' calculates the temperature and the heat flux of given 
% rectangular domains with the finite element method.
%
% Input Data (InputDataSet)
%             - param    : vector containing general calculation parameters
%             - connect  : matrix containing node connectivity
%             - coord    : matrix containing node coordinates
%             - material : matrix containing material parameters
%             - loadNode : matrix containing nodal   loads
%             - loadEle  : matrix containing element loads
%             - bc       : matrix containing temperature boundary conditions
%
% Output Data
%       Cross-sectional cuts of the temperature field       
%             - connect     : matrix containing node connectivity
%             - coord       : matrix containing node coordinates
%             - material    : matrix containing material parameters
%             - v           : Calculated temperature values
%             - Sigma_tilda : Calculated heat flux


%%                           PREPROCESSING
% The necessary input for the program is generated in the 
% PREPROCESSING-Part. Therefore, we need to define input and output
% directories, load the input data file and generate the variables from 
% the given data set.

% - Input and output directories:
%--------------------------------

dir_in  = 'InputData/';

% - Loading the calculation data:
%--------------------------------

load ([dir_in, InputDataSet])

% - The calculation data (.mat file) contains the following matrices:
%---------------------------------------------------------------------

% - param    : vector containing general calculation parameters
% - connect  : matrix containing node connectivity
% - coord    : matrix containing node coordinates
% - material : matrix containing material parameters
% - loadNode : matrix containing nodal   loads
% - loadEle  : matrix containing element loads
% - bc       : matrix containing temperature boundary conditions

% - Sorting of the control data to obtain the relevant variables:
%----------------------------------------------------------------

% The letter "n" in front of the matrices (such as nNodes or nUnkn) stands
% for "number of".

nUnknNode      = param(1);              % number of unknowns per node
                                        % (temperature values)

nUnknEleFlux   = param(2);              % number of unknowns per element
                                        % (heat fluxes)

nNodes         = size(coord,1);         % number of nodes
nEle           = size(connect,1);       % number of elements
nNodesEle      = size(connect,2);       % number of nodes per element

nUnknEleTemp   = nUnknNode * nNodesEle; % number of unknowns per element
                                        % (temperature values)
nUnkn          = nUnknNode * nNodes;    % total number of unknowns

nLoadNode      = size(loadNode,1);      % number of nodal   loads
nLoadEle       = size(loadEle,1);       % number of element loads

nBc            = size(bc,1);            % number of temperature boundary
                                        % conditions                                                                            

% - Sorting of the temperature boundary conditions:
%--------------------------------------------------

% The vector iBc contains the nodes that are affected by the boundary
% conditions (first column of the matrix bc). The vector rBc contains the
% value of the respective temperature boundary condition (second column of
% the matrix bc). 

% The letter "i" in iBc stands for "integer", since there are only integers
% in the matrix iBc. The letter "r" in rBc stands for "real" since there
% could be real numbers in the vector rBc.

iBc = zeros(nBc,1);
rBc = zeros(nBc,1);

% Loop over the number of temperature boundary conditions:
for ii = 1 : nBc
    
    iBc(ii,:) = bc(ii,1);
    rBc(ii,:) = bc(ii,2);

end

% - Sorting of the nodal loads:
%------------------------------

% The matrix iLoadNode contains the affected nodes. The vector rLoadNode
% contains the value of the nodal load.

iLoadNode = zeros(nLoadNode,1);
rLoadNode = zeros(nLoadNode,1);

% Loop over the number of nodal loads
for ii = 1 : nLoadNode

    iLoadNode(ii,:) = loadNode(ii,1);
    rLoadNode(ii,:) = loadNode(ii,2);

end

% - Sorting of the element loads:
%--------------------------------

% The vector iLoadEle contains the affected elements. The vector rLoadEle
% contains the value of the element load.

iLoadEle = zeros(nLoadEle,1);
rLoadEle = zeros(nLoadEle,1);

% Loop over the number of element loads:
for ii = 1 : nLoadEle
    
    iLoadEle(ii,:) = loadEle(ii,1);
    rLoadEle(ii,:) = loadEle(ii,2); 
    
end

%%                              SOLUTION
% This part of the program is dedicated to the solution of the linear
% equation system $\textbf{K} \cdot \textbf{v} = \textbf{f}$. $\textbf{K}$
% is the stiffness matrix of the system, $\textbf{f}$ is the load vector
% $\textbf{v}$ is the temperature vector, containing the temperature
% values of all nodes of the system.

% - Building of the stiffness matrix K of the overall system of equations:
%-------------------------------------------------------------------------

% The first step for solving the equation is to build the stiffness matrix
% K.

% Initialising the stiffness matrix K of the overall system of equations:
K = zeros(nUnkn,nUnkn); 

ipos = zeros(1,nUnknEleTemp);

% Loop over the number of elements of the system:
for ii = 1 : nEle
    
    % The routine BuildEleMat is in charge of building the element
    % stiffness matrix KEle for every element of the system. In order to do
    % so, the routine needs the material data of the affected element, the
    % coordinates of the element nodes and number of unknowns per element.
    
    KEle = BuildEleMat_bilinear(material(ii,:), coord(connect(ii,:),:), nUnknEleTemp);
    
    % In the following the matrix KEle needs to be placed at the right
    % position into the global stiffness matrix K. Therefore, the
    % connectivity relating global and local node numbers is needed.
    
    z = 1;
    
    % Loop over the number of nodes per element:
    for m = 1 : nNodesEle
        
        % The vector ipos is a positioning vector. The value
        % connect(i,m) is the number of the actual node. By setting ipos to
        % this value we get to the actual node, which is the place where
        % the first element of KEle is inserted later on.  Therefore, the
        % vector ipos contains (in the right order) the number of rows and
        % columns resp. in which each element of KEle is inserted.
            
        ipos(z) = connect(ii,m);
        z = z + 1;
        
    end
    
    % Insertion of the element matrix into the stiffness matrix K:
    K(ipos,ipos) = K(ipos,ipos) + KEle;
    
end

% sparsity_K = nnz(K)/numel(K);
% disp(sparsity_K);

% Convert to sparse matrix
K = sparse(K);

% - Building of the load vector f: 
%---------------------------------

% In order to build the load vector we need to implement the nodal loads
% and the element loads.

% Initialising the load vector f of the overall system of equations:
f = zeros(nUnkn,1); 

% Insertion of the nodal loads into the load vector f:
for ii = 1 : nLoadNode
     
    % The vector ipos works very much the same as the one helping to build
    % the stiffness matrix K (see above). By setting ipos to
    % iLoadNode(i,1) we "jump" to the node for which we want to implement
    % an element load.
    
	ipos    = iLoadNode(ii,1);
   	f(ipos) = f(ipos) + rLoadNode(ii);
     
end

% Implementation of the element loads:
ipos = zeros(1,nUnknEleTemp);

% Loop over the number of element loads:
for ii = 1 : nLoadEle
    
    % The routine BuildEleLoad is in charge of building the element load 
    % vector fEle for every element of the system. In order to do so, the 
    % routine needs the value of the element load and the coordinates of
    % the element nodes. 
    
    fEle = BuildEleLoad_bilinear( rLoadEle(ii,:), ...
                                coord(connect(iLoadEle(ii,1),:),:));
        
    z = 1;
    
    % Loop over the number of nodes per element
    for m = 1 : nNodesEle
        
        % The already known positioning vector ipos:
        ipos(z) = connect(iLoadEle(ii,1),m);
        z = z + 1;
        
    end
    
    % Insertion of the element load vector fEle into the load vector f
   	f(ipos) = f(ipos) + fEle;

end

% - Implementation of the temperature boundary conditions:
%---------------------------------------------------------

% Boundary conditions are already known temperature values, which no longer
% need to be calculated. Therefore, we can discard the affected equations
% after having regarded the influence of these values on the other
% equations.

% Loop over the number of temperature boundary conditions:
for ii = 1 : nBc
    
    % The positioning vector ipos works the same as the all the ones
    % before. By setting ipos = iBc(i,1) we jump to the node for which we
    % want to implement the boundary condition.
    
    ipos = iBc(ii,1);
    
    % The load vector f is decreased by the multiplication of the value of
    % the given temperature rBc(i) with the respective column K(:,ipos) of
    % the stiffness matrix K.
    
    f(:)    = f(:) - K(:,ipos) * rBc(ii);
    
    % Discarding the equation which is no longer needed implies deleting
    % the respective row out of the stiffness matrix:
    
    K(ipos,:) = zeros(1,nUnkn);
    
    % Since the value of the temperature has already been considered, we
    % can also delete the respective column:
    
    K(:,ipos) = zeros(nUnkn,1);
    
    % Last, the value on the main diagonal needs to be set to 1, so that
    % the stiffness matrix can be inverted. In addition, we insert the
    % value of the given boundary condition into the the load vector f.
    % That way, the number of rows and columns of the stiffness matrix
    % doesn't need to be change, thus saving calculation time.
    
    K(ipos,ipos) = 1;
    f(ipos) = rBc(ii);
     
end

% - Solution of the overall system of equations:
%-----------------------------------------------
v = K \ f;

%%                          POSTPROCESSING


% - Calculation of the heat fluxes:
%----------------------------------

% To obtain the heat fluxes in each node of an element we need to derive
% the now known temperature distribution across each element.

Sigma_tilda    = zeros(nUnknEleFlux,nEle);
ipos = zeros(1,nUnknEleTemp);

% Loop over the number of elements:
for ii = 1 : nEle
           
    % The routine CalcEleFlux is in charge of calculating the element
    % flux vector Sigma. In order to do so, the routine needs the material
    % parameters of the element, the node coordinates, the node
    % temperature values as well as the number of unknown heat fluxes in
    % the element. 
    
    z = 1;
    
    % Loop over the number of nodes per element:
    for m = 1 : nNodesEle        
                    
        ipos(z) = connect(ii,m);
        z = z + 1;
      
    end
    
    Sigma   = CalcEleFlux_bilinear(material(ii,:), coord(connect(ii,:),:),...
                                nUnknEleFlux, v(ipos));

    Sigma_tilda(:,ii) = Sigma;  

end

%% Saving of the current workspace variables

save_workspace_variables = 1;

if save_workspace_variables == 1

    % Creation of the needed path
    Workspace_variables = './Results';
    if isempty(dir(Workspace_variables))
        mkdir(Workspace_variables);
    end

    % Saving
    save([Workspace_variables,'/',InputDataSet], 'coord'        , ...
                                                 'connect'      , ...
                                                 'material'     , ...
                                                 'K'            , ...
                                                 'Sigma_tilda'  , ...
                                                 'v'            , ...
                                                 'f'            , ...
                                                 'nUnknNode'    , ...
                                                 'nUnknEleFlux' , ...
                                                 'nNodes'       , ...
                                                 'nEle'         , ...
                                                 'nNodesEle'    , ...
                                                 'nUnknEleTemp' , ...                                 
                                                 'nUnkn'        , ...
                                                 'nLoadNode'    , ...   
                                                 'nLoadEle'     , ...      
                                                 'nBc'          , ... 
                                                 'bc'           , ...
                                                 'loadNode'     , ...
                                                 'loadEle');

end
