function InputDataFunc(nEleX, nEleY, lambda_x, lambda_y)

param = [1 8] ; % [(Unknowns temperature per node) (Unknown fluxes per element)]

xMin = 0;
xMax = 0.5;
yMin = 0;
yMax = 0.25;
nEleXY = [ nEleX, nEleY ];
coordCornerPoints = [ xMin, xMax, yMin, yMax ];

% Create 2D mesh
[coord, connect] = MakeRectMesh2D( coordCornerPoints, nEleXY );

% Thermal conductivity in x and y direction, 
% size: (elements,direction)    
material = [lambda_x*ones(nEleX*nEleY,1) lambda_y*ones(nEleX*nEleY,1)] ;   

% Node Load vector
loadNode = [] ;

%Dirichlet Boundary Condition at ends
l_nodes = transpose((linspace(1,nEleY+1,nEleY+1)));
r_nodes = transpose(linspace((nEleX+1)*(nEleY+1)-nEleY,(nEleX+1)*(nEleY+1),nEleY+1));
bc_nodes = [l_nodes;
         r_nodes];

T_left = 20+273;
T_right = 10+273; 

bc_temp = [ones(size(l_nodes))*T_left;
   ones(size(r_nodes))*T_right];

bc = [bc_nodes bc_temp];

% Element load vector
loadEle = zeros(nEleY*nEleX/4,2);

%Source term in upper right corner

j = 1;

for i = 1:size(connect(:,1),1) % run through all elements

%temporary variable to check location of all nodes
temp = coord(connect(i,:),:); 

    % check if x and y coordinate of lower left node of 
    % the element is in the specified region
    if temp(1,1) >= 0.25 && temp(1,2) >=0.125

        % store element number in the loadEle matrix
        loadEle(j,1) = i;

        % apply specified heat flux to the element 
        loadEle(j,2) = 2000; % =0

         j = j+1;
    end
    
end

% stores all variables from the current workspace in a (MAT-file)
save('InputData/CalculationData.mat')
    
end

