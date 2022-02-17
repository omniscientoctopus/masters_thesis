nEleX = 4*2;
nEleY = nEleX/2;
lambda_x = 4;
lambda_y = 2;

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

% matrix representing element numbers 
% as seen in the mesh
mesh = zeros(nEleY, nEleX);

c = 1;
for i = 1 : nEleX
    for j = 1 : nEleY     
        mesh(nEleY+1-j,i) = c;
        c = c + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Plot Mesh %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot elements 

for i = 1:nEleX*nEleY % run through all elements
    
    % detect left and right corner node numbers
    left_lower_corner_node_number = connect(i,1);
    right_upper_corner_node_number = connect(i,3);
    
    % coordinates of left and right nodes
    left_lower_corner_coord = coord(left_lower_corner_node_number,:);
    right_upper_corner_coord = coord(right_upper_corner_node_number,:);
    
    % side lenghts of elements
    side_lengths = right_upper_corner_coord - left_lower_corner_coord;
    
    % center of the element
    center = left_lower_corner_coord + side_lengths/2;
    
    % plot the elements
    rectangle('Position',[left_lower_corner_coord side_lengths])
    text(center(1),center(2),string(i),'Color','blue')
    axis([0 0.6 0 0.3])
    hold on

end

% Plot node numbers

for i = 1:size(coord(:,1),1) % run through all nodes
    
    % coordinate of node
    node_coordinate = coord(i,:);
    
    % change coordinate of node to improve readability
    position = [node_coordinate(1)+0.005, node_coordinate(2)+0.005];
    
    % position of text on the mesh
    text(position(1),position(2),string(i),'Color','black')
    
end