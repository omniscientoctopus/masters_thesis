function [coord, connect] = MakeRectMesh2D( coordCornerPoints, nEleXY )
% Function generating finite element mesh coordinate and connectivity data
% for a rectangular domain using 4 node quadrilateral elements.
%
% Input Data:
% --------------
%
% coordCornerPoints: coordinates of corner points
%                    1 x 4 matrix (float)
%                    [ xMin, xMax, yMin, yMax ]
%
% connect:           number of elements in x- and y-direction
%                    1 x 2 matrix (integer)
%                    [ nEleX, nEleY ]
%
%
% Output Data:
% ---------------
%
% coord:   (x,y)-coordinates of nodes
%          nNode x 2 matrix (float)
%          [ x | y ]
%
% connect: element connectivity
%          nEle x 4 (integer)
%          [ nodeA | nodeB | nodeC | nodeD ]
%
% Usage:
% ---------
% [coord, connect] = MakeRectMesh2D( coordCornerPoints, nEleXY )


% Institut fuer Statik | TU Braunschweig
% Beethovenstrasse 51
% 38106 Braunschweig


%% Read out input data
xMin = coordCornerPoints(1);
xMax = coordCornerPoints(2);

yMin = coordCornerPoints(3);
yMax = coordCornerPoints(4);

nEleX = nEleXY(1);
nEleY = nEleXY(2);


% Calculate numbers of nodes and elements
nNodes =  (nEleX + 1) * (nEleY +1);
nEle   =  nEleX * nEleY;

% Get basic mesh parameters:
numNodes = nNodes;
numEle   = nEle;


% Get the number of nodes in each direction for a mesh
% with quadratic shape functions:
nNodesX  = nEleX + 1;
nNodesY  = nEleY + 1;


% The node coordinates can be given by Seeds or will be
% evenly set from Min to Max using linspace:
posX = linspace(xMin, xMax, nNodesX)';
posY = linspace(yMin, yMax, nNodesY)';



%% Create node coordinates
coord = zeros(numNodes, 2); % Initialisation

c = 1;
for i = 1 : nNodesX
    for j = 1 : nNodesY     
        coord(c,:) = [posX(i), posY(j)];
        c = c + 1;
    end
end

%% Create connectivity 
connect = zeros(numEle, 4); % Initialisation

currElem = 1;
for k = 1 : nEleX
    for m = 1 : nEleY
        % Initialization:
        currInzi = zeros(1,4);

        % Get global number for local node 1:
        currInzi(1) = (k-1)*(nEleY+1) + (m-1) + 1;

        % Get further global nodes:
        currInzi(2) = currInzi(1) + (nEleY+1);
        currInzi(3) = currInzi(2) + 1;
        currInzi(4) = currInzi(1) + 1;

        connect(currElem, :) = currInzi;
        currElem = currElem + 1;
    end
end



end %function
