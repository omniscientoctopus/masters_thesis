
function output = temperature_at_qp(qp, solution, nEleX, nEleY)

% infer dimensions of plate
xMin = solution.coord(1,1);
xMax = solution.coord(end,1);
yMin = solution.coord(1,2);
yMax = solution.coord(end,2);

% element size in X and Y direction
h_x = (xMax - xMin)/nEleX;
h_y = (yMax - yMin)/nEleY;
h = [h_x, h_y];

% locating element in which QP is present
qp_element_location = ceil(qp./h);

% computing element number of above location
qp_element_number = (qp_element_location(1) - 1) * nEleY + qp_element_location(2);  

% extract node numbers of element
connecting_nodes = solution.connect(qp_element_number,:);

% extract coordinates of above nodes
coord_connecting_nodes = solution.coord(connecting_nodes,:)';

X = coord_connecting_nodes(1,:); 
Y = coord_connecting_nodes(2,:);

syms a b 

%ansatz functions to transfer from local to global coordinate system
Omega = [ 0.25*(a-1)*(b-1);
         -0.25*(a+1)*(b-1);
          0.25*(a+1)*(b+1);
         -0.25*(a-1)*(b+1)];

% equations to infer local coordinates
eq1 = Omega(1)*X(1) + Omega(2)*X(2) + Omega(3)*X(3) + Omega(4)*X(4) - qp(1);
eq2 = Omega(1)*Y(1) + Omega(2)*Y(2) + Omega(3)*Y(3) + Omega(4)*Y(4) - qp(2);

% solve to obtain local coordinates
% add constraint to ensure solution is within [-1,1]^2
[val_a, val_b] = solve([eq1 eq2], a>=-1, a<=1, b>=-1, b<=1, [a b]);

% temperature at nodes of the current element
T_at_nodes = solution.u(connecting_nodes);

% temperature at any point in the element
temperature_at_a_point =  Omega(1)*T_at_nodes(1) + Omega(2)*T_at_nodes(2) ...
                        + Omega(3)*T_at_nodes(3) + Omega(4)*T_at_nodes(4);

% evaluate temperature at qp
T = matlabFunction(temperature_at_a_point);

output = vpa(T(val_a, val_b));

end
