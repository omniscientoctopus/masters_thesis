%% FlowControl (function) - Matlab Code
%
% Institut fuer Statik | TU Braunschweig
% Beethovenstrasse 51
% 38106 Braunschweig

function [solution_coord ,solution_connect, solution_u, solution_material,...
          solution_p, temperature_at_inquiry_points] = FlowControl_func(nEleX, nEleY, ...
          lambda_x, lambda_y, Inquiry_points)

%% Paths

path(genpath([pwd,'/FECode'] ),path)
path(genpath([pwd,'/PlotFunctions'] ),path)
path(genpath([pwd,'/InputData'] ),path)

%% Input Data

InputDataFunc(nEleX, nEleY, lambda_x, lambda_y)

InputDataSet = 'CalculationData.mat';

%% Calculation

[solution.coord     ,...
 solution.connect   ,...
 solution.u         ,...
 solution.material  ,...
 solution.p         ] = FEM(InputDataSet);

% convert struct items to variables for python
solution_coord = solution.coord;
solution_connect =solution.connect;
solution_u = solution.u;
solution_material = solution.material;
solution_p = solution.p;

number_inquiry_points = size(Inquiry_points,1);

temperature_at_inquiry_points = zeros(number_inquiry_points,1);

for i = 1:number_inquiry_points

temperature_at_inquiry_points(i) = temperature_at_qp(Inquiry_points(i,:), solution, nEleX, nEleY);

end

%% Plot of temperature distribution and heat fluxes

% Plot_bilinear(solution)

end