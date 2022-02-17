%% FlowControl (script) - Matlab Code
%
% Institut fuer Statik | TU Braunschweig
% Beethovenstrasse 51
% 38106 Braunschweig


%% 
clc, clear, close all

%% Paths

path(genpath([pwd,'/FECode'] ),path)
path(genpath([pwd,'/PlotFunctions'] ),path)
path(genpath([pwd,'/InputData'] ),path)


%% Input Data

nEleX = 20;
nEleY = nEleX*0.5;

lambda_x = 40;
lambda_y = 2; %2

InputDataFunc(nEleX, nEleY, lambda_x, lambda_y)

InputDataSet = 'CalculationData.mat';

%% Calculation

[solution.coord     ,...
 solution.connect   ,...
 solution.u         ,...
 solution.material  ,...
 solution.p         ] = FEM(InputDataSet);

% qp = [0.05635083 0.02817542];
% 
% output = temperature_at_qp(qp, solution, nEleX, nEleY)

%% Plot of temperature distribution and heat fluxes

Plot_bilinear(solution)