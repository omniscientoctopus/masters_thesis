%% CHABOCHE MODEL CALIBRATION

%% 0 - Introduction
% In this example, the measured stresses in a monotonic 1D tensile test $\sigma_{xx}$ 
% are used to calibrate the Chaboche Model.
% 
% The Chaboche model has 8 parameters. However, for the monotonic loading, we 
% do not consider the kinemtic hardening parameters therefore, we are left with 
% 6 parameters.
% 
% The calibration is carried out with default model/data discrepancy options.
%% 1 - INITIALIZE UQLAB
% Clear all variables from the workspace, set the random number generator for 
% reproducible results, and initialize the UQLab framework:

tic
clearvars
close all

% diary full_bayesian_inversion_output.out % write outputs to file

addpath('~/UQLab/core') % Add UQLab path
uqlab

%% 2 - FORWARD MODEL
% The forward model computes the stress $\sigma_{xx}$ in the material
% 
% The computation is carried out using a python module: 'chaboche_wrapper.py'
% 
% The input variables X = [N, M] 
% 
% $N$: number of realizations
% 
% M : number of input variables
% 
% The 8 variables are given in the following order:
% 1. Young's Modulus: $E \, \text{[N/mm²]}$ 
% 2. Initial Yield Limit: $\sigma_{y|0} \, \text{[N/mm²]}$ 
% 3. Isotropic Asympotote: $Q_{iso} \, \text{[N/mm²]}$ 
% 4. Isotropic growth rate: $b_{iso}$
% 5. Isotrpic exponent: $n_{iso}$
% 6. Isotropic threshold: $D_{iso} \, \text{[N/mm²]}$
% 7. Kinematic Asympotote: $Q_{kin} \, \text{[N/mm²]}$ 
% 8. Kinematic growth rate: $b_{kin}$
% Define the forward model as a UQLab MODEL object:

ModelOpts.mFile = 'chaboche_wrapper_elastic';
ModelOpts.isVectorized = false;

myForwardModel = uq_createModel(ModelOpts);
%% 3 - PRIOR DISTRIBUTION OF THE MODEL PARAMETERS
% The prior information about the model parameters is gathered in a probabilistic 
% model that includes both known and unknown parameters.
% 
% Specify random variables as a UQLab INPUT object:

PriorOpts.Marginals(1).Name = 'E';                  % Young's Modulus 
PriorOpts.Marginals(1).Type = 'Uniform';
PriorOpts.Marginals(1).Parameters = [2e5 2.1e5];    % (N/mm²)

PriorOpts.Marginals(2).Name = 'sigmaY';            % $\sigma_{y|0}$ 
PriorOpts.Marginals(2).Type = 'Uniform';
PriorOpts.Marginals(2).Parameters = [200 400];      % (N/mm²)

PriorOpts.Marginals(3).Name = 'Qiso';              % Q_iso$ 
PriorOpts.Marginals(3).Type = 'Uniform';
PriorOpts.Marginals(3).Parameters = [0 500];        % (N/mm²)

PriorOpts.Marginals(4).Name = 'biso';              % b_iso$ 
PriorOpts.Marginals(4).Type = 'Uniform';
PriorOpts.Marginals(4).Parameters = [0 1e3];        % (-)

PriorOpts.Marginals(5).Name = 'n';              % n_iso$ 
PriorOpts.Marginals(5).Type = 'Uniform';
PriorOpts.Marginals(5).Parameters = [1 6];          % (-)

PriorOpts.Marginals(6).Name = 'sigmaD';            % sigma_D$ 
PriorOpts.Marginals(6).Type = 'Uniform';
PriorOpts.Marginals(6).Parameters = [1 100];        % (N/mm²)

% Print and display priors

myPriorDist = uq_createInput(PriorOpts);

uq_print(myPriorDist)

%% 4- Sensitivity Analysis

% SobolOpts.Type = 'Sensitivity';
% SobolOpts.Method = 'Sobol';
% SobolOpts.Sobol.Order = 1;
% SobolOpts.Bootstrap.Replications = 10;
% SobolOpts.Sobol.SampleSize = 1e3;
% 
% mySobolAnalysisMC = uq_createAnalysis(SobolOpts);
% mySobolResultsMC = mySobolAnalysisMC.Results;
% 
% uq_print(mySobolAnalysisMC)

% uq_figure
% uq_display(mySobolAnalysisMC)
% savefig('Sensitivity_analysis.fig')


%% 5 - MEASUREMENT DATA
% The measurement data consists of $n$ independent measurements of stress.
% 
% The data is stored in the column vector Y = [N, 1] 

DiscrepancyOpts.Type = 'Gaussian';
DiscrepancyOpts.Parameters = 5e-4; % variance

truePara = [2e5, 250, 50, 100, 2, 100];
trueOut = chaboche_wrapper_elastic(truePara)

n = 10; % Number of observations

myFakeData.y = trueOut*ones(n,1) + normrnd(0, sqrt(DiscrepancyOpts.Parameters),[n, 1]);
myFakeData.Name = 'Stress at t = 0.01s';

%% 6 - DISCREPANCY MODEL
% By default, the Bayesian calibration module of UQLab assumes an independent 
% and identically distributed discrepancy $\varepsilon\sim\mathcal{N}(0,\sigma^2)$ 
% between the observations and the predictions for each data.
% 
% The variance $\sigma^2$$of the discrepancy term is by default given a uniform 
% prior distribution:
% 
% $$\pi(\sigma^2) = \mathcal{U}(0,\mu_{\mathcal{Y}}^2) \mathrm{with} \quad  
% \mu_{\mathcal{Y}} = \frac{1}{N}\sum_{j=1}^{N}y_{j}$$

%% 7 - BAYESIAN ANALYSIS

% Solver options
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'AIES'; % Default: Affine Invariant Ensemble Sampler (AIES)
Solver.MCMC.NChains = 4;
Solver.MCMC.Steps = 4e3;

% The options of the Bayesian analysis are specified with the following
% structure:
BayesOpts.Type = 'Inversion';
BayesOpts.Data = myFakeData;
BayesOpts.Discrepancy = DiscrepancyOpts;
BayesOpts.Solver = Solver;

% Run the Bayesian inversion analysis:
myBayesianAnalysis = uq_createAnalysis(BayesOpts);

% Print out a report of the results:
uq_print(myBayesianAnalysis)

save('Model_E_allvars.mat')

toc
% diary off
