# Global Sensitivity Analysis for Vector-Valued Responses of Mechanical Models

**Abstract**

Mathematical models are approximations of reality which enable us predict the outcome of natural phenomena. However, the inputs to such models may not be known perfectly which results in output uncertainty. Sensitivity analysis aims to quantify the contributions of the stochastic inputs towards the output uncertainty. In this work, we focus on global sensitivity analysis of time-dependent models and use generalised Sobol indices, an extension of the classical variance-based Sobol indices, as a metric to compute their sensitivities. This method is applied to the viscoplastic response of the Chaboche model, to quantify the contribution of the material parameters towards the stress response. In order to expedite computations, two surrogate modeling techniques are employed. Finally,the insights gathered from the sensitivity studies are used to carry out a basic Bayesian calibration of the Young’s modulus.

**Keywords**: Global Sensitivity Analysis, Sobol Indices, Time-dependent processes, Chaboche Model, Surrogate modeling, Karnhunen-Loève Expansion, Polynomial Chaos, Bayesian Calibration, Uncertainty Quantification

## Models

1. [Mechanical Oscillator](Models/Mechanical_Oscillator/mechanical_oscillator.py)
2. [2D Heat Conduction](Models/Heat_Conduction/Heat_2D_FEM.py)
3. [Chaboche Model](Models/Chaboche_Model/ChabocheModel.py)

   **3.1. Monotonic Loading**

      3.1.1. [1D bar (explicit solver)](Models/Chaboche_Model/Examples/Monotonic_Loading/1D_explicit.ipynb)

      3.1.2. [1D bar (implicit solver)](Models/Chaboche_Model/Examples/Monotonic_Loading/1D_implicit.ipynb)

      3.1.3. [2D plate (explicit solver)](Models/Chaboche_Model/Examples/Monotonic_Loading/2D_explicit.ipynb)

      3.1.4. [2D plate (implicit solver)](Models/Chaboche_Model/Examples/Monotonic_Loading/2D_implicit.ipynb)

    **3.2 Cyclic Loading**

    3.2.1. [1D bar (explicit solver)](Models/Chaboche_Model/Examples/Monotonic_Loading/1D_explicit.ipynb)

    3.2.2. [1D bar (implicit solver)](Models/Chaboche_Model/Examples/Monotonic_Loading/1D_implicit.ipynb)
## Surrogate Modeling

1. Local Surrogate: [Polynomial Chaos Expansion](Surrogates/PolynomialChaosExpansion.ipynb)

   Example: [Mechanical Oscillator](Surrogates/Examples/Mechanical_oscillator_PCE.ipynb), [Chaboche Model](Chaboche/Monotonic_Loading/Surrogate/Surrogate_PCE.ipynb)

2. Global Surrogate: [Karhunen-Loeve Expansion](Surrogates/KarhunenLoeveExpansion.py)

   Example: [Mechanical Oscillator](Surrogates/Examples/Mechanical_oscillator_KLE.ipynb), [Chaboche Model](Chaboche/Monotonic_Loading/Surrogate/Surrogate_KLE.ipynb)

## Sensitivity Analysis (for time-dependent processes)
### Sobol Indices 

[Point-wise Sobol Indices](PointwiseSobolIndices.ipynb)

### Generalised Sobol Indices

Example: [Mechanical Oscillator](), [2D Heat Conduction](), [Chaboche Model]()

1. [Point-wise Sobol Indices](PointwiseSobolIndices.ipynb)
2. [Generalised First and Total order indices using the model](GeneralisedSobolIndices.ipynb)
3. [Generalised First and Total order indices using the PCE surrogate](GSI_PCE.ipynb)
4. [Generalised First and Total order indices using the KLE surrogate](GSI_KLE.ipynb)

## Bayesian Calibration

Bayesian calibration is a method for estimating the parameters of a model from data. The method is based on the Bayesian posterior distribution of the model parameters computed using Markov Chain Monte Carlo(MC). The method is applied to the Chaboche model to calibrate the Young’s modulus. The MCMC runs are compared using the `model`, the `PCE surrogate model`, and the `KLE surrogate model`. The MCMC runs are carried out in UQLab in MATLAB.

 1. Bayesian Calibration (with Model): 
 
   Example: [Chaboche Model](Chaboche/Monotonic_Loading/Bayesian_Calibration/Model/chaboche_calibration_elastic.m)
 
 2. Bayesian Calibration (with PCE): 
 
   Example: [Chaboche Model (PCE)](Chaboche/Monotonic_Loading/Bayesian_Calibration/PCE/chaboche_calibration_elastic.m) 

 3. Bayesian Calibration (with KLE):

   Example: [Chaboche Model (KLE)](Chaboche/Monotonic_Loading/Bayesian_Calibration/KLE/chaboche_calibration_elastic.m)

Comparison of the outputs can be found here: [Comparison](Chaboche/Monotonic_Loading/Bayesian_Calibration/Compare_MODEL_KLE_PCE/chaboche_calibration_elastic.m)

## Installation

Create a conda environment

```bash
conda create --name <environment-name>
```

Install Python

```bash
conda install python==3.6
```

The code should also work for 'python >=3.6'. But make sure you install a version of python that is compatible with your version of MATLAB if you are plannning to use files such as Domain-aware Sobol, which call the MATLAB engine. Python and MATLAB version compatibility can be found [here](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf) and a detailed guide to installing the MATLAB engine can be found [here](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). Make sure you install the MATLAB engine in the correct location in your conda environment [see](https://stackoverflow.com/questions/33357739/problems-installing-matlab-engine-for-python-with-anaconda/58560599#58560599).

In case MATLAB is not available, the code could be run (with slight modification) using Octave. However, this has not been tested!!

Additional packages are installed using the package manager `pip`. In case you are using a conda environment, you can install the packages using the command 

```bash
conda install pip
```

```bash
pip install numpy scipy matplotlib seaborn==0.11.1 ipykernel tikzplotlib tqdm
```

## References

1. [Alen Alexanderian, Pierre A. Gremaud, Ralph C. Smith, Variance-based sensitivity analysis for time-dependent processes](https://arxiv.org/abs/1711.08030)

2. [Bruno Sudret. Polynomial chaos expansions and stochastic finite element methods. Kok-Kwang Phoon, Jianye Ching. Risk and Reliability in Geotechnical Engineering, CRC Press, pp.265-300, 2015, 9781482227215. ffhal-01449883](https://hal.archives-ouvertes.fr/hal-01449883/document)

3. [Alen Alexanderian, A brief note on the Karhunen-Loève expansion](https://arxiv.org/abs/1509.07526)

4. [G Blatman and B Sudret. “Sparse polynomial chaos expansions of vector-valued response quantities”. In: Safety, Reliability, Risk and Life-Cycle Performance of Structures and Infrastructures. Ed. by George Deodatis, Bruce Ellingwood, and Dan Frangopol. CRC Press, 2014, pp. 3245–3252. isbn: 978-1-138-00086-5.](https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/risk-safety-and-uncertainty-dam/publications/international-conferences/2013-ICOSSAR-Blatman-Sudret.pdf) 


