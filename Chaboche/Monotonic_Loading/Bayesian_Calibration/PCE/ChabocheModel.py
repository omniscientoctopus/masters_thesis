# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import time 
import copy
from tqdm import tqdm # module for progress bar (https://pypi.org/project/tqdm/)
import itertools
import multiprocessing 

# %% [markdown]
# # Random Variables
# 
# **8** Random variables ($\nu$ if fixed to 0.3)
# 
# 1. Young's Modulus: $E$ N/mm²
# 
# 2. Initial Yield Limit: $\sigma_{y|0}$ N/mm²
# 
# ### Isotropic Parameters
# 
# 3. Isotropic Asympotote: $Q_{iso}$ N/mm²
# 
# 4. Isotropic growth rate: $b_{iso}$
# 
# 5. Isotrpic exponent: $n_{iso}$
# 
# 6. Isotropic threshold: $D_{iso}$ N/mm²
# 
# ### Kinematic Parameters
# 
# 7. Kinematic Asympotote: $Q_{kin}$ N/mm²
# 
# 8. Kinematic growth rate: $b_{kin}$

# %%
def chaboche_samples(number_training_points):

    '''
    Continuous Uniform Distribution

    Inputs:
        N = integer
            Number of samples 


    Outputs:
    samples =   [N, 8] matrix
                0: Young's Modulus
                1: initial yield limit
                2: Q_iso
                3: b_iso
                4: n_iso
                5: sigma_D
                6: Q_kin
                7: b_kin
                Each column contains N realisations of each random variable  
                Each row contains one set of realisations

    '''

    N = number_training_points

    B = [[2e5, 2.1e5],
         [200, 400],
         [0, 500],  # [0,500]
         [0, 1e3],  # [0, 1e3]
         [1, 6],    # [1,6]
         [1, 100],  # [1,100]
         [0, 500],
         [0, 1e3]]

    # Material properties
    Youngs_Modulus = np.random.uniform(B[0][0], B[0][1], (N,1))
    initial_yield_limit = np.random.uniform(B[1][0], B[1][1], (N,1)) 

    ## Isotropic hardening
    Q_iso = np.random.uniform(B[2][0], B[2][1], (N,1))
    b_iso = np.random.uniform(B[3][0], B[3][1], (N,1))
    n_iso = np.random.uniform(B[4][0], B[4][1], (N,1))
    sigma_D = np.random.uniform(B[5][0], B[5][1], (N,1))

    ## Kinematic hardening
    Q_kin = np.random.uniform(B[6][0], B[6][1], (N,1))
    b_kin = np.random.uniform(B[7][0], B[7][1], (N,1))

    return np.hstack((Youngs_Modulus, initial_yield_limit, Q_iso, b_iso, n_iso, sigma_D, Q_kin, b_kin))


# %%
def chaboche_uniform_isoprob_transform(any_X):

    '''
    Inputs:
                0: Young's Modulus
                1: initial yield limit
                2: Q_iso
                3: b_iso
                4: n_iso
                5: sigma_D
                6: Q_kin
                7: b_kin
        samples = [N, 8] matrix, uniform distribution


    Outputs:
    samples = [N, 8] matrix
              Mapping uniform distribution U(a,b) onto U(-1,1)

    '''

    B = [[2e5, 2.1e5],
         [200, 400],
         [0, 500],
         [0, 1e3],
         [1, 6],
         [1, 100],
         [0, 500],
         [0, 1e3]]

    temp = copy.deepcopy(any_X)

    sum = (np.sum(B, axis = 1)/2).reshape(1, -1)

    diff = (np.diff(B, axis = 1)/2).reshape(1, -1)

    temp = ( temp - sum ) / diff
    
    return temp

def chaboche_uniform_isoprob_monotonic_transform(any_X):

    '''
    Inputs:
                0: Young's Modulus
                1: initial yield limit
                2: Q_iso
                3: b_iso
                4: n_iso
                5: sigma_D
        samples = [N, 6] matrix, uniform distribution


    Outputs:
    samples = [N, 6] matrix
              Mapping uniform distribution U(a,b) onto U(-1,1)

    '''

    B = [[2e5, 2.1e5],
         [200, 400],
         [0, 500], # [0,500]
         [0, 1e3], # [0,1e3]
         [1, 6], # [1, 6]
         [1, 100]] # [1,100]

    temp = copy.deepcopy(any_X)

    sum = (np.sum(B, axis = 1)/2).reshape(1, -1)

    diff = (np.diff(B, axis = 1)/2).reshape(1, -1)

    temp = ( temp - sum ) / diff

    return temp

def chaboche_uniform_isoprob_twoparams_transform(any_X):

    '''
    Inputs:
                0: n_iso
                1: sigma_D
        samples = [N, 2] matrix, uniform distribution


    Outputs:
    samples = [N, 2] matrix
              Mapping uniform distribution U(a,b) onto U(-1,1)

    '''

    B = [[1, 6], # [1, 6]
         [1, 100]] # [1,100]

    temp = copy.deepcopy(any_X)

    sum = (np.sum(B, axis = 1)/2).reshape(1, -1)

    diff = (np.diff(B, axis = 1)/2).reshape(1, -1)

    temp = ( temp - sum ) / diff

    return temp


# %% [markdown]
# # Implicit scheme 
# 
# $\left[\begin{array}{c}
# \boldsymbol{F} \\
# g\\
# \boldsymbol{H} \end{array}\right] =
# \left[\begin{array}{c}
# \boldsymbol{\varepsilon}_{i+1} - \boldsymbol{\varepsilon}_{i} - \Delta t \cdot\left\{\boldsymbol{E}^{-1} \cdot \big( \frac{\boldsymbol{\sigma}_{i+1}-\boldsymbol{\sigma}_{i}}{\Delta t}\big )+\frac{1}{\sigma_{v, i+1}} \cdot\left\langle\frac{\sigma_{e x, i+1}}{D}\right\rangle^{n} \cdot \dot{\boldsymbol{\varepsilon}}_{0} \cdot \boldsymbol{M} \cdot (\boldsymbol{\sigma}_{i+1} - \boldsymbol{X}_{i+1})\right\} \\
# K_{i+1}-K_{i}-\Delta t \cdot b_{iso} \cdot\left(Q_{iso}-K_{i+1}\right) \cdot\left\langle\frac{\sigma_{e x, i+1}}{D}\right\rangle^{n} \\
# \boldsymbol{X}_{i+1} - \boldsymbol{X}_{i} - \Delta t \cdot b_{kin} \cdot \Big (\frac{2}{3} Q_{kin} \cdot \frac{1}{\sigma_{v, i+1}} \boldsymbol{M} \cdot (\boldsymbol{\sigma}_{i+1} - \boldsymbol{X}_{i+1}) - \boldsymbol{X}_{i+1} \Big) \cdot \left\langle\frac{\sigma_{e x, i+1}}{D}\right\rangle^{n} 
# \end{array}\right]$

# %%
class ChabocheModel():

    '''
    Chaboche Model: 
    - Constitutive model for metals (like steel) subjected to cyclic loading
    - Restricted to simulating low cycle fatigue and infinitesimal strains
    - Combines isotropic and kinematic hardening (and more)

    Capabilites of this class:
    - 1D and 2D problems
    - Monotonic and cyclic loading

    '''

    def __init__(self, problem_type, loading_type, max_applied_strain, time_to_max_strain):

        '''
        Input: 
        problem_type:       str
                            '1D' or '2D'

        loading_type:       str
                            'monotonic' or 'cyclic'

        max_applied_strain:[scalar]
                            maximum strain applied in X direction

        time_to_max_strain: [scalar]
                            time required to reach the maximumstrain applied

        '''

        self.loading_type = loading_type
        self.problem_type = problem_type

        self.max_applied_strain = max_applied_strain

        self.time_to_max_strain = time_to_max_strain

        # strain-rate vector [3,1]
        # maximum strain in x direction
        self.strain_rate = max_applied_strain/time_to_max_strain

        if problem_type == '2D':

            self.M = np.array( [[1, -0.5, 0],
                                [-0.5, 1, 0],
                                [0, 0, 3] ] )

    def loading(self, type_of_loading):

        # copy variables for local scope
        strain_rate = self.strain_rate # [1,1]
        max_applied_strain = self.max_applied_strain # [1,1]
        T = self.time_to_max_strain

        if type_of_loading == 'monotonic':

            def strain(t):
                return strain_rate*t if t < T else max_applied_strain

            def func_strain_rate(t):
                return strain_rate if t < T else 0

        if type_of_loading == 'cyclic':

            def strain(t):

                '''
                Stain function 

                Input:
                    t : time 
                
                Output: [1,1] scalar
                        strain at time t
                '''

                # initial loading to +max_applied_strain in [0, T]
                if t <= T:
                    return strain_rate*t
                
                # rest of the cycle
                # unloading to 0 in [T, 2T]
                # loading to -max_applied_strain in [2T, 3T]
                # unloading to 0 in [3T, 4T]
                else: 
                    # compute int of time/time_to_max
                    # if odd, set as lower limit
                    # else int(t/T) -1 will be the lower limit
                    lower_limit = int(t/T) if int(t/T)%2 != 0 else int(t/T) -1

                    # sign of the curve at any point
                    # Hint: Arithmetic Progression with starting point 1 and common difference 4
                    sign = 1 if (lower_limit-1)%4 == 0 else -1

                    return (sign)*max_applied_strain -(sign)*strain_rate*(t-lower_limit*T)


            def func_strain_rate(t):

                '''
                Stain rate function 
                
                Inputs:
                    t : time
                
                Output: [scalar]
                        strain rate at time t
                '''

                # loading 1
                if t <= T:
                    return strain_rate
                
                else:
                    # compute int of time/time_to_max
                    # if odd, set as lower limit
                    # else int(t/T) -1 will be the lower limit
                    lower_limit = int(t/T) if int(t/T)%2 != 0 else int(t/T) - 1

                    # sign of the curve at any point
                    # Hint: Arithmatic Progression with starting point 1 and common difference 4
                    sign = 1 if (lower_limit-1)%4 == 0 else -1

                    return -(sign)*strain_rate

        return strain, func_strain_rate


    def equivalent_stress(self, sigma, backstress):

        '''
        von Mises equivalent stress for a given stress state
        after accounting for the backstress(es)
        
        Input:
        stress :    [scalar]
                    [sigma_xx, sigma_yy, sigma_xy]

        backstress: [scalar]
                    [X_xx, X_yy, X_xy]
        
        Output:     [scalar]
                    equivalent stress
        '''

        sigma_xx = sigma - backstress

        return np.sqrt(sigma_xx**2)

    def plastic_multiplier(self, sigma_D, sigma_y, sigma, backstress, K, power):

        '''
        Plastic multiplier 

        Input:
        sigma_D:    [scalar]
                    Material parameter

        sigma_y:    [scalar]
                    initial_yield limit

        stress :    [scalar]
                    [sigma_xx, sigma_yy, sigma_xy]

        backstress: [scalar]
                    [X_xx, X_yy, X_xy]

        K:          [scalar]
                    isotropic hardening
        
        power:      [scalar]
                    exponent in the term
        
        Output:     [scalar]
                    Plastic multiplier

        '''

        sigma_ex = self.equivalent_stress(sigma, backstress) - (sigma_y + K)

        return (sigma_ex/sigma_D)**power if sigma_ex > 0 else 0

    'Implicit scheme'

    def system_of_functions(self, Z, delta_t, sigma_i, t0, K_i, X_i):

        '''
        System of non-linear equations arising from the implicit scheme

        Input: 

        Z :     [3, 1]
                [sigma [1,1], K[1,1], X[1,1]] @ (t_i + 1)

        delta_t:[scalar]
                time step

        sigma_i:[1, 1]
                stresses at t_i

        t0 :    [scalar]
                time t_i

        K_i :   [scalar]
                isotropic hardness at t_i

        X_i :   [1, 1]
                backstresses at t_i

        Output: [3, 1]
                system of non-linear eqs evaluated at t_i+1
        '''

        sigma_i_plus_1 = Z[0]
        K_i_plus_1 = Z[1]
        X_i_plus_1 = Z[2]

        # Plane stress
        fa = self.strain(t0 + delta_t) - self.strain(t0) # [scalar]
        fb = (1/self.E) * ((sigma_i_plus_1-sigma_i)/delta_t) # [scalar]
        fc = 1/self.equivalent_stress(sigma_i_plus_1, X_i_plus_1) # [scalar]
        fd = self.plastic_multiplier(self.sigma_D, self.sigma_y, sigma_i_plus_1, X_i_plus_1, K_i_plus_1, self.n_iso) # [scalar]
        fe = (sigma_i_plus_1 - X_i_plus_1) # [scalar]

        F = fa - delta_t * (fb + fc * fd * fe) # [3,1]

        g = K_i_plus_1 - K_i - delta_t * self.b_iso * (self.Q_iso - K_i_plus_1) * fd # [scalar]

        H = X_i_plus_1 - X_i - delta_t * self.b_kin * ( (2/3) * self.Q_kin * fc * fe - X_i_plus_1) * fd # [scalar]

        return np.array([F, g, H]) # [3,1]

    def derivative_system_of_equations(self, Z_prime, delta_t, sigma_i, t0, K_i, X_i):

        sigma_i_plus_1 = Z_prime[0]
        K_i_plus_1 = Z_prime[1]
        X_i_plus_1 = Z_prime[2]

        ## common terms

        # functions 
        fc = 1/self.equivalent_stress(sigma_i_plus_1, X_i_plus_1) # [scalar]
        fd = self.plastic_multiplier(self.sigma_D, self.sigma_y, sigma_i_plus_1, X_i_plus_1, K_i_plus_1, self.n_iso) # [scalar]
        fe = sigma_i_plus_1 - X_i_plus_1 # [scalar]

        dsigmaMac_dsigmaex = self.n_iso * (1/self.sigma_D) * self.plastic_multiplier(self.sigma_D, self.sigma_y, sigma_i_plus_1, X_i_plus_1, K_i_plus_1, self.n_iso-1) #[scalar]

        # derivatives wrt K
        dsigmaex_dK = -1 # [scalar]
        dfd_dK = dsigmaMac_dsigmaex * dsigmaex_dK # [scalar]

        # direct derivatives
        dsigmav_dsigma = fc * (sigma_i_plus_1 - X_i_plus_1) # [scalar]
        dfc_dsigma = - (fc**2) * dsigmav_dsigma # [scalar]
        dsigmaex_dsigma = dsigmav_dsigma # [scalar]
        dfd_dsigma = dsigmaMac_dsigmaex * dsigmaex_dsigma # [scalar]
        dfe_dsigma = 1 # [scalar]

        dsigmav_dX = - dsigmav_dsigma # [scalar]
        dfc_dX = - (fc**2) * dsigmav_dX # [scalar]
        dsigmaex_dX = dsigmav_dX # [scalar]
        dfd_dX = dsigmaMac_dsigmaex * dsigmaex_dX # [scalar]
        dfe_dX = -1 # [scalar]

        # Jacobian Assembly
        J = np.zeros((3,3))

        # derivatives of vector function F
        
        # [1,1]
        J[0, 0] = - delta_t* ( (1/delta_t) * (1/self.E) + \
                                fc * fd * dfe_dsigma + \
                                fd * fe * dfc_dsigma + \
                                fc * fe * dfd_dsigma)
        # [1,1]
        J[0, 1] = - delta_t * fc * dfd_dK * fe
        
        # [1,1]
        J[0, 2] = - delta_t*(fc * fd * dfe_dX + \
                                fd * fe * dfc_dX + \
                                fc * fe * dfd_dX )

        # derivatives of scalar function g

        # [1,1]
        J[1, 0]  = - delta_t * self.b_iso * (self.Q_iso - K_i_plus_1) * dfd_dsigma

        # [1,1]
        J[1, 1] = 1 - delta_t * self.b_iso * (-1*fd + (self.Q_iso - K_i_plus_1) * dfd_dK)

        # [1,1]
        J[1, 2] = - delta_t * self.b_iso * (self.Q_iso - K_i_plus_1) * dfd_dX


        # derivatives of vector function H

        # [1,1]
        J[2, 0] = - delta_t * self.b_kin * \
                    ( ((2/3) * self.Q_kin * fc * fe - X_i_plus_1) * dfd_dsigma + \
                        (2/3) * self.Q_kin * (fe * dfc_dsigma + fc * dfe_dsigma) * fd)

        # [1,1]
        J[2, 1] = - delta_t * self.b_kin * ( (2/3) * self.Q_kin * fc * fe - X_i_plus_1) * dfd_dK

        # [1,1]
        J[2, 2] = 1 - \
                delta_t * self.b_kin * \
                ( ((2/3) * self.Q_kin * fc * fe - X_i_plus_1) * dfd_dX + \
                    ((2/3) * self.Q_kin * fe * dfc_dX +  (2/3) * self.Q_kin * fc * dfe_dX - 1) * fd)

        return J

    def Time_Integration(self, parameters, v, solver_type, t_min, t_max, delta_t, display_time):

        # Parameters
        Young_Mod = parameters[0]
        self.E = Young_Mod
        self.sigma_y = parameters[1]
        self.Q_iso = parameters[2]
        self.b_iso = parameters[3]
        self.n_iso = parameters[4]
        self.sigma_D = parameters[5]
        self.Q_kin = parameters[6]
        self.b_kin = parameters[7]

        # loading 
        self.strain, self.func_strain_rate = self.loading(self.loading_type)

        # time integration from tmin to tmax [s]
        N = int((t_max-t_min)/delta_t)
        self.T = np.linspace(t_min, t_max, N+1)

        start_time = time.time()

        '''
        Implicit Scheme
        
        '''

        if solver_type == 'implicit':

            # initialise variables
            t_0 = 0
            K_0 = 0
            solver_starting_point = np.array([[1], [1], [0]]) # [3,1]
            sigma_0 = 1e-15
            X_0 = 0
            parameters = (delta_t, sigma_0, t_0, K_0, X_0)

            store_values = np.zeros((N+1,1))
            strain_linspace = np.zeros((N+1,1))

            for i in range(N):

                start_time1 = time.time()

                # Method lm solves the system of nonlinear equations in a least squares sense using a modification of the 
                # Levenberg-Marquardt algorithm as implemented in MINPACK
                # compute quantities at time t_{i+1}
                answer = scipy.optimize.root(self.system_of_functions, solver_starting_point, jac=self.derivative_system_of_equations, 
                                            args = parameters, method='lm', 
                                            options={'col_deriv': 0, 
                                                    'xtol': 4e-8, 
                                                    'ftol': 4e-8, 
                                                    'gtol': 0.0, 
                                                    'maxiter': 0, 
                                                    'eps': 0.0, 
                                                    'factor': 10, 
                                                    'diag': None}) 

                # notify if scipy.optimize.root fails
                if answer.success == False:
                    print("Fail at {0} after {1} because {2}".format(t_0 + (i+1)*delta_t, np.round(time.time() - start_time1, 4), answer.message)) 

                # update parameters
                # quantities at time t_{i+1} become quantities at t_{i}
                sigma_i = answer.x[0]
                K_i = answer.x[1]
                X_i = answer.x[2]
                t_i = t_0 + (i+1)*delta_t # previous time step
                parameters = (delta_t, sigma_i, t_i, K_i, X_i)

                # guess for starting point for root finding algorithm using explicit scheme
                ## common terms
                fc = 1/self.equivalent_stress(sigma_i, X_i) # [scalar]
                fd = self.plastic_multiplier(self.sigma_D, self.sigma_y, sigma_i, X_i, K_i, self.n_iso) # [scalar]

                sigma_i_guess = sigma_i + delta_t * self.E * ( self.func_strain_rate(t_i) - fc * fd * (sigma_i - X_i) )
                K_i_guess = K_i + delta_t * self.b_iso * (self.Q_iso-K_i) * fd
                X_i_guess = X_i + delta_t * self.b_kin * ( (2/3) * self.Q_kin * fc * (sigma_i - X_i) - X_i ) * fd

                # Is stress state undergoing plasticity? 
                if fd > 0:
                    # If yes, is Plastic multiplier > strain rate?
                    if fd - self.strain_rate > 1e-8:
                        print("Plastic multiplier error:", fd - self.strain_rate)

                # stacking initial point guess
                solver_starting_point = np.array([[sigma_i_guess], [K_i_guess], [X_i_guess]]) # [3,1]

                # store equivalent stress and isotropic hardening
                store_values[i+1, 0] = sigma_i
                strain_linspace[i+1] = self.strain(t_i)


        '''
        Explicit Scheme

        '''

        if solver_type == 'explicit':

            # initialise variables
            t_i = 0
            K_i = 0
            X_i = 0
            sigma_i = 1e-15

            store_values = np.zeros((N+1,1))
            strain_linspace = np.zeros((N+1,1))

            for i in range(N):

                ## common terms
                fc = 1/self.equivalent_stress(sigma_i, X_i) # [scalar]
                fd = self.plastic_multiplier(self.sigma_D, self.sigma_y, sigma_i, X_i, K_i, self.n_iso) # [scalar]

                sigma_i = sigma_i + delta_t * self.E * ( self.func_strain_rate(t_i) - fc * fd * (sigma_i - X_i) )

                K_i = K_i + delta_t * self.b_iso * (self.Q_iso-K_i) * fd

                X_i = X_i + delta_t * self.b_kin * ( (2/3) * self.Q_kin * fc * (sigma_i - X_i) - X_i ) * fd

                # store stress, isotropic, kinematic hardening
                store_values[i+1, 0] = sigma_i
                strain_linspace[i] = self.strain(t_i)

                # update variables
                t_i += delta_t

        if display_time == True:
            print(f"Computed in {np.round(time.time()-start_time,2)}s")

        return store_values


    def stress_plot(self, t_min, t_max, delta_t, stress_values):

        N = int((t_max-t_min)/delta_t)
        T = np.linspace(t_min, t_max, N+1)

        fig, ax = plt.subplots(figsize=(5,4))

        ax.plot(T, stress_values)

        ax.set_title(r'1D bar, $\sigma_{11}$ vs time')
        ax.set_xlabel('time [s]')
        ax.set_ylabel(r'$\sigma_{11} [N/mm^2]$')
        ax.grid()

        plt.show()


# %%
def generate_model_evaluations(model, t_min, t_max, delta_t, method, SampleSpace, parallel_computation=True):

    ''''
    Chaboche Model evaluations for a set of random parameters 

    Inputs:
            model:  Object 
                    Object of 'ChabocheModel' class from above, with specified strain rate.

            t_min:  [Scalar]
                    Start of time integration, usually 0

            t_max:  [Scalar]
                    End of time integration

            delta_t:    [Scalar] 
                        Integration time step

            method: [string]
                    Integration scheme: 'explicit' or 'implicit'

            SampleSpace: [N, 8] matrix
                        0: Young's Modulus
                        1: initial yield limit
                        2: Q_iso
                        3: b_iso
                        4: n_iso
                        5: sigma_D
                        6: Q_kin
                        7: b_kin
                        Each column contains N realisations of each random variable  
                        Each row contains one set of realisations
    
            parallel_computation:   [Boolean]
                                    Default: True
                                    If True, model evaluations will be carried out in parallel.
                                    
    Outputs:
        func_evaluations:   [n_t+1, N]
                            Model evaluations for desired set of parameters
                            n_t: (t_max-t_min)/delta_t

    '''

    number_of_realizations = SampleSpace.shape[0]

    n_t = int((t_max-t_min)/delta_t)

    func_evaluations = np.zeros((n_t+1, number_of_realizations))

    if parallel_computation == True:

        cpu_count = multiprocessing.cpu_count() # cpu count = number of cores * logical cores

        # create an iterable for the starmap
        # repeat input args and zip
        iterable = zip( SampleSpace, 
                        itertools.repeat((0.3), number_of_realizations), 
                        itertools.repeat(method, number_of_realizations),
                        itertools.repeat(t_min, number_of_realizations),
                        itertools.repeat(t_max, number_of_realizations),
                        itertools.repeat(delta_t, number_of_realizations),
                        itertools.repeat(False, number_of_realizations)
                    )

        # we use 0.3 * total number of cpus
        # could be more or less based on use case
        cpus_to_use = int(cpu_count*0.8)
        with multiprocessing.Pool(cpus_to_use) as pool:

            # pool.starmap( f(x), ( (x1,x2,x3...), (x1,x2,x3...) ) )
            # returns a list of evaluations
            # cannot pass a wrapper!
            list_func_evaluations = pool.starmap(model.Time_Integration, iterable)

        func_evaluations = np.hstack(list_func_evaluations)


    else: # series computation

        # progress bar 'tqdm' wrapped around iterable 
        for i in tqdm(range(number_of_realizations), ncols=100):

            # solve problem for a set of parameters
            solution = model.Time_Integration(SampleSpace[i, :], 0.3, method, t_min, t_max, delta_t, display_time=False)

            func_evaluations[:,i] = solution.flatten()

    return func_evaluations
