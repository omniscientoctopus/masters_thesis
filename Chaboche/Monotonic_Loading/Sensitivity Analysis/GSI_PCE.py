"""
Notes
-----

Generalised Sobol Indices(GSI) 
(In general this code can be used for models that do not have a closed form solution but a discretised soltion in time is available) 

Summary:
1.  Consider a time-model: f(t, X) = f(t, x1, x2, x3, x4, x5.....xm) with m parameters.

2.  In order to compute the evolution of GSI in [0, T], we compute GSI at various points in [0, T].

3.  For example, we compute the GSI at t_i in [0, T] for t_i in linspace(0, T, n). We build 'n_qp' PCE surrogates between [0, t_i], 
    which we then numerically integrate, where n_qp is the number of quadrature points.  

4.  However, constructing PCE surrogates at any point t may not always be possible, especially when the closed-form solution is not available.
    Therefore, we use funtion interpolations (of a high-fidelity solution) in such cases to build PCE surrogates at time points where it is not possible 
    to evaluate the model.
    For example in the monotonic loading case of the Chaboche model, the solver uses an implicit scheme with a time step 'delta_t' and therefore solution 
    f(t, X) is only available at multiples of delta_t and we must use interpolations to build PCE surrogates at time points that lie between these multiples.

"""

# Bring packages into the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join('../../..')))

import numpy as np
factorial = np.math.factorial
import matplotlib.pyplot as plt
import scipy.interpolate

import Surrogates.PolynomialChaosExpansion as PCE


def GaussLegendreQuadrature(lowerlimit, upperlimit, number_of_quadrature_points):
    """
    Gauß-Legendre Quadrature for numerical integration in limits: [a, b]

    Exactly integrates polynomial of degree: 2*number_of_quadrature_points-1
    (NumPy package has been tested upto 100 points)

    Parameters
    ----------
    lowerlimit : float
                 lower limit of integration
    lowerlimit : float
                 upper limit of integration
    number_of_quadrature_points : int
                                  Number of quadrature points

    Returns
    -------
        quad_points: ndarray
                     [quad_points, 1]
                     Points at which integrand must be evaluated

        quad_weights: ndarray
                      [quad_points, 1]
                      Weights with which integrand evaluation must be weighted
    """

    # numpy method that provides Gauß-Legendre quadrature points and weights in [-1, 1]
    quad_points, quad_weights = np.polynomial.legendre.leggauss(number_of_quadrature_points)

    # scaling for change of interval from [-1, 1] to [a, b]
    quad_points = ( (upperlimit-lowerlimit)/2 ) * quad_points + ( (upperlimit+lowerlimit)/2 )
    quad_weights = ( (upperlimit-lowerlimit)/2 ) * quad_weights 

    return quad_points, quad_weights


def model_interpolator(input_time_points, func_evaluations, interpolation_points):
    """
    Method to interpolate a function at any point 't' using evalutions

    Parameters
    ----------
    input_time_points : ndarray
                        [n_t, 1]
                        Time points where function was evaluated to generate func_evaluations
    func_evaluations : ndarray
                       [N, n_t]
                       A matrix containing function evalutions at n_t points
                       for N sets of random variables 
    interpolation_points : ndarray
                           [n_ip, 1]
                           Time points at which model must be interpolated.

    Returns
    -------
    f_at_interpolation_points: ndarray
                               [n_ip, 1] 
                               odel evalutions at required time points
    """

    N = func_evaluations.shape[0]
    n_ip = interpolation_points.shape[0]

    # store evaluated values at interpolation points
    f_at_interpolation_points = np.zeros((N, n_ip))

    for i in range(N):

        # We interpolate for each set of random variables
        y_interpolate = func_evaluations[i, :]

        # create callable from class: scipy.interpolate
        # It creates a 'cubic' interpolation using 
        # function evalutions at input_time_points and function value at input time points: y_interpolate
        f_approximate = scipy.interpolate.interp1d(input_time_points, y_interpolate, kind='cubic') 

        f_at_interpolation_points[i, :] = f_approximate(interpolation_points) 

    return f_at_interpolation_points


class GSI_PCE(PCE.PCE_surrogate):
    """
    Compute generalised Sobol indices

    Inherit from class PCE_surrogate from PCE module

    Parameters
    ----------
    t_min : float
            Process start time
    t_max : float
            Process end time
    delta_t : float
              Discretization step in process
    number_quad_points : int
                         Number of quadrature points for numerical integration
    SampleSpace : ndarray
                  [N, m]
                  Design of experiments matrix
                  N: Number of Monte Carlo Samples
                  m: Number of parameters
    func_evaluations : ndarray
                       [n_t, N]
                       n_t: Number of disretizations points in [t_min, t_max] with step delta_t
                       N: Number of Monte Carlo Samples
    total_polynomial_degree : int
                              Total polynomial degree for PCE surrogate
    polynomial_classes_of_random_variables : list of strings
                                              List specifying polynomial class of each random variable
                                              Uniform: 'Legendre'
                                              Normal: 'Hermite'
                                              (Only above RV types supported)
    isoprob_transform : method
                        Isoprobabilistic transform required for PCE surrogate
    """

    def __init__(self, t_min, t_max, delta_t, number_quad_points, SampleSpace, func_evaluations, total_polynomial_degree, 
                polynomial_classes_of_random_variables, isoprob_transform):

        self.X = SampleSpace
        self.total_quad_points = number_quad_points
        self.n = total_polynomial_degree
        self.pcrv = polynomial_classes_of_random_variables 
        self.N_p = len(self.pcrv) # number of parameters
        self.isoprob_transform = isoprob_transform

        # function interpolations
        self.function_evalutions_at_qp  = self.model_evaluations_at_quadrature_points(t_min, t_max, delta_t, func_evaluations.T)

        # set classes of random variables
        for i in range(self.N_p):

            if self.pcrv[i] == 'Hermite':
                self.pcrv[i] = PCE.Hermite

            if self.pcrv[i] == 'Legendre':
                self.pcrv[i] = PCE.Legendre

        #total number of polynomial terms in the PCE
        self.number_of_PCE_terms = int (factorial(self.N_p+self.n) / ( factorial(self.N_p) * factorial(self.n) ))

        # compute all permutations of monomials which have total degree <= n
        self.all_permutations, self.comb_dict = PCE.compute_all_permutations(self.n, self.N_p)


    def model_evaluations_at_quadrature_points(self, t_min, t_max, delta_t, func_evaluations):

        '''
        Method to generate function evaluations at quadrature points required for PCE surrogates 
        and their numerical integration

        '''

        n_t = int((t_max-t_min)/delta_t)

        x_interpolate = np.linspace(t_min, t_max, n_t+1)

        self.quad_points, self.quad_weights = GaussLegendreQuadrature(t_min, t_max, self.total_quad_points)

        return model_interpolator(x_interpolate, func_evaluations, self.quad_points)


    def generalised_Sobol(self, PCE_error):

        # method from parent class: PCE.PCE_surrogate
        first_order_picker, total_order_picker = self.coefficient_pickers()

        # store error at each quad_point
        self.store_PCE_error = np.zeros(self.total_quad_points)

        # store coefficients of polynomial surrogates
        # each column contains polynomial coefficients corresponding to each point(t_m)
        # size = [number_of_PCE_terms, number of quadrature points] '''
        store_beta = np.zeros((self.number_of_PCE_terms, self.total_quad_points))

        for i in range(self.total_quad_points):

            store_beta[:,i] = self.find_coefficients(self.X, self.function_evalutions_at_qp[:,i])

            if PCE_error:

                # error_estimate
                self.store_PCE_error[i] = self.LeaveOneOut(self.X, self.function_evalutions_at_qp[:,i])

        G_first_numerator = first_order_picker @ (store_beta**2) @ self.quad_weights
        G_tot_numerator = total_order_picker @ (store_beta**2) @ self.quad_weights

        G_denominator = (store_beta**2) @ self.quad_weights

        # leave out constant coefficient
        G_denominator = np.sum(G_denominator[1:])

        # first order generalised Sobol index
        # size [N_p, 1]
        G_first = G_first_numerator/G_denominator

        # total generalised Sobol index
        # size [N_p, 1]
        G_tot = G_tot_numerator/G_denominator

        # convert [N_p,1] matrix to [N_p,]
        # easier to handle
        self.G_tot = G_tot.reshape(self.N_p)
        self.G_first = G_first.reshape(self.N_p)

        if PCE_error:
            self.plot_PCE_error()

        return self.G_first, self.G_tot


    def plot_GSI(self, G_first, G_tot):

        'Bar Plots'
        fig, ax = plt.subplots(figsize=(7,4))

        params = [  r'$E$', 
                    r'$\sigma_{y|0}$',
                    r'$Q_{iso}$',
                    r'$b_{iso}$',
                    r'$n$',
                    r'$D$']

        x = np.arange(len(params))
        width = 0.45

        rect1 = ax.bar(x - width / 2, G_first, width, color = 'dodgerblue', label='First Order')
        rect2 = ax.bar(x + width / 2, G_tot, width, color = 'indianred', label='Total Order')

        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.set_ylabel('Generalised Sobol indices')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.legend()

        'Annotate each bar'
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()

                if height <= 1e-8:
                    height = 0

                ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

        autolabel(rect1)
        autolabel(rect2)

        print("Total order effects:",G_tot)
        print("First order effects:", G_first, )
        print("Interaction effects:",G_tot - G_first)

        # plt.savefig('Chaboche_GSI.jpeg', bbox_inches = "tight", dpi = 300)

        plt.show()


    def plot_PCE_error(self):

        # Leave One Out error

        # resolution of plot goverened by number of quadrature points in self.T
        fig, ax = plt.subplots(figsize=(6,5)) 

        ax.plot(self.quad_points, self.store_PCE_error*100, color = 'red')
        ax.set_xlabel('time')
        ax.set_ylabel('PCE Error %')
        ax.set_title('Leave One Out PCE Error %')

        plt.show()