# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
factorial = np.math.factorial

import matplotlib.pyplot as plt
import copy

# custom module
import Surrogates.PolynomialChaosExpansion as PCE

# %% [markdown]
# # Karhunen Loeve Expansion

# %%
class KLE(PCE.PCE_surrogate):
    """
    Karhunen Loeve Expansion surrogate for random processes

    Inherit from class PCE_surrogate from PolynomialChaosExpansion module

    Parameters
    ----------
    SampleSpace : [type]
                  [N, N_p]
                  Samples of random variables
    func_evals : ndarray
                 [n_t, N]
                 n_t : Number of discretizations in time (n_t: (t_max-t_min)/delta_t) + 1) 
                 N : Number of MC samples
                 Model evaluations at SampleSpace points using which surrogate was created
    linspace : list
               Discretisation points in time
               Example : np.linspace(t_min, t_max, n_t+1)
    total_polynomial_order : int
                             total polynomial order
    n_kl : int
           truncation level of KL-expansion
    polynomial_classes_of_random_variables : list of strings
                                              List specifying polynomial class of each random variable
                                              Uniform: 'Legendre'
                                              Normal: 'Hermite'
                                              (Only above RV types supported)
    isoprob_transform : method
                        Isoprobabilistic transform required for PCE surrogate
    PCE_error_flag : bool, optional
                     Flag to compute PCE error for each eigenmode projection approximated by PCE, by default False
    """

    def __init__(self, SampleSpace, func_evals, linspace, total_polynomial_order, n_kl, 
                 polynomial_classes_of_random_variables, isoprob_transform, PCE_error_flag=False):

        super().__init__(total_polynomial_order, polynomial_classes_of_random_variables, isoprob_transform)
        self.X = SampleSpace
        self.N = self.X.shape[0]
        self.func_evals = func_evals
        self.PCE_error_flag = PCE_error_flag
        self.t_linspace = linspace
        self.n_kl = n_kl
        # self.N_p = len(self.pcrv)

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

        # ! Surrogate created upon class initialisation
        # Compute PCE surrogates for each eigenmode
        self.compute_PCE_surrogates()

    def compute_covariance_matrix(self):
        """"Compute covariance matrix of the process"""

        Y = copy.deepcopy(self.func_evals)

        self.Y_mean = np.mean(Y, axis=0) # compute mean at each (t_i)

        # center the process
        self.Y_c = Y - self.Y_mean

        # Covariance matrix
        # size: [N_quad, N_quad]
        # unbiased estimator: 1/(N-1)
        C = (1/(self.N-1))*np.dot(self.Y_c.T, self.Y_c)

        return C

    def eigenvalues_covariance_matrix(self):
        """"
        Compute eigenvalues and eigenvectors of the covariance matrix
        using Nystörm's method

        The trapezoidal rule is used for the quadrature necessary in the Nyström's method
        (Assuming a uniformly discretized solution is available)

        (But also works without a quadrature?!)
        """

        K = self.compute_covariance_matrix()

        # Check uniform discretization
        assert self.t_linspace[1] == self.t_linspace[2] - self.t_linspace[1]

        # Trapezoidal quadrature
        delta_t = self.t_linspace[1]
        N_quad = len(self.t_linspace)

        # Create diagonal matrix with quadrature weights 
        self.W = np.eye(N_quad) * delta_t
        self.W[0,0] = self.W[0,0] / 2
        self.W[-1,-1] = self.W[-1,-1] / 2
        self.W_sqrt = np.diag(np.sqrt(np.diag(self.W)))

        # Eigenvalue Decomposition
        self.eig_values, eig_vectors = np.linalg.eig(self.W_sqrt @ K @ self.W_sqrt)
        eig_vectors = np.linalg.inv(self.W_sqrt) @ eig_vectors

        # Discard imaginary part, as eigenvalues are real
        self.eig_vectors = np.real(eig_vectors)
        self.eig_values = np.real(self.eig_values)

        # # Sort eigenvalues in descending order
        # # reverse sign -> sort in ascending -> reverse sign again
        #  = -np.sort(-eig_values)

        # variance quantified by a given truncation level: n_kl
        r_kl = np.sum(self.eig_values[:self.n_kl])/np.sum(self.eig_values)

        print("Variance quantified by", self.n_kl, "terms =", r_kl)

        # plot of eigenvalues
        plt.semilogy(self.eig_values, '*')
        plt.xlabel('n')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues of covariance matrix')
        plt.show()

    def plot_eigenvectors(self):
        """
        Plot eigenmodes for visualisation

        (Figure placement varies slightly depending on whether
        truncated eigenvalues are even or odd)

        """

        # if even number of truncated eigenvalues
        try: 
            assert(int(self.n_kl%2) == 0)

            fig, ax = plt.subplots(2, int(self.n_kl/2), figsize=(3*self.n_kl, 8))

            # change colors in for loop
            colors = iter(plt.cm.jet(np.linspace(0,1,self.n_kl)))

            for i in np.arange(self.n_kl):

                row = int(i//(self.n_kl/2))
                column = int(i%(self.n_kl/2))

                ax[row][column].plot(self.t_linspace, np.real(self.eig_vectors[:,i]), color=next(colors))

                'Formatting'
                if column == 0:
                    ax[row][column].set_ylabel('model output')
                if row == 1:
                    ax[row][column].set_xlabel('time')
                
                ax[row][column].set_title('EV #' + str(i+1))

            fig.suptitle("First n_kl = " + str(self.n_kl) + " Eigenvectors", fontsize=20, fontweight= "bold")

            plt.show()

        except AssertionError:
            print('Plotting odd number of eigenmodes not handled, please input an even number')

    def compute_projections(self):
        """
        Compute projection of function evalutions on eigenmodes
        """

        self.eigenvalues_covariance_matrix()

        # truncate eigenvector space
        self.trunc_eig_vectors = self.eig_vectors[:, :self.n_kl] 

        # project f_c (bzw. Y_c) on eigenvectors
        # resulting matrix has size [N x n_kl]
        # N evaluations for each of the n_kl vectors
        self.Y_i =  self.Y_c @ self.W @ self.trunc_eig_vectors

    def plot_projections(self):
        """
        Plot the projection of the first realisation on the eigenmodes
        """

        try: 
            assert(int(self.n_kl%2) == 0)

            fig, ax = plt.subplots(2, int(self.n_kl/2), figsize=(3*self.n_kl, 10))

            # change colors in for loop
            colors = iter(plt.cm.jet(np.linspace(0,1,self.n_kl)))

            for i in np.arange(self.n_kl):

                row = int(i//(self.n_kl/2))
                column = int(i%(self.n_kl/2))

                ax[row][column].plot(self.t_linspace, np.real(self.eig_vectors[:,i]), color=next(colors))

                ax[row][column].plot(self.t_linspace, self.Y_c[0,:], color = 'black')

                'Formatting'
                if column == 0:
                    ax[row][column].set_ylabel('model output')
                if row == 1:
                    ax[row][column].set_xlabel('time')

                dot_prod_i = np.around(np.real(self.Y_i[0,i]), 3)

                ax[row][column].set_title(r'<f(t, $\xi^{(1)}$), EV #' + str(i+1) + '> = ' + str(dot_prod_i))

            fig.suptitle("Dot product of first (centered) realisation with first n_kl=" + str(self.n_kl) + " Eigenvectors", fontsize=20, fontweight= "bold")

            plt.show()

        except AssertionError:
            print('Plotting odd number of eigenmodes not handled, please input an even number')


    def compute_PCE_surrogates(self):
        """
        Compute PCE surrogates for the model projections on the eigenmodes

        Additionally, computes Leave One Out error estimate for the modes if 
        PCE_error_flag is True

        """

        self.compute_projections()

        print(f"Number of PCE terms: {self.number_of_PCE_terms}")
        print(f"Number of evaluations needed (empirical estimate): {(self.N_p - 1)*self.number_of_PCE_terms}") # (M-1)*P [Sudret]
        print(f"Number of function evaluations: {self.N}")

        # store coefficients of polynomial surrogates for all eigenmodes
        # each column contains polynomial coefficients corresponding to each eigenmode
        # size = [number_of_PCE_terms, number of eigenmodes]
        self.store_beta = np.zeros((self.number_of_PCE_terms, self.n_kl))

        self.PCE_errors = np.zeros(self.n_kl)

        # compute coefficients of polynomial surrogates (beta) vector for each eigenmode (f_i)
        # f_i (bzw. Y_i) = projection of f (bzw. Y) on eig_vector_i
        for i in range(self.n_kl):

            X_train_test = self.X
            Y_train_test = self.Y_i[:,i]

            self.store_beta[:,i] = self.find_coefficients(X_train_test, Y_train_test)

            # error_estimate
            if self.PCE_error_flag :
                self.PCE_errors[i] = self.LeaveOneOut(X_train_test, Y_train_test)

        if self.PCE_error_flag:
            self.plot_PCE_errors()

    def plot_PCE_errors(self):

        # resolution of plot goverened by number of points in self.T
        fig, ax = plt.subplots(figsize=(6,5)) 

        x = np.arange(start = 1, stop=self.n_kl+1, dtype= int)
        y = self.PCE_errors

        ax.plot(x, y, '-*', color = 'red')
        ax.set_xticks(x)
        ax.set_xlabel('Eigenmode')
        ax.set_ylabel('PCE Error')
        ax.set_title('PCE Error (Leave One Out) in constructing projections on eigenmodes')

        plt.show()

    def surrogate_evaluate(self, X_test):
        """Evaluate the surrogate"""

        X_test = self.isoprob_transform(X_test)

        vanderMonde_test = self.compute_vanderMonde_matrix(X_test)

        # Output: [N, n_kl]
        Y_hat = vanderMonde_test @ self.store_beta

        # Project on eigenmodes/eigenbasis
        Y_surrogate = np.dot(Y_hat, self.trunc_eig_vectors.T)

        return Y_surrogate + self.Y_mean



class KLE_surrogate_evaluate(PCE.PCE_surrogate):

    # Inherit from class PCE_surrogate from PolynomialChaosExpansion module

    def __init__(self, SURROGATE_INFO: dict, isoprob_transform):
        """
        Manually unpickling surrogate object using data saved in JSON 
        
        Recreate surrogate object using precomputed coefficients, eigenvectors stored in a JSON.
        Example: A surrogate was computed and must be re-used without having to recompute surrogate object 

        Parameters
        ----------
        SURROGATE_INFO : dict {
                         ['total_polynomial_order'] : int
                                                      Total polynomial degree for PCE surrogate
                         ['n_kl'] : int
                                    Number of eigenmodes to use
                         ['polynomial_classes_of_random_variables'] : list of strings
                                                                      List specifying polynomial class of each random variable
                                                                      Uniform: 'Legendre'
                                                                      Normal: 'Hermite'
                                                                      (Only above RV types supported)
                         ['store_beta'] : ndarray
                                          [number_of_PCE_terms, n_kl]
                                          coefficients of eigenmode projections in PCE basis
                         ['trunc_eig_vectors'] : [n_t+1, n_kl]
                                                 KLE basis eigenvectors 
                         ['Y_mean'] : ndarray
                                      [n_t+1, 1]
                                      Process mean at each point in time
                              }
        isoprob_transform :  method
                             Isoprobabilistic transform required for PCE surrogate
        """

        self.n = SURROGATE_INFO['total_polynomial_order']
        self.n_kl = SURROGATE_INFO['n_kl']
        self.pcrv = SURROGATE_INFO['polynomial_classes_of_random_variables']
        self.N_p = len(self.pcrv)
        self.store_beta = np.array(SURROGATE_INFO['store_beta'])
        self.trunc_eig_vectors = np.array(SURROGATE_INFO['trunc_eig_vectors'])
        self.Y_mean = np.array(SURROGATE_INFO['Y_mean'])
        self.isoprob_transform = isoprob_transform

        # set classes of random variables
        for i in range(self.N_p):

            if self.pcrv[i] == 'Hermite':
                self.pcrv[i] = PCE.Hermite

            if self.pcrv[i] == 'Legendre':
                self.pcrv[i] = PCE.Legendre

        # total number of polynomial terms in the PCE
        self.number_of_PCE_terms = int (factorial(self.N_p+self.n) / ( factorial(self.N_p) * factorial(self.n) ))

        # compute all permutations of monomials which have total degree <= n
        self.all_permutations, self.comb_dict = PCE.compute_all_permutations(self.n, self.N_p)

    def surrogate_evaluate(self, X_test):

        X_test = self.isoprob_transform(X_test)

        vanderMonde_test = self.compute_vanderMonde_matrix(X_test)

        # Output: [N, n_kl]
        Y_hat = vanderMonde_test @ self.store_beta

        # Project on eigenmodes/eigenbasis
        Y_surrogate = np.dot(Y_hat, self.trunc_eig_vectors.T)

        return Y_surrogate + self.Y_mean