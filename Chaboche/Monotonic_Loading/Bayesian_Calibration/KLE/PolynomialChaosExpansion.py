# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

factorial = np.math.factorial

from itertools import permutations, combinations
import matplotlib.pyplot as plt

# %% [markdown]
# # Orthonormal Polynomials
# 
# ## 1._Hermite Polynomials_
# 
# Recursion relation : 
# ${\mathit {He}}_{n+1}(x)=x{\mathit {He}}_{n}(x)-n{\mathit {He}}_{n-1}(x)$
# 
# alternatively,${\mathit {He}}_{n}(x)=x{\mathit {He}}_{n-1}(x)-(n-1){\mathit {He}}_{n-2}(x)$
# 
# Normalisation constant: $\sqrt{n!}$
# 
# Normalised Polynomial: $ \tilde{P_{n}}(x) = \frac{P_{n}(x)} {\sqrt{n!}} $
# 
# ## 2._Legendre Polynomials_
#  
# Recursion relation : 
# $ (n+1)P_{n+1}(x)=(2n+1)xP_{n}(x)-nP_{n-1}(x) $
# 
# alternatively, $ P_{n}(x)=\frac{(2n-1)xP_{n-1}(x)-(n-1)P_{n-2}(x)}{n} $
# 
# Normalisation constant: $\frac{1}{\sqrt{(2n+1)}}$
# 
# Normalised Polynomial: $ \tilde{P_{n}}(x) = P_{n}(x) \cdot \sqrt{(2n+1)} $
# 
# 

# %%
'''
Input:
    n: degree of polynomial
    x: random variable

Output:
    out: normalised polynomial of degree 'n'
         evaluated at 'x'

'''

def Hermite(n, x):
    
    def recurence_relation(n, x):
    
        if n<=0:
            return 1

        else:
            return x*recurence_relation(n-1,x) - (n-1)*recurence_relation(n-2,x)
    
    # normalise
    out = recurence_relation(n,x)/np.sqrt(factorial(n))
    
    return out


def Legendre(n,x):

    def recurence_relation(n,x):    
        
        if n<=0:
            return 1

        else:
            return ((2*n-1)*x*recurence_relation(n-1,x))/n - (n-1) * (recurence_relation(n-2,x)/n)

    # normalise        
    out = recurence_relation(n,x)*np.sqrt(2*n+1)
            
    return out

# %% [markdown]
# # Combinatorics
# 
# This section is divided into three simple sub-sections:
# 
# 1. method: __combination_for_a_given_sum__ </p>
# This function computes the sets of natural numbers (<= n) whose sum is 'n'. </p>
# Example: For n = 3, we get {{1,1,1}, {1,2}, {3}}. The function also discards set with cardinality greater than N_p. </p>
# 
# 2. method:  __find_permutations__ </p>
# This function finds the permutations of the sets from the above function in a set of size N_p. </p>
# Example: For set q = {1,2}, and N_p = 3, we get Q = {(0, 2, 1), (1, 2, 0), (2, 1, 0), (2, 0, 1), (0, 1, 2), (1, 0, 2)}. </p>
#     
# 3. __A for loop__ </p>
#    To iterate over each q from __combination_for_a_given_sum__ to get corresponding Q and store them in the variable 'all_permutations'
#     
# recursion in first function adapted from [this source](https://www.techiedelight.com/print-all-combination-numbers-from-1-to-n/)

# %%
def compute_all_permutations(n, N_p):
    
    '''
    Input :
        n : total degree of polynomial
        N_p : Number of random variables
        
    Output: 
        all_permutations: [{}, {}, ..]
                          All sets sets of size N_p containing all combinations 
                          of whole numbers whose sum is equal 
                          to 'n'.
        comb_dict: A dictionary which lists above combinations systematically
    
    '''

    def combination_for_a_given_sum(n):

        '''
            Input: 
               n : [natural number]
                   sum of natural numbers

            Output: 
            store_combinations: [{}, {}, ..]
                                sets of natural numbers
                                Each set has size <= N_p and 
                                sum of elements = n
        '''

        def compute_Combinations(i, n, out, index):

            '''
            Recursive function to compute all combinations of numbers from 'i' to 'n' having sum 'n'. 
            The 'index' denotes the next free slot in the output list 'out'.

            '''

            # if the sum becomes 'n', store the combination
            if n == 0:
                combination = out[:index]
                store_combinations.append(combination)

            # start from the previous element in the combination till 'n'
            for j in range(i, n + 1):

                # place current element at the current index
                out[index] = j

                # recur with a reduced sum
                compute_Combinations(j, n - j, out, index + 1)

        out = [None] * n

        # store the combinations
        store_combinations = []  

        # store trimmed combinations
        trimmed_combinations = []
        
        # compute all combinations of numbers from 1 to 'n' having sum 'n'
        compute_Combinations(1, n, out, 0)

        # Remove combinations with number of elements greater than N_p
        for combination in store_combinations:
            if len(combination) <= N_p:
                trimmed_combinations.append(combination)
        
        return trimmed_combinations


    def find_permutations(store_combinations):

        store = set() # an empty set to store all permutations for a given sum 'p'

        for i, combination in enumerate(store_combinations):
            
            # create an array 'C' of size 'N_p' filled with 'combination' elements 
            # and padded with zeros if necessary
            C = np.zeros(N_p, dtype = int)
            C[0:len(combination)] = combination
            
            # find all permutations of 'combination' elements
            # 'set' method removes duplicates, as 'combination' sometimes contains repeating elements        
            perm = set(permutations(C))

            # add to previous set and store the permutation 
            store = store.union(store, perm)

        return store


    comb_dict = {}
    
    all_permutations = set()
    
    for i in range(1, n+1):

        store_combinations = combination_for_a_given_sum(i)

        all_permuatations_for_given_sum = find_permutations(store_combinations)

        # store in dictionary 
        comb_dict[i] = all_permuatations_for_given_sum

        # store in set
        all_permutations = all_permutations.union(all_permutations,all_permuatations_for_given_sum)

    return all_permutations, comb_dict

# %% [markdown]
# # PCE Surrogate Class

# %%
class PCE_surrogate():
    
    def __init__(self, number_training_points, total_polynomial_degree, polynomial_classes_of_random_variables, isoprob_transform):
        
        self.training_points = number_training_points
        self.n = total_polynomial_degree
        self.pcrv = polynomial_classes_of_random_variables
        self.N_p = len(self.pcrv)
        self.isoprob_transform = isoprob_transform 

        #total number of polynomial terms in the PCE
        self.number_of_PCE_terms = int (factorial(self.N_p+self.n) / ( factorial(self.N_p) * factorial(self.n) ))
        
        self.all_permutations, self.comb_dict = compute_all_permutations(self.n, self.N_p)

    def print_combinations(self):
        
        print("Number of polynomial terms:", self.number_of_PCE_terms)
        
        for key, value in self.comb_dict.items():

            print("Sum =",key,":",value, '\n')


    # Orthogonal Multivariate Polynomial Basis
    def phi(self, alpha, X):
        
        phi_value = 1

        for i in range(self.N_p):

            phi_value *= self.pcrv[i](alpha[i],X[:,i])

        return phi_value


    def compute_vanderMonde_matrix(self, any_X):
                
        # first column must contain ones
        # other columns will be replaced by polynomial evaluations
        vanderMonde_matrix =  np.ones((any_X.shape[0],self.number_of_PCE_terms))
        
        for i, alpha in enumerate (self.all_permutations):

            vanderMonde_matrix[:,i+1] = self.phi(alpha, any_X)

        return vanderMonde_matrix

    
    def Moore_Penrose_inverse(self, X, Y):
        
        N = np.matmul(X.T,X)
        M = np.matmul(X.T,Y)
        beta = np.linalg.solve(N,M) # Moore Penrose inverse
        
        return beta 
    

    def find_coefficients(self, X, Y):

        # iso_probabilistic transform
        # apply transform to the samples 
        # to convert to the range [-1,1]
        X = self.isoprob_transform(X)

        # compute Vandermonde matrix
        vanderMonde_train =  self.compute_vanderMonde_matrix(X) 
        
        # compute coefficients
        self.beta = self.Moore_Penrose_inverse(vanderMonde_train, Y)

        return self.beta
        

    def evaluate(self, X):

        X_test = self.isoprob_transform(X)

        vanderMonde_test = self.compute_vanderMonde_matrix(X_test)

        Y_hat = vanderMonde_test @ self.beta

        return Y_hat

    # Sobol Indices
    def coefficient_pickers(self):

        coefficient_store = np.zeros((self.N_p, self.number_of_PCE_terms), dtype = int)

        for i, alpha in enumerate (self.all_permutations):
            coefficient_store[:,i+1] = list(alpha)

        # create coefficient pickers : Matrices which multiply coefficients by 0 or 1 based on
        # first order or total Sobol index

        first_order_index_coefficient_picker = np.copy(coefficient_store)

        # Coefficient picker for First order index 
        # set to zero if other coefficients are present
        for i in range(self.number_of_PCE_terms):
            column_i = first_order_index_coefficient_picker[:,i]

            if np.count_nonzero(column_i) > 1:
                first_order_index_coefficient_picker[:,i] = np.zeros((self.N_p))
        
        # set all non-zero elements to 1
        first_order_index_coefficient_picker = np.where(first_order_index_coefficient_picker == 0, 0, 1)
        total_index_coefficient_picker = np.where(coefficient_store == 0, 0, 1)

        return first_order_index_coefficient_picker, total_index_coefficient_picker


    def LeaveOneOut(self, X, Y):

        # total variance 
        denominator = np.var(Y)

        # H matrix
        A = self.compute_vanderMonde_matrix(X)
        N = A.T @ A
        H = A @ np.linalg.solve(N,A.T)
        h = H.diagonal()

        # evalute model
        Y_hat = self.evaluate(X)

        # numerator
        numerator = ((Y - Y_hat) / (1-h))**2
        numerator = numerator.sum()/Y.size

        return numerator/denominator