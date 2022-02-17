# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

# %% [markdown]
# # Mechanical Oscillator

# %%
def mech_oscillator(x,t):
    
    '''
    Inputs:
        x = [N,n] matrix
            N samples of each of the 'n' input-parameters 
            Each row contains one set of realisations
        
    Outputs:
        Y = [N,1] vector
            N outputs corresponding to the N set of realisations

    '''
    
    alpha = x[:,0:1] 
    beta = x[:,1:2] 
    l = x[:,2:3] 

    Y = l * np.exp(-alpha*t) * (np.cos(beta*t) + alpha/beta * np.sin(beta*t)) # compute y for each 'instance'
    
    return Y

# %% [markdown]
# # Sampling Random Variables

# %%
def mech_oscillator_samples(N):

    '''
    Inputs:
        N = integer
            Number of samples 

        
    Outputs:
    samples = [N, 3] matrix
              Each row contains one set of realisations

    '''
    
    a = [3/8, 5/8]
    b = [10/4, 15/4]
    c = [-5/4, -3/4]

    alpha = np.random.uniform(a[0], a[1], (N,1))
    beta = np.random.uniform(b[0], b[1], (N,1))
    l = np.random.uniform(c[0], c[1], (N,1))
    
    samples = np.hstack((alpha,beta,l))
    
    return samples

# %% [markdown]
# # Isoprobabilistic Transform
# 
# Convert a uniform distribution U(a,b) to U(-1,1) so that the polynomials in the Polynomial Chaos Expansion are normalised

# %%
def mech_oscillator_isoprob_transform(any_X):
    
    a = [3/8, 5/8]
    b = [10/4, 15/4]
    c = [-5/4, -3/4]
    
    temp = np.copy(any_X)
    
    temp[:,0] = ( temp[:,0] - ( a[1] + a[0] )/2 )/ ( ( a[1] - a[0] )/2 )
    temp[:,1] = ( temp[:,1] - ( b[1] + b[0] )/2 )/ ( ( b[1] - b[0] )/2 )
    temp[:,2] = ( temp[:,2] - ( c[1] + c[0] )/2 )/ ( ( c[1] - c[0] )/2 )
    
    return temp