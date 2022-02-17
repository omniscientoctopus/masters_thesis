# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # MATLAB Engine

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# import MATLAB engine
import matlab.engine
eng = matlab.engine.start_matlab()
# eng.quit()

# change directory to where Matlab files are stored
# ! Absolute path
# ? How to use relative path for directory above?
eng.cd('/home/i1000609/Prateek/Code/Models/Heat_Conduction/2D_Heat_FEM', nargout=0)


# %%
class Heat_2D_FEM_solver():


    def __init__(self, nEleX, nEleY):

        '''
        Input:
        nEleX = number of elements in the X-direction
        nEleY = number of elements in the Y-direction

        '''
        self.nEleX = nEleX
        self.nEleY = nEleY

        # Calculate numbers of nodes and elements
        self.nNodes = int( (nEleX + 1) * (nEleY +1) )
        self.nEle   = int( nEleX * nEleY )

    def solve(self, lambda_x, lambda_y, Quadrature_points):

        '''
        Input:
        lambda_x = Thermal Conductivity in the X-direction
        lambda_y = Thermal Conductivity in the Y-direction 

        Output:
            solution_coord = [nNodes, 2] 
                            coordinates of nodes  
            solution_u = [nNodes, 1] 
                        Temperature at each node 
            (Node number is consistent in both arrays)

        '''

        self.lambda_x = lambda_x
        self.lambda_y = lambda_y

        # MATLAB datatype conversion
        Quadrature_points = matlab.double(np.ndarray.tolist(Quadrature_points))

        # '__main__' MATLAB file

        # input must to float for compatibility
        # MATLAB initializes all varibles as 'double' a.k.a 'float'
        [solution_coord ,solution_connect, solution_u,
        solution_material, solution_p, temperature_at_quadrature_points] = eng.FlowControl_func(float(self.nEleX), float(self.nEleY), float(self.lambda_x), float(self.lambda_y), Quadrature_points, nargout=6)


        # Output of MATLAB engine is a 'double'
        # Convert to numpy array float
        self.solution_coord = np.copy(solution_coord)
        self.solution_connect = np.copy(solution_connect)
        self.solution_u = np.copy(solution_u)
        self.solution_material = np.copy(solution_material)
        self.solution_p = np.copy(solution_p)
        temperature_at_quadrature_points = np.copy(temperature_at_quadrature_points)

        return self.solution_coord, self.solution_u, temperature_at_quadrature_points


    def plot(self, quantity_of_interest):

        sns.set(style = "darkgrid")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        x = self.solution_coord[:,0]
        y = self.solution_coord[:,1]
        z = quantity_of_interest

        ax.scatter(x, y, z)
        # ax.scatter(x, y, c=z)

        plt.show()


