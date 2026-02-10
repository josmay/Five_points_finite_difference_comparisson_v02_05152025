import numpy as np


def fp_operator(dx:float, n:int, size:int):
    """
    This method return the five points operator for the 1D n-th derivative
    
    Args:
        dx: float, numerical resolution 
    n:  int, order of the derivative. Should be from 1 to 4
        size:  int, the dimension of the array
        
    Return:
        A size x size matrix representing the five point operator for the derivative of n order 
    """
    
    #b matrix
    b=np.array([[0,0,0,0],
                [1/dx,0,0,0],
                [0,2/dx**2,0,0],
                [0,0,6/dx**3,0],
                [0,0,0,24/dx**4]],dtype=float)
    
    
    #Stencil 1 -> x0
    S1=np.array([[1,1,1,1,1],
                 [0,1,2,3,4],
                 [0,1,4,9,16],
                 [0,1,8,27,64],
                 [0,1,16,81,256]], dtype=float)
    
    #Stencil 2 -> x1
    S2=np.array([[1,1,1,1,1],
                 [-1,0,1,2,3],
                 [1,0,1,4,9],
                 [-1,0,1,8,27],
                 [1,0,1,16,81]], dtype=float)
    
    #Stencil 3 -> xi
    S3=np.array([[1,1,1,1,1],
                 [-2,-1,0,1,2],
                 [4,1,0,1,4],
                 [-8,-1,0,1,8],
                 [16,1,0,1,16]], dtype=float)
    
    #Stencil 4 -> x_n-1
    S4=np.array([[1,1,1,1,1],
                 [-3,-2,-1,0,1],
                 [9,4,1,0,1],
                 [-27,-8,-1,0,1],
                 [81,16,1,0,1]], dtype=float)
    
    #Stencil 5 -> x_n
    S5=np.array([[1,1,1,1,1],
                 [-4,-3,-2,-1,0],
                 [16,9,4,1,0],
                 [-64,-27,-8,-1,0],
                 [256,81,16,1,0]], dtype=float)
    
    #alphas    
    a1=np.linalg.solve(S1,b[:,n-1])
    a2=np.linalg.solve(S2,b[:,n-1])
    a3=np.linalg.solve(S3,b[:,n-1])
    a4=np.linalg.solve(S4,b[:,n-1])
    a5=np.linalg.solve(S5,b[:,n-1])
    
    operator=\
        a3[0]*np.eye(size, k=-2, dtype=np.float64) +\
        a3[1]*np.eye(size, k=-1, dtype=np.float64) +\
        a3[2]*np.eye(size, k=0, dtype=np.float64) +\
        a3[3]*np.eye(size, k=1, dtype=np.float64) +\
        a3[4]*np.eye(size, k=2, dtype=np.float64)
    
    operator[:2]=0  #cleaning the values from a3
    operator[-2:]=0  #cleaning the values from a3
    operator[0, :5] = a1
    operator[1, :5] = a2
    operator[-2, -5:] = a4
    operator[-1, -5:] = a5
    
    #returning the operator
    return operator

