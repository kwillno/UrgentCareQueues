import numpy as np 
import matplotlib.pyplot as plt
from numpy import lib
from scipy.stats import norm
from scipy.stats.mstats_basic import linregress
from scipy.linalg import inv



'''
PROBLEM 2a
'''


# ARRAYS AND VARIABLES

grid = np.arange(0.25, 0.505, 0.005) # an array with 51 elements from 0.25 to 0.50
xB = np.array([0.30, 0.35, 0.39, 0.41, 0.45]) # measured values of theta given in the problem sheet
yB = np.array([0.5, 0.32, 0.40, 0.35, 0.60]) # measured values of y(theta) given in the problem sheet




# FUNCTIONS


def make_grid(xy1=0.25, xy2=0.505, delta=0.005):
    '''
    This function makes a (xy2-xy1/delta)x(xy2-xy1/delta) dimensional grid, which makes it easier to create the sigma-matrix
    '''
    xy_arr = np.arange(xy1, xy2, delta)
    X,Y = np.meshgrid(xy_arr, xy_arr)
    return (X+Y)/2


def make_mu(t, m=5):
    '''
    here we initialize each component of the mu-vector with the help of m()
    '''
    n = len(t)
    mu = np.zeros(n)
    for i in range(len(mu)):
        mu[i] = m
    return mu



def r(t1, t2):
    '''
    This is the corrolation-function implementes as given in the problem-sheet.
    '''
    return (1+15*abs(t1-t2))*np.exp(-15*abs(t1-t2))



def C(t1, t2, s=0.5**2):
    '''
    This is the covariance-function impemented as given in the problem-sheet
    '''
    return s*r(t1, t2)



def make_standard_sigma(xB):
    '''
    Uses the covariance-function to initialize aech component of the sigma-matrix
    '''
    sigma = np.zeros((len(xB), len(xB)))
    n = len(sigma)
    for i in range(n):
        for j in range(n):
            sigma[i,j] = C(xB[i], xB[j])
    return sigma


def index(tB, t):
    '''
    returns the boolean-indicies of where the theta-values from tB are in t
    '''
    new_tB = np.zeros(len(t), dtype=bool)
    for i in range(len(t)):
        new_tB[i] = any(abs(tB-t[i])<1e-10)
    return new_tB



    

def main() -> None:

    inds = index(xB, grid)
    inds_Compliment = np.invert(inds)

    #construction of the original muA, muB and sigma
    muA = make_mu(grid)
    muB = make_mu(xB) 
    sigma = make_standard_sigma(grid)

    # Construction of the components of mu_C and sigma_C

    mu_A = muA[inds]
    sigma_BB = sigma[np.ix_(inds_Compliment, inds_Compliment)]
    sigma_AB = sigma[np.ix_(inds, inds_Compliment)]

    sigma_AA = sigma[np.ix_(inds, inds)]
    sigma_BA = sigma[np.ix_(inds_Compliment, inds)]

    print(np.shape(sigma_AA))
    print(np.shape(sigma_BB))
    print(np.shape(sigma_AB))
    print(np.shape(sigma_BA))

    # the making of mu_C and sigma_C
    #mu_C:
    a = (xB - muB)
    b = np.matmul(inv(sigma_BB), a)
    print(b)
    c = np.matmul(sigma_AB, b)
    mu_C = mu_A + c
    #sigma_C
    a = np.matmul(sigma_BA, (sigma_BB))
    b = np.matmul(sigma_AB, a)
    sigma_C = sigma_AA - b 

    print(mu_C)

main() 