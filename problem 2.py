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


def make_mu(t, m=0.5):
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


def make_mu_sigma_C():
    inds = index(xB, grid)
    inds_Complement = np.invert(inds)

    #construction of the original muA, muB and sigma
    muA = make_mu(grid)
    muB = make_mu(xB) 
    sigma = make_standard_sigma(grid)

    # Construction of the components of mu_C and sigma_C

    mu_A = muA[inds_Complement]
    sigma_BB = sigma[np.ix_(inds, inds)]
    sigma_AB = sigma[np.ix_(inds_Complement, inds)]

    sigma_AA = sigma[np.ix_(inds_Complement, inds_Complement)]
    sigma_BA = sigma[np.ix_(inds, inds_Complement)]


    # the making of mu_C and sigma_C
    #mu_C:
    mu_C = mu_A + sigma_AB @ np.linalg.solve(sigma_BB, (yB - muB))
    #sigma_C
    a = np.matmul(inv(sigma_BB), sigma_BA)
    b = np.matmul(sigma_AB, a)
    sigma_C = sigma_AA - b

    var_C = np.diagonal(sigma_C)

    return mu_C, sigma_C, var_C, inds


def predInt(i, var_C):
    """Returns 90% prediciton interval for theta_i"""
    return 1.645*np.sqrt(var_C[i])


def make_conf_intervalls(mu, var):
    """Returns upper and lower bound with respect to the prediciton interval"""
    upper = np.zeros_like(mu)
    lower = np.zeros_like(mu)
    for i, mu_i in np.ndenumerate(mu):
        upper[i], lower[i] = mu_i + predInt(i, var), mu_i - predInt(i, var)
    return upper, lower
 

def plot_mu_C():
    # We construct the mu_C values
    mu_C, sigma_C, var_C, inds = make_mu_sigma_C()
    ind_values = np.array([i for i in range(len(inds)) if inds[i]])
    ind_values -= np.arange(len(ind_values))
    mu_C_new = np.insert(mu_C, ind_values, yB)
    var_C = np.insert(var_C, ind_values, np.zeros(len(ind_values)))

    # here we construct the confidence interval 
    print(np.shape(mu_C_new))
    print(np.shape(var_C))
    upper, lower = make_conf_intervalls(mu_C_new, var_C)

    # here we plot it all
    plt.figure(figsize=(10,6))
    plt.plot(xB, yB,'o')
    plt.plot(grid, mu_C_new, 'b')
    plt.fill_between(grid, lower, upper, color='b', alpha=0.1)
    plt.grid()
    plt.show()


def main() -> None:
    a =1
main() 

plot_mu_C()