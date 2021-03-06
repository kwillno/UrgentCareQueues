import numpy as np
import matplotlib.pyplot as plt
from numpy import lib, nan
from scipy.stats import norm
from scipy.linalg import inv


"""
PROBLEM 2a
"""

# ARRAYS AND VARIABLES

grid = np.arange(0.25, 0.505, 0.005)  # an array with 51 elements from 0.25 to 0.50
xB = np.array(
    [0.30, 0.35, 0.39, 0.41, 0.45]
)  # measured values of theta given in the problem sheet
yB = np.array(
    [0.5, 0.32, 0.40, 0.35, 0.60]
)  # measured values of y(theta) given in the problem sheet

# FUNCTIONS


def make_mu(t, m=0.5):
    """
    here we initialize each component of the mu-vector (who has the length of t) with the value of m=0.5
    """
    n = len(t)
    mu = np.zeros(n)
    for i in range(len(mu)):
        mu[i] = m
    return mu


def r(t1, t2):
    """
    This is the corrolation-function implementes as given in the problem-sheet.
    """
    return (1 + 15 * abs(t1 - t2)) * np.exp(-15 * abs(t1 - t2))


def C(t1, t2, s=0.5 ** 2):
    """
    This is the covariance-function impemented as given in the problem-sheet
    """
    return s * r(t1, t2)


def make_standard_sigma(xB):
    """
    Uses the covariance-function to initialize aech component of the sigma-matrix
    """
    sigma = np.zeros(
        (len(xB), len(xB))
    )  # a matrix where we are going to place the values of sigma
    n = len(sigma)
    for i in range(n):
        for j in range(n):
            sigma[i, j] = C(
                xB[i], xB[j]
            )  # each element of sigma are to be implemented with covariance-function
    return sigma


def index(tB, t):
    """
    returns the boolean-indicies of where the theta-values from tB are in t
    """
    new_tB = np.zeros(len(t), dtype=bool)  # new array to hold boolean values
    for i in range(len(t)):
        new_tB[i] = any(
            abs(tB - t[i]) < 1e-10
        )  # uses any() to check if the i_th element of t is in tB
    return new_tB


def make_mu_sigma_C():
    """
    This function constructs all the relevant submatricies of sigma, so we can use them
    to construct the conditional mu and sigma matricies.
    """
    inds = index(
        xB, grid
    )  # inds is a boolean array where every element of grid that is in xB is True
    inds_Complement = np.invert(
        inds
    )  # inverts inds so Truth becomes False and vice verca

    # construction of the original muA, muB and sigma
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
    # mu_C:
    mu_C = mu_A + sigma_AB @ np.linalg.solve(
        sigma_BB, (yB - muB)
    )  # we use the formula for conditional mu
    # sigma_C
    a = np.matmul(
        inv(sigma_BB), sigma_BA
    )  # in these lines we du much the same as above, but in more steps
    b = np.matmul(sigma_AB, a)
    sigma_C = sigma_AA - b

    var_C = np.diagonal(sigma_C)

    return mu_C, sigma_C, var_C, inds


def predInt(i, var_C):
    """
    Returns 90% prediciton interval for theta_i
    """
    return 1.645 * np.sqrt(var_C[i])


def make_conf_intervalls(mu, var):
    """
    Returns upper and lower bound with respect to the prediciton interval
    """
    upper = np.zeros_like(mu)
    lower = np.zeros_like(mu)
    for i, mu_i in np.ndenumerate(mu):
        upper[i], lower[i] = mu_i + predInt(i, var), mu_i - predInt(i, var)
    return upper, lower


def plot_mu_C():
    """
    In this function we use the functions above to initialize conditional mu and conditional var, and use these
    arrays toughether with grid (that holds the theta-values) to plot conditional theta and a 90% prediction
    interval.
    """
    mu_C, sigma_C, var_C, inds = make_mu_sigma_C()
    ind_values = np.array([i for i in range(len(inds)) if inds[i]])
    ind_values -= np.arange(len(ind_values))
    mu_C_new = np.insert(mu_C, ind_values, yB)
    var_C = np.insert(var_C, ind_values, np.zeros(len(ind_values)))

    # here we construct the upper and lower limits of the 90% confidence interval
    upper, lower = make_conf_intervalls(mu_C_new, var_C)

    # here we plot the confidence-interval toughether with conditional mu
    plt.figure(figsize=(10, 6))
    plt.plot(xB, yB, "o")
    plt.plot(grid, mu_C_new, color="b", label="Conditional mean")
    plt.fill_between(grid, lower, upper, color="b", alpha=0.1)
    plt.title("Prediction, with 90% prediction interval")
    plt.xlabel("$\\theta$")
    plt.ylabel("$Y(\\theta)$")
    plt.grid()
    plt.show()


""" 
PROBLEM 2B
"""


def problem_2_b():
    """
    here we use norm.cdf to find the conditional probability that y(theta)<0.30.
    We call the array with those values cd_val, and plot them as a function of theta.
    """

    # down below we initialize the necessarry variables needed to construct cd_vals
    mu_C, sigma_C, var_C, inds = make_mu_sigma_C()
    ind_values = np.array([i for i in range(len(inds)) if inds[i]])
    ind_values -= np.arange(len(ind_values))
    mu_C_new = np.insert(mu_C, ind_values, yB)
    var_C = np.insert(var_C, ind_values, 1e-5 * np.ones(len(ind_values)))

    cd_vals = norm.cdf(
        0.3 * np.ones_like(mu_C_new), mu_C_new, np.sqrt(var_C)
    )  # use norm.cdf to find cd

    # down here we plot
    plt.plot(grid, cd_vals)
    plt.title("Cond. Prob. Y(??) < 0.30")
    plt.xlabel("$\\theta$")
    plt.ylabel("Conditional probability")
    plt.grid()
    plt.show()


"""
PROBLEM 2C

Here we just use the functions that we have constructed, explained and used earlier to 
plot the same things as in a) and b), but with a point added to grid
"""

# these are the same values as before, but with the point (0.33, 0.40) added
xB = np.array([0.30, 0.33, 0.35, 0.39, 0.41, 0.45])
yB = np.array([0.5, 0.40, 0.32, 0.40, 0.35, 0.60])


def problem_2_c1():

    # as before we construct the necissarry variables
    mu_C, sigma_C, var_C, inds = make_mu_sigma_C()
    ind_values = np.array([i for i in range(len(inds)) if inds[i]])
    ind_values -= np.arange(len(ind_values))
    mu_C_new = np.insert(mu_C, ind_values, yB)
    var_C = np.insert(var_C, ind_values, np.zeros(len(ind_values)))

    # here we construct the confidence interval
    upper, lower = make_conf_intervalls(mu_C_new, var_C)

    # here we plot it all
    plt.figure(figsize=(10, 6))
    plt.plot(xB, yB, "o")
    plt.plot(grid, mu_C_new, color="b", label="Conditional mean")
    plt.fill_between(grid, lower, upper, color="b", alpha=0.1)
    plt.grid()
    plt.title("Prediction, with 90% PI, using $6$ points")
    plt.xlabel("$\\theta$")
    plt.ylabel("$Y(\\theta)$")
    plt.show()


def problem_2_c2():
    # We construct the mu_C values
    mu_C, sigma_C, var_C, inds = make_mu_sigma_C()
    ind_values = np.array([i for i in range(len(inds)) if inds[i]])
    ind_values -= np.arange(len(ind_values))
    mu_C_new = np.insert(mu_C, ind_values, yB)
    var_C = np.insert(var_C, ind_values, 1e-5 * np.ones(len(ind_values)))

    # here we construct the confidence interval
    upper, lower = make_conf_intervalls(mu_C_new, var_C)

    cd_vals = norm.cdf(0.3 * np.ones_like(mu_C_new), mu_C_new, np.sqrt(var_C))

    maks = np.amax(cd_vals)
    maks_index = np.argmax(cd_vals)

    print(f"max value: {maks:.4f}, index of max value: {grid [maks_index]:.2f}")

    plt.plot(grid, cd_vals)
    plt.title("P{Y(??) < 0.30} with $6$ points")
    plt.xlabel("$\\theta$")
    plt.ylabel("P{Y(??) < 0.30}")
    plt.grid()
    plt.show()


def main():

    plot_mu_C()

    problem_2_b()

    problem_2_c1()
    problem_2_c2()


main()
