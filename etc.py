import numpy as np

def main() -> None:
    inds = np.array([False, True, True, False, True])
    inds_C = np.invert(inds)

    mu = np.random.uniform(0, 1, 5)
    Sigma = np.random.uniform(0, 1, (5, 5))
    # print(mu, Sigma)

    mu_A = mu[inds]
    Sigma_AB = Sigma[np.ix_(inds, inds_C)]

    print(mu_A)
    print(Sigma_AB)

main()

################################################################################################## TRASH

# trenger ikke denne
""" def m(theta):
    '''
    mean-function used to make mu=(m(t1), m(t2), ... , m(tn))
    '''
    return 5 """



# trenger kanskje ikke denne
""" def make_sigma(xA, xB):
    '''
    Uses the covariance-function to initialize aech component of the sigma-matrix according to the values of xA and xB
    '''
    lA = len(xA)
    lB = len(xB)
    sigma = np.zeros((len(lA), len(lB)))
    for i in range(lA):
        for j in range(lB):
            sigma[i,j] = C(i, j)
    return sigma """


# trenger ikke denne    
""" def invert(tB):
    '''
    inverts the boolean values of the index-function
    '''
    for i, elem in enumerate(tB):
        if elem==0:
            tB[i] = 1
        else:
            tB[i] = 0
    return tB """


# trenger kanskje ikke denne
""" def count(tB):
    '''
    counts the number of True-values in an boolean array
    '''
    n = 0
    for i in tB:
        if i == 1:
            n += 1
    return n """



# trenger kanskje ikke denne funksjonen
""" def muA(tB,t):
    '''
    creates average value of the non-observed values of y(theta) 
    '''
    vals_index = invert(index(tB, t))
    numerator = np.dot(t, vals_index)
    denumerator = count(vals_index)
    return numerator/denumerator """


# trenger ikke denne
""" def muB(tB,t):
    '''
    creates mu-values from the observed values of y(theta)
    '''
    vals_index = index(tB, t)
    numerator = np.dot(t, vals_index)
    denumerator = count(vals_index)
    return numerator/denumerator """