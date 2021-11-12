import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import append 
from scipy import stats as st

def simulateX(lam, mu, t=50):

	timeslices = int(t*24)
	X = np.zeros(timeslices)

	# Array for arrivals and departures
	arrivals = np.random.poisson(lam, (timeslices))
	departures = np.random.poisson(mu, (timeslices))

	X[0] = 0
	for i in range(1,timeslices):
		X[i] = X[i-1] + arrivals[i] - departures[i]
		if X[i] < 0:
			X[i] = 0 


	return X

"""
plt.figure("Test")
for i in range(1):
	X = simulateX(lam = 5, mu = 6, t = 50)
	plt.plot(X)
"""

# Problem b, 2
# Finding expected time a patient spends in the UCC

# Using Little's law to find expected wait time
expectedTime = lambda avgCustomers, arrivalRate: avgCustomers / arrivalRate

"""
lamb = 5
my = 6
plt.figure("Test")
for i in range(10):
	X = simulateX(lamb, my, t = 50)
	print(f"Expected wait time is: {expectedTime(np.average(X),lamb)*60:.0f} min")
	plt.plot(X,label=i)
plt.legend()
plt.show()
"""


# Problem b, 3
# Finding 95% CI

"""
lam = 5
mu = 6
waitTimes = np.zeros(30)
for i in range(len(waitTimes)):
	X = simulateX(lam, mu)
	waitTimes[i] = expectedTime(np.average(X),lam)
print("Normal distribution")
CI = st.norm.interval(0.95, loc=np.mean(waitTimes), scale=st.sem(waitTimes))
print(f"Wait times 95% CI: {CI[0]*60:.0f} - {CI[1]*60:.0f} mins, {CI} hours")
"""

# Problem b, 4
# Plot one realization of {X(t) : t ≥ 0} for the time 0–12 hours.


def plotRealizationX(lam=5, mu=6):
	X = simulateX(lam, mu, t = 1/2)
	curves = []
	i = 1
	while i < len(X):
		curves.append(([i-1,i], [X[i-1], X[i-1]]))
		i = i+1 

	plt.figure("Try")
	plt.title("Realization of $X(t)$")
	for x,y in curves:
		plt.plot(x,y,"b")
	plt.xlabel("Hours")
	plt.xlim([-0.4,12.4])
	plt.ylabel("Patients")
	plt.show()

##########################################################################################

def U(p,lam,t):
    return np.random.poisson(p*lam, t)

def N(q, lam, t):
    return np.random.poisson(q*lam, t)

def simulate_U_N(p=0.8, lam=5, mu=6, t=50):
    q = 1-p

    # we need two containers to store number of urgent and normal patients
    U_container = 0
    N_container = 0

    # we also need two arrays to store the N and U values at specific times
    U_arr = np.array([0])
    N_arr = np.array([0])

    # we make time-array for time between realizations
    t_vals = np.array([])
    while np.sum(t_vals)<50:
        val = np.random.exponential((lam), 1)[0]
        t_vals = np.append(t_vals,val)

    print(t_vals)

    for i in range(len(t_vals)):

        # here we register new arrivals
        U_container += U(p,lam,t_vals[i])
        N_container += N(q,lam,t_vals[i])

        # Here we register people leaving
        departures = np.random.poisson(mu, t_vals[i])

        # in these loops we distribute from which container the healthy patiens depart
        if U_container == 0:
            for j in range(departures):
                if N_container != 0:
                    N_container -= 1
        if U_container != 0:
            for j in range(departures): 
                if U_container != 0:
                    U_container -= 1
                else: 
                    N_container -=1
        
        # now we place the number of current urgent and normal pacient at the correct place in the arrays.
        np.append(U_arr, U_container)
        np.append(N_arr, N_container)

    return U_arr, N_arr, t_vals

def plot_U_N(U_arr, N_arr, t_vals): 

    # we need a for loop to plot the U values and N values at correct time_intervals
    i=0
    while i<len(t_vals): 
        plt.plot(t_vals[i], U_arr[i])
        plt.plot(t_vals[i], N_arr[i])
    plt.show

U_arr, N_arr, t_vals = simulate_U_N(0.8, 5, 6, 50)

plot_U_N(U_arr, N_arr, t_vals)

plt.plot(np.linspace(0,1,100), np.linspace(0,1,100))
plt.show()


