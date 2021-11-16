import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st


# Problem b, 1
# Simulate X(t)

def simulateX(lam, mu, t=50):
	"""
	Function to create a realization of X(t).
	"""

	# Redefining lambda and mu to account for minute timesteps.
	lam, mu = lam/60, mu/60

	# Setting timesteps to the minutes in each day.
	N = int(t*24*60)
	X = np.zeros(N)
	t = np.zeros(N)

	# Running through all timesteps for simulation
	for i in range(1,N):

		# Getting exponentially distributed sojourn time
		# and blocking action until it ends.
		S = int(np.random.exponential(lam + mu))
		block = t[i-1] + S - 1

		# Action when updating state.
		if t[i] > block:
			u = np.random.uniform()
			if u < lam/(lam + mu):
				X[i] = X[i-1] + 1
			else:
				X[i] = X[i-1] - 1
		else:
			X[i] = X[i-1]

		# Making sure state is nonnegative
		if X[i] < 0:
			X[i] = 0

	return X


def plotRealizations(n, lam, mu, t=50):
	"""
	Testing function.

	Plots multiple(n) realizations of X(t)
	"""
	plt.figure("Test")
	for i in range(n):
		X = simulateX(lam, mu, t)
		plt.plot(X)
	plt.show()

# Problem b, 2
# Finding expected time a patient spends in the UCC

# Using Little's law to find expected wait time
expectedTime = lambda avgCustomers, arrivalRate: avgCustomers / arrivalRate


# Problem b, 3
# Finding 95% CI

def getCI(lam, mu, percentage=0.95):
	"""
	Function for finding confidence interval of wait time using
	a realization of X(t).
	"""
	waitTimes = np.zeros(30)
	for i in range(len(waitTimes)):
		X = simulateX(lam, mu, t = 50)
		waitTimes[i] = expectedTime(np.average(X),lam)

	print("Normal distribution")
	CI = st.norm.interval(percentage, loc=np.mean(waitTimes), scale=st.sem(waitTimes))
	print(f"Wait times {percentage*10:.0f}% CI: {CI[0]*60:.2f} - {CI[1]*60:.2f} mins, {CI[0]:.4f} - {CI[1]:.4f} hours")


exactWaitTime = lambda lam, mu : 1/(mu-lam)
def getExactWaitTime(lam, mu):
	"""
	Gets the exact wait time using the analytical solution.
	"""
	waitHour = exactWaitTime(lam,mu)
	waitMin = waitHour*60
	print(f"Exact wait time: {waitHour:.4f} hours, {waitMin:.2f}")

# Problem b, 4
# Plot one realization of {X(t) : t ≥ 0} for the time 0–12 hours.


def plotRealizationX(lam=5, mu=6):
	"""
	Plots one realization of X(t) as a discrete-plot where
	the functions are not continous. 
	"""
	X = simulateX(lam, mu, t = 1/2)
	curves = []
	i = 1
	while i < len(X):
		curves.append(([i-1,i], [X[i-1], X[i-1]]))
		i = i+1 

	plt.figure("Try")
	plt.title("Realization of $X(t)$")
	for x,y in curves:
		x,y = np.array(x), np.array(y)
		plt.plot(x/60,y,"b")
	plt.xlabel("Hours")
	plt.xlim([-0.4,12.4])
	plt.ylabel("Patients")
	plt.show()


# ---------------------------------
# Problem 1f

W_U = lambda p, lam, mu : 1/(mu-p*lam) 
W_N = lambda p, lam, mu : mu/((mu - lam)*(mu - p*lam))

def plotWaits(lam, mu):
	"""
	Plots the analytical functions W_U(p) and W_N(p) 
	with given lambda and mu.
	"""
	p = np.linspace(0,1)

	W_u = W_U(p, lam, mu)
	W_n = W_N(p, lam, mu)

	plt.figure("waitTimes")
	plt.title("Expected time in UCC")
	plt.plot(p,W_u,label="$W_U$ - Urgent patients")
	plt.plot(p,W_n,label="$W_N$ - Normal patients")
	plt.xlabel("Probability that patient is urgent")
	plt.ylabel("Hours")
	plt.legend()
	plt.show()

def getWaitTimeExtremes(lam, mu):
	print(f"Expected wait time for a normal patient is {W_N(0,lam,mu)}, p~0. ")
	print(f"Expected wait time for a normal patient is {W_N(1,lam,mu)}, p~1. ")



# ---------------------------
# problem 1g

def muFnc(u, n, mu):
	"""
	Helperfunction for simulating realizations of U(t) and N(t).
	"""
	if u+n == 0:
		return 0
	return mu


def simulateUN(p, lam, mu, t=50):
	"""
	Function to create a realization of U(t) and N(t).
	"""

	# Redefining lambda and mu to account for minute timesteps.
	lam, mu = lam/60, mu/60
	
	# Setting timesteps to the minutes in each day.
	# Creating needed arrays and fill with zeros.
	timeslices = int(t*24*60)
	U = np.zeros(timeslices)
	N = np.zeros(timeslices)
	block = 0

	# Running through all timesteps for simulation
	for i in range(1, timeslices):
		U[i] = U[i-1]
		N[i] = N[i-1]

		# Blocking action until iteration given by block.
		if not (i < block):

			# Getting a valid mu for i'th iteration
			mu_i = muFnc(U[i-1], N[i-1], mu)

			# Getting exponentially distributed sojourn time
			S = (np.random.exponential((mu_i + lam)**(-1)))
			block = i + S

			# Finding increment/decrement irrespective of U/N
			change = np.random.choice([-1,1], p=( mu_i/(mu_i + lam) , lam/(mu_i + lam) ))

			if ( # All the possible situations for change in N(t)
				(U[i-1] == 0 and change == -1) or 
				(np.random.uniform() > p and change == 1)
			):
				N[i] += change
			else:
				U[i] += change

	return U,N

def plotRealizationsUN(n, p, lam, mu, t=50):
	"""
	Testing function.

	Plots multiple(n) realizations of U(t) and N(t)
	"""
	plt.figure("Test")
	for i in range(n):
		U, N = simulateUN(p, lam, mu, t)
		plt.plot(U, "b", label="Urgent")
		plt.plot(N, "r", label="Normal")
	plt.grid()
	plt.legend()
	plt.show()


def plotRealizationUN(p=0.8, lam=5, mu=6, t=1):
	"""
	Plots one realization of X(t) as a discrete-plot where
	the functions U(t) and N(t)are not continous. 
	"""
	# Completing simulation
	U, N = simulateUN(p, lam, mu, t)

	# Transform to discrete functions
	curvesU = []
	curvesN = []
	i = 1
	while i < len(U):
		curvesU.append(([i-1,i], [U[i-1], U[i-1]]))
		curvesN.append(([i-1,i], [N[i-1], N[i-1]]))
		i = i+1 

	# Big scary plotting mess
	plt.figure("Try")
	plt.title("Realization of $U(t)$ and $N(t)$")

	# Plots fot U(t)
	plt.plot(0,0,"b",label="$U(t)$")
	for x,y in curvesU:
		x,y = np.array(x), np.array(y)
		plt.plot(x/60,y,"b")

	# Plots fot N(t)
	plt.plot(0,0,"r",label="$N(t)$")
	for x,y in curvesN:
		x,y = np.array(x), np.array(y)
		plt.plot(x/60,y,"r")

	plt.xlabel("Hours")
	plt.ylabel("Patients")
	plt.legend()
	plt.show()


def getCI_UN(p, lam, mu, percentage=0.95):
	"""
	Function for finding confidence interval of wait time using
	a realization of U(t) and N(t).
	Gives separate Confidence intervals for U(t) and N(t).
	"""
	waitTimesU = np.zeros(30)
	waitTimesN = np.zeros(30)

	for i in range(len(waitTimesU)):
		U, N = simulateUN(p, lam, mu, t = 50)
		waitTimesU[i] = expectedTime(np.average(U),p*lam)
		waitTimesN[i] = expectedTime(np.average(N),(1-p)*lam)

	print("Normal distribution")
	CI_U = st.norm.interval(percentage, loc=np.mean(waitTimesU), scale=st.sem(waitTimesU))
	print(f"Wait times {percentage*10:.0f}% CI Urgent: {CI_U[0]*60:.2f} - {CI_U[1]*60:.2f} mins, {CI_U[0]:.4f} - {CI_U[1]:.4f} hours")

	CI_N = st.norm.interval(percentage, loc=np.mean(waitTimesN), scale=st.sem(waitTimesN))
	print(f"Wait times {percentage*10:.0f}% CI Normal: {CI_N[0]*60:.2f} - {CI_N[1]*60:.2f} mins, {CI_N[0]:.4f} - {CI_N[1]:.4f} hours")


def getExact_UN(p=0.8, lam=5, mu=6):
	"""
	Gets the exact wait time using the analytical solution.
	These are here given by W_U and W_N functions defined above.
	"""
	Wu = W_U(p, lam, mu)
	Wn = W_N(p, lam, mu)

	WuMin, WnMin = Wu*60, Wn*60
	print(f"Exact wait time: {WuMin:.2f} minutes, {Wu:.4f} hours")
	print(f"Exact wait time: {WnMin:.2f} minutes, {Wn:.4f} hours")

# Main section
# --------------------------------------------------
# Function calls go below this line

lam, mu = 5, 6

plotRealizations(2, lam, mu)

getCI(lam,mu)

getExactWaitTime(lam,mu)

plotRealizationX(lam, mu)

plotWaits(lam,mu)
getWaitTimeExtremes(lam, mu)

plotRealizationsUN(1, 0.8, lam, mu, t=1/8)

plotRealizationUN(t=1/2)

getCI_UN(0.8, lam, mu)

getExact_UN()