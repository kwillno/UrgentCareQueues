import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st


# Problem b, 1
# Simulate X(t)

def simulateX(lam, mu, t=50):

	# Redefining lambda and mu to account for minute timesteps.
	lam, mu = lam/60, mu/60

	# Setting timesteps to the minutes in each day.
	timeslices = int(t*24*60)
	X = np.zeros(timeslices)

	# Array for arrivals and departures
	arrivals = np.random.poisson(lam, (timeslices))
	departures = np.random.poisson(mu, (timeslices))

	# Running through all timesteps for simulation
	for i in range(1,timeslices):
		# Setting next X to be previous X + arrivals - departures
		X[i] = X[i-1] + arrivals[i-1] - departures[i-1]
		if X[i] < 0:
			X[i] = 0 
	
	
	"""
	# Attempt 2, not correct either, here takes multiple patients at a time
	# Array for arrivals	
	arrivals = np.random.poisson(lam, (timeslices))

	Y = np.zeros(timeslices)
	for i in range(1, timeslices):
		X[i] = X[i-1] + arrivals[i]
		counter = 0
		for j in range(int(arrivals[i])):
			counter += 1
			try:
				expTime = int(np.round(np.random.exponential(mu)))
				i = i+expTime
				X[i] = X[i] - 1 
			except IndexError as e:
				# print(e)
				break
	
	"""
	"""
	# Attempt 3
	# Array for arrivals and departures
	arrivals = np.random.poisson(lam, (timeslices))
	departures = np.random.poisson(mu, (timeslices))

	for i in range(1,timeslices):
		X[i] = X[i-1] + arrivals[i-1] - departures[i-1]
		if X[i] < 0:
			print(X[i])
			X[i] = 0
	"""

	return X


def plotRealizations(n, lam, mu, t=50):
	plt.figure("Test")
	for i in range(n):
		X = simulateX(lam, mu, t)
		plt.plot(X)
	plt.show()

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

def getCI(lam, mu, percentage=0.95):

	waitTimes = np.zeros(30)
	for i in range(len(waitTimes)):
		X = simulateX(lam, mu, t = 50)
		waitTimes[i] = expectedTime(np.average(X),lam)

	print("Normal distribution")
	CI = st.norm.interval(0.95, loc=np.mean(waitTimes), scale=st.sem(waitTimes))
	print(f"Wait times 95% CI: {CI[0]*60:.2f} - {CI[1]*60:.2f} mins, {CI} hours")


exactWaitTime = lambda lam, mu : 1/(mu-lam)
def getExactWaitTime(lam, mu):
	waitHour = exactWaitTime(lam,mu)
	waitMin = waitHour*60
	print(f"Exact wait time: {waitHour:.4f} hours, {waitMin:.2f}")

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
		x,y = np.array(x), np.array(y)
		plt.plot(x/60,y,"b")
	plt.xlabel("Hours")
	plt.xlim([-0.4,12.4])
	plt.ylabel("Patients")
	plt.show()




# Main section
# --------------------------------------------------
# Function calls go below this line

lam, mu = 5, 6

# plotRealizations(2, lam, mu)

getCI(lam,mu)

getExactWaitTime(lam,mu)

plotRealizationX(lam, mu)