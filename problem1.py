import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st


# Problem b, 1
# Simulate X(t)

def simulateX(lam, mu, t=50):

	timeslices = int(t*24)
	X = np.zeros(timeslices)

	"""
	# Array for arrivals and departures
	arrivals = np.random.poisson(lam, (timeslices))
	departures = np.random.poisson(mu, (timeslices))

	X[0] = 0
	for i in range(1,timeslices):
		X[i] = X[i-1] + arrivals[i] - departures[i]
		if X[i] < 0:
			X[i] = 0 
	"""

	# Array for arrivals	
	arrivals = np.random.poisson(lam, (timeslices))

	Y = np.zeros(len(X))
	for i in range(1, timeslices):
		X[i] = X[i-1] + arrivals[i]
		for j in range(int(X[i])):
			try:
				expTime = np.round(np.random.exponential(mu))
				X[int(i+expTime)] = X[int(i+expTime)] - 1 
			except IndexError as e:
				# print(e)
				pass

	return X


def plotRealizations(n, lam, mu, t=50):
	plt.figure("Test")
	for i in range(n):
		X = simulateX(lam, mu, t)
		plt.plot(X)
	plt.show()

# plotRealizations(1, 5, 6, t = 5)

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

getCI(5,6)

waitHour = exactWaitTime(5,6)
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
		plt.plot(x,y,"b")
	plt.xlabel("Hours")
	plt.xlim([-0.4,12.4])
	plt.ylabel("Patients")
	plt.show()

# plotRealizationX()