import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st


# Problem b, 1
# Simulate X(t)

def simulateX(lam, mu, t=50):
	"""
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
	
	lam, mu = lam/60, mu/60

	N = t*24*60
	X = np.zeros(N)
	t = np.zeros(N)

	for i in range(1,N):
		a = X[i-1]
		s = int(np.random.exponential(lam + mu))
		block = t[i-1] + s - 1
		u = np.random.uniform()
		if t[i] > block:
			if u < lam/(lam + mu):
				X[i] = a + 1
			else:
				X[i] = a - 1
		else:
			X[i] = a

		if X[i] < 0:
			X[i] = 0


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
	print(f"Wait times 95% CI: {CI[0]*60:.2f} - {CI[1]*60:.2f} mins, {CI[0]:.4f} - {CI[1]:.4f} hours")


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


def simulateUN(p, lam, mu, t=50):

	lam, mu = lam/60, mu/60

	lamU, lamN = p*lam, (1-p)*lam
	muU, muN = mu, mu-lamU

	timeslices = int(t*24*60)
	U = np.zeros(timeslices)
	N = np.zeros(timeslices)
	t = np.linspace(0,timeslices, timeslices+1)

	for i in range(1,timeslices):

		sU = int(np.random.exponential(lamU + muU))
		blockU = t[i-1] + sU
		uU = np.random.uniform()

		if U[i-1] > 0:
			# print(f"blockU should be True : {blockU > t[i]} => {blockU - t[i]}")
			blockN = blockU
		else: 
			sN = int(np.random.exponential(lamN + muN))
			uN = np.random.uniform()
			blockN = t[i-1] + sN

		# Handling Normal patients
		if uN < lamN/(lamN + muN):
			N[i] = N[i-1] + 1
		elif uN > lamN/(lamN + muN) and t[i] >= blockN:
			N[i] = N[i-1] - 1
		else:
			N[i] = N[i-1]


		# Handling urgent patients
		if uU < lamU/(lamU + muU):
			U[i] = U[i-1] + 1
		elif uU > lamU/(lamU + muU) and t[i] >= blockU:
			U[i] = U[i-1] - 1
		else:
			U[i] = U[i-1]

		# Not allowing negative number of patients
		if U[i] < 0:
			U[i] = 0

		if N[i] < 0:
			N[i] = 0

	return U,N

def plotRealizationsUN(n, p, lam, mu, t=50):
	plt.figure("Test")
	for i in range(n):
		U, N = simulateUN(p, lam, mu, t)
		plt.plot(U, "b", label="Urgent")
		plt.plot(N, "r", label="Normal")
	plt.legend()
	plt.show()


def plotRealizationUN(p=0.8, lam=5, mu=6, t=1):
	U, N = simulateUN(p, lam, mu, t)
	curvesU = []
	curvesN = []
	i = 1
	while i < len(U):
		curvesU.append(([i-1,i], [U[i-1], U[i-1]]))
		curvesN.append(([i-1,i], [N[i-1], N[i-1]]))
		i = i+1 

	plt.figure("Try")
	plt.title("Realization of $X(t)$")
	plt.plot(0,0,"b",label="Urgent")
	for x,y in curvesU:
		x,y = np.array(x), np.array(y)
		plt.plot(x/60,y,"b")

	plt.plot(0,0,"r",label="Normal")
	for x,y in curvesN:
		x,y = np.array(x), np.array(y)
		plt.plot(x/60,y,"r")
	plt.xlabel("Hours")
	plt.ylabel("Patients")
	plt.legend()
	plt.show()


def getCI_UN(p, lam, mu, percentage=0.95):

	waitTimesU = np.zeros(30)
	waitTimesN = np.zeros(30)

	for i in range(len(waitTimesU)):
		U, N = simulateUN(p, lam, mu, t = 50)
		waitTimesU[i] = expectedTime(np.average(U),lam)
		waitTimesN[i] = expectedTime(np.average(N),lam)

	print("Normal distribution")
	CI_U = st.norm.interval(0.95, loc=np.mean(waitTimesU), scale=st.sem(waitTimesU))
	print(f"Wait times 95% CI Urgent: {CI_U[0]*60:.2f} - {CI_U[1]*60:.2f} mins, {CI_U[0]:.4f} - {CI_U[1]:.4f} hours")

	CI_N = st.norm.interval(0.95, loc=np.mean(waitTimesN), scale=st.sem(waitTimesN))
	print(f"Wait times 95% CI Normal: {CI_N[0]*60:.2f} - {CI_N[1]*60:.2f} mins, {CI_N[0]:.4f} - {CI_N[1]:.4f} hours")

# Main section
# --------------------------------------------------
# Function calls go below this line

lam, mu = 5, 6

# plotRealizations(2, lam, mu)

# getCI(lam,mu)

# getExactWaitTime(lam,mu)

# plotRealizationX(lam, mu)

# plotRealizationsUN(1, 0.8, lam, mu, t=1/8)

# plotRealizationUN(t=1)

# getCI_UN(0.8, lam, mu)