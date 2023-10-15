import numpy as np
import matplotlib.pyplot as plt

x = np.genfromtxt("Sample Data//ex2x.dat")
y = np.genfromtxt("Sample Data//ex2y.dat")

theta = [10, 10]

m = len(x) #no. of data points
ls1 = []
for i in range(m):
	ls1.append([1, x[i]])

X = np.array(ls1)

#Theta*X is actual prediction, minimising error...
n = 100 #no. of iterations
alpha = 1

x_loc = np.linspace(3,7,15)
y_loc = []
gam = 0.5
lamb = 2
for l in range(len(x_loc)):
	for i in range(n):
		alpha = 0.5
		for j in range(len(theta)):
			h = X.dot(theta)
			for k in range(m):
				sum = alpha*np.exp(-(x[k]-x_loc[l])**2/(gam**2))*(h[k]-y[k])*x[k]
			theta[j] = theta[j] - sum - alpha*lamb*theta[j]
	y_loc.append(theta[0] + x_loc[l]*theta[1])

y_loc = np.array(y_loc)

plt.xlabel("Height (x)--->")
plt.ylabel("Age (y)--->")
plt.plot(x_loc, y_loc)
plt.scatter(x,y)
plt.show()
