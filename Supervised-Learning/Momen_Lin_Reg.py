import numpy as np
import matplotlib.pyplot as plt

x = np.genfromtxt("Sample Data//ex2x.dat")
y = np.genfromtxt("Sample Data//ex2y.dat")

theta = [1/2, 1/2]
#X = np.array([1,x])

m = len(x) #no. of data points
ls1 = []
for i in range(m):
	ls1.append([1, x[i]])

X = np.array(ls1)

#Theta*X is actual prediction, minimising error...
n = 100 #no. of iterations
alpha = 0.025 #learning rate
beta = 0.5 #momentum coefficient

V = []
for i in range(n):
	for j in range(len(theta)):
		h = X.dot(theta)
		for k in range(m):
			v = (h[k]-y[k])*x[k]
			V.append(v)
			sume = beta*(h[k]-y[k])*x[k]-sum(V)*(1-beta)/len(V)
		theta[j] = theta[j] - alpha*sume

def lin(x): #Defining regressed affine function
	return theta[0] + x*theta[1]

plt.xlabel("Height (x)--->")
plt.ylabel("Age (y)--->")
plt.plot([min(x), max(x)], [lin(min(x)), lin(max(x))])
plt.scatter(x,y)
plt.show()
