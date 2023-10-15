import numpy as np
from sklearn import datasets as ds
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

#Building data using SkLearn
n = 200 #int(input("Enter the no. of data points to generate with:"))
X, y = ds.make_blobs(n, 2, centers=2, cluster_std=1.05) #, random_state=2)

#Initialising centroids
mu1 = np.array(X[0]) #np.array([5,-5])
mu2 = np.array(X[1]) #np.array([-5,5])

Epsilon = 10
z = []

sum1 = np.array([0,0])
sum2 = np.array([0,0])

count1 = 0
count2 = 0

for j in range(10):
	z = []
	for i in range(n):
		if np.transpose(mu1-X[i]).dot(mu1-X[i])<Epsilon:
			z.append(0)
			sum1 = sum1 + X[i]
			count1 = count1 + 1
		else: #elif np.transpose(mu2-X[i]).dot(mu2-X[i])<Epsilon:
			z.append(1)
			sum2 = sum2 + X[i]
			count2 = count2 + 1
	mu1 = sum1/count1
	mu2 = sum2/count2
	sum1 = np.array([0,0])
	sum2 = np.array([0,0])

plt.scatter(X[:,0],X[:,1], c = z)
plt.show()
