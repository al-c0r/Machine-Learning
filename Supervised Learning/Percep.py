import numpy as np
from sklearn import datasets as ds
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

#Building data using SkLearn
n = 200 #int(input("Enter the no. of data points to generate with:"))
X, y = ds.make_blobs(n, 2, centers=2, cluster_std=1.05, random_state=2)

'''
print(X, y)
plt.scatter(X[:,0],X[:,1], marker="*", c=y)
plt.show()
'''

#Defining a step-function
def h(x):
	if x<0:
		return 0
	else:
		return 1

#Initialising Variables
w = np.zeros(2)	 #Weight vector
l = 0.01 #float(input("Please enter the learning rate (Choose value less 1 for better accuracy):"))	 #Learning rate
b = 0	 #Bias

#Algorithm
for i in range(int(7*n/10)):
	y1 = h(w.dot(X[i,:]) + b)
	w = w + l*(y1 - y[i])
	b = b + 5.5*l*(y1 - y[i])

'''Equation of Hyperplane:
w(trans.)x+b = 0
w[0]*x + w[1]*y + b = 0
y = -(b + w[0]*x)/w[1]'''

print("Weight vector is", w, "Bias is", b)
def y_pred(x):
	return -(b + w[0]*x)/w[1]

#Testing
Acc = 0
for i in range(int(7*n/10),n):
	y2 = h(w.dot(X[i,:]) + b)
	if y2==y[i]:
		Acc = Acc + 1
print("Accuracy is", Acc*1000/(3*n), "%")

plt.scatter(X[:,0],X[:,1], c=y)
plt.plot([min(X[:,0]), max(X[:,0])], [y_pred(min(X[:,0])), y_pred(max(X[:,0]))])
plt.show()
