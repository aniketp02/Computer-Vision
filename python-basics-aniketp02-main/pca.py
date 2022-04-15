import sys
import numpy as np

# taking the input from terminal
file = open(sys.argv[1], 'r')

# storing the values in an array
arr = []
for data in file:
    a = []
    for num in data.split(','):
        a.append(float(num))
    arr.append(a)

arr = np.array(arr)

rows, cols = arr.shape

# standerdizing the data
arr = (arr - np.mean(arr, axis=0))/np.std(arr, axis=0)

# calculating the covariance
cov = np.cov(arr.T) / rows

# performing eigen decomposition on covariance matrix 
# to get eigenvalues and eigenvectors
values, vec = np.linalg.eig(cov)

# getting the indices of the sorted eigenvalue array
# and using them to sort eigenvectors
idx = np.argsort(values)[::-1]

values = values[idx]
vec = vec[:, idx]

# projection matrix 2D
proj_arr = (vec.T[:][:2]).T

# getting the 2D arr
arr_pca = arr.dot(proj_arr)

# plotting the data
import matplotlib.pyplot as plt

x = 2.5*(arr_pca[:,0])
y = 10*(arr_pca[:,1])

plt.scatter(x, y)
plt.savefig('data/out.png')