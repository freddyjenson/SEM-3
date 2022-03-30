import numpy as np

m1 = np.array([[2,3,6],[9,5,6]])
m2 = np.array([[5,6,3],[8,5,3]])
m3 = np.array([[5,3],[6,2],[5,8]])

print("Matrix 1:\n",m1)
print("Matrix 2: \n",m2)
print("\nAddition:")
print(np.add(m1,m2))

print("\nSubtraction:")
print(np.subtract(m1,m2))

print("\nMultiplication:")
print(np.multiply(m1,m2))

print("\nSquare root of :\n", m1)
print(np.sqrt(m1))

print("\nSummation of elements:")
print(sum(m1))

print("\nColumn wise summation:")
print(np.sum(m1,axis=0))

print("\nRow wise summation:")
print(np.sum(m1,axis=1))

print("\nProduct of matrixes: ")
print("1: \n",m1)
print("2: \n",m3)
print("Product:\n",np.dot(m1,m3))

print("\nTranspose of : \n",m3)
print("Transpose: ")
print(m3.T)