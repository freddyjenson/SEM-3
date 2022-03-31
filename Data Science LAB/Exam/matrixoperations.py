import numpy as np

A = np.array([[1,2,3,6],[8,9,4,6],[2,7,5,9],[3,1,8,4]])

B = np.array([[4,7,9,6],[2,4,6,7],[9,5,8,3],[6,9,2,6]])

print("\nMultiplication of matrices A and B\n",np.multiply(A,B))

print("\nTranspose of matrix A\n",A.T)

C = A.T

print("\nMultiplication of matrices A transpose and B\n",np.multiply(C,B))

print("\nLast two elements of 3rd and 4th row of matrix B are\n",B[2,-2],",",B[2,-1],"and",B[3,-2],",",B[3,-1])