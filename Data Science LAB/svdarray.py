from numpy import array
from scipy.linalg import svd

ar = array([[3, 2, 4], [7, 3, 5]])
print(ar)

a, b, c = svd(ar)
print(a)
print(b)
print(c)
