import numpy as np
A = np.array([[1, 2], [3, 4]])
print(A.shape)

B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))

print('---------')

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

B = np.array([[1,2], [3,4], [5,6]])
print(B.shape)
print(np.dot(A, B))

print('---------')

C = np.array([[1,2], [3,4]])
#print(C.shape)
#print(A.shape)
#print(np.dot(A, C)) ValuError

print('---------')

A = np.array([[1, 2], [4, 5], [5, 6]])
print(A.shape)

B = np.array([7,8])
print(B.shape)
print(np.dot(A, B))

print('---------')

X = np.array([1, 2])
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
W.shape
Y = np.dot(X, W)
print(Y)
