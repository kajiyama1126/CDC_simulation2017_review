import numpy as np

b = np.array([1,2,3,4,5])
# A = np.array([[1 for i in range(5)] for j in range(100)])
c = np.ones(20)
d = np.reshape(c,(20,-1))
A = np.kron(d,np.identity(5))

print(A)
# print(d)
print(np.dot(A,b))