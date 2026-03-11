import numpy as np



n = int(input("enter the size of the square matrix :"))



A = np.zeros((n,n))



print("Enter row wise elements:")



for i in range(n):

 for j in range(n):

  A[i][j] = float(input())



print(A)



ATA = A.T @ A

eigenvalues,eigenvectors = np.linalg.eig(ATA)

singular_values = np.sqrt(np.abs(eigenvalues))

idx = np.argsort(-singular_values)

singular_values = singular_values[idx]

eigenvectors = eigenvectors[:,idx]



U = np.zeros((n,n))



for i in range(n):

 if singular_values[i] !=0:

    U[:,i] = (A @ eigenvectors[:,i])/singular_values[i]



sigma = np.zeros((n,n))

for i in range(n):

    sigma[i][i] = singular_values[i]





print("\n U = ")

print(U)



print("\n Sigma = ")

print(sigma)



print("\n V^T =")

print(eigenvectors.T)
