import numpy as np

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

m = int(input("Enter number of rows (m): "))
n = int(input("Enter number of columns (n): "))

print("Enter matrix A row by row:")

A = []
for i in range(m):
    row = list(map(float, input().split()))
    A.append(row)

A = np.array(A)

mode = input("Choose QR type ('reduced' or 'full'): ").lower()

# ------------------------------------------------------------
# REDUCED QR USING GRAM-SCHMIDT
# ------------------------------------------------------------

Q = np.zeros((m, n))
R = np.zeros((n, n))

for j in range(n):

    v = A[:, j].copy()

    for i in range(j):
        R[i, j] = np.dot(Q[:, i], A[:, j])
        v = v - R[i, j] * Q[:, i]

    R[j, j] = np.linalg.norm(v)

    if R[j, j] == 0:
        print("Matrix has linearly dependent columns.")
        exit()

    Q[:, j] = v / R[j, j]

# ------------------------------------------------------------
# SELECT OUTPUT TYPE
# ------------------------------------------------------------

if mode == "reduced":

    Q_final = Q
    R_final = R

elif mode == "full":

    # Extend Q to m x m
    Q_full = np.zeros((m, m))
    Q_full[:, :n] = Q

    for j in range(n, m):

        v = np.random.rand(m)

        for i in range(j):
            v = v - np.dot(Q_full[:, i], v) * Q_full[:, i]

        v = v / np.linalg.norm(v)
        Q_full[:, j] = v

    R_full = np.zeros((m, n))
    R_full[:n, :] = R

    Q_final = Q_full
    R_final = R_full

else:
    print("Invalid option. Choose 'reduced' or 'full'.")
    exit()

# ------------------------------------------------------------
# OUTPUT RESULTS
# ------------------------------------------------------------

print("\nMatrix Q:")
print(Q_final)

print("\nMatrix R:")
print(R_final)

# ------------------------------------------------------------
# VERIFY A = QR
# ------------------------------------------------------------

A_reconstructed = Q_final @ R_final

print("\nReconstructed A (Q @ R):")
print(A_reconstructed)

print("\nOriginal A:")
print(A)

# ------------------------------------------------------------
# ORTHOGONALITY CHECK
# ------------------------------------------------------------

print("\nQ^T Q (should be identity):")
print(Q_final.T @ Q_final)

# ------------------------------------------------------------
# RECONSTRUCTION ERROR
# ------------------------------------------------------------

error = np.linalg.norm(A - A_reconstructed)

print("\nReconstruction error ||A - QR||:")
print(error)
