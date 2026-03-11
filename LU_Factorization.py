import numpy as np

def checkLUcondition(A):
    rows, columns = A.shape

    if rows != columns:
        print("The matrix must be square")
        return False

    for i in range(rows):
        minor = A[:i+1, :i+1]
        detMinor = np.linalg.det(minor)

        if abs(detMinor) < 1e-10:
            print(f"Leading principal minor of order {i+1} is zero")
            print("Not LU decomposable without pivoting")
            return False

    print("LU decomposable without pivoting")
    return True


def LU(A):
    if not checkLUcondition(A):
        return None, None

    n = A.shape[0]

    L = np.eye(n)
    U = A.copy()

    for k in range(n-1):
        for i in range(k+1, n):
            L[i][k] = U[i][k] / U[k][k]

            for j in range(k, n):
                U[i][j] = U[i][j] - L[i][k] * U[k][j]

    return L, U


# -------- USER INPUT PART --------

n = int(input("Enter the number of dimensions (n): "))

A = np.zeros((n, n), dtype=float)

print("Enter the elements of the matrix one by one:")

for i in range(n):
    for j in range(n):
        A[i][j] = float(input(f"A[{i+1}][{j+1}] = "))

L, U = LU(A)

if L is not None:
    print("\nMatrix A:")
    print(A)

    print("\nLower triangular matrix L:")
    print(L)

    print("\nUpper triangular matrix U:")
    print(U)

    print("\nVerification (L @ U):")
    print(L @ U)
