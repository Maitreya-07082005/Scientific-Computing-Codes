import numpy as np
import sys

def scaled_partial_pivoting(A, b):
    n = len(b)
    s = np.max(np.abs(A), axis=1)

    if np.any(s == 0):
        sys.exit("Error: Zero row detected. System may be singular.")

    for k in range(n - 1):
        ratios = np.abs(A[k:n, k]) / s[k:n]
        pivot_index = np.argmax(ratios) + k

        if A[pivot_index, k] == 0:
            sys.exit("Error: Singular matrix detected. No unique solution.")

        # Swap rows
        if pivot_index != k:
            A[[k, pivot_index]] = A[[pivot_index, k]]
            b[[k, pivot_index]] = b[[pivot_index, k]]
            s[[k, pivot_index]] = s[[pivot_index, k]]

        # Elimination
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    if A[n - 1, n - 1] == 0:
        sys.exit("Error: No unique solution exists.")

    return A, b


def back_substitution(A, b):
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


def gaussian_elimination_solver():
    n = int(input("Enter number of variables: "))

    A = np.zeros((n, n))
    b = np.zeros(n)

    print("\nEnter coefficients row-wise:")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i+1}][{j+1}] = "))

    print("\nEnter constants:")
    for i in range(n):
        b[i] = float(input(f"b[{i+1}] = "))

    A, b = scaled_partial_pivoting(A, b)
    solution = back_substitution(A, b)

    print("\nSolution:")
    for i in range(n):
        print(f"x{i+1} = {solution[i]:.6f}")


if __name__ == "__main__":
    gaussian_elimination_solver()
