import numpy as np

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

n = int(input("Enter dimension of the matrix (n): "))

print("Enter matrix A row by row:")

A = []
for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

A = np.array(A)

print("Enter vector b:")
b = np.array(list(map(float, input().split())))

print("Enter initial guess x0:")
x = np.array(list(map(float, input().split())))

max_iter = int(input("Enter maximum iterations: "))
tol = float(input("Enter tolerance (e.g. 1e-6): "))

# ------------------------------------------------------------
# CONJUGATE GRADIENT METHOD
# ------------------------------------------------------------

r = b - A @ x
p = r.copy()

print("\nStarting Conjugate Gradient iterations...\n")

for k in range(max_iter):

    Ap = A @ p

    alpha = (r.T @ r) / (p.T @ Ap)

    x_new = x + alpha * p

    r_new = r - alpha * Ap

    print(f"Iteration {k+1}")
    print("x =", x_new)
    print("Residual norm =", np.linalg.norm(r_new))
    print()

    if np.linalg.norm(r_new) < tol:
        print("Converged.")
        x = x_new
        break

    beta = (r_new.T @ r_new) / (r.T @ r)

    p = r_new + beta * p

    x = x_new
    r = r_new

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

print("\nApproximate Solution:")
print(x)

print("\nFinal Residual Norm:")
print(np.linalg.norm(b - A @ x))
