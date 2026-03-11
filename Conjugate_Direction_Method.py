import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------

n = int(input("Enter dimension n: "))

print("Enter matrix A row by row (space separated):")
A = []
for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

A = np.array(A)

print("Enter vector b:")
b = np.array(list(map(float, input().split())))

print("Enter starting point x0:")
x0 = np.array(list(map(float, input().split())))

# ------------------------------------------------------------
# Function and gradient
# ------------------------------------------------------------

def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x

def grad(x):
    return A @ x - b


# ------------------------------------------------------------
# Steepest Descent Algorithm
# ------------------------------------------------------------

def steepest_descent(x0, max_iter=100, tol=1e-8):

    x = x0.copy()

    path = [x.copy()]
    losses = [f(x)]

    for k in range(max_iter):

        g = grad(x)

        if np.linalg.norm(g) < tol:
            break

        alpha = (g.T @ g) / (g.T @ A @ g)

        x = x - alpha * g

        path.append(x.copy())
        losses.append(f(x))

    return np.array(path), np.array(losses)


# ------------------------------------------------------------
# Run algorithm
# ------------------------------------------------------------

path, losses = steepest_descent(x0)

print("\nApproximate solution:", path[-1])

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(12,5))

# ------------------------------------------------------------
# CONTOUR + PATH (only if n=2)
# ------------------------------------------------------------

if n == 2:

    xmin = min(path[:,0]) - 1
    xmax = max(path[:,0]) + 1
    ymin = min(path[:,1]) - 1
    ymax = max(path[:,1]) + 1

    x_vals = np.linspace(xmin, xmax, 300)
    y_vals = np.linspace(ymin, ymax, 300)

    X, Y = np.meshgrid(x_vals, y_vals)

    Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2) - b[0]*X - b[1]*Y

    ax[0].contour(X, Y, Z, 30)

    ax[0].plot(path[:,0], path[:,1], 'ro-', linewidth=2)

    # True minimum
    x_star = np.linalg.solve(A, b)
    ax[0].plot(x_star[0], x_star[1], 'g*', markersize=15)

    ax[0].set_title("Steepest Descent Path")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")

else:

    ax[0].text(0.5,0.5,"Contour plot only available for n = 2",
               ha='center',va='center')
    ax[0].set_title("Contour Plot")

# ------------------------------------------------------------
# LOSS CONVERGENCE
# ------------------------------------------------------------

ax[1].plot(losses, 'bo-')

ax[1].set_title("Loss Convergence")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("f(x)")

plt.tight_layout()
plt.show()
