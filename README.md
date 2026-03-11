# Numerical Linear Algebra Algorithms (Python)

A collection of **core numerical linear algebra algorithms implemented in Python**, along with visualization and practical applications such as.
This repository is intended for **learning, experimentation, and demonstration of numerical optimization and matrix factorization techniques** commonly used in scientific computing, machine learning, and data science.

---

# 📚 Contents

This repository currently includes implementations of the following algorithms:

### 1. Gaussian Elimination

Implementation of **Gaussian elimination** for solving systems of linear equations.

Features:

* Forward elimination
* Back substitution
* Solves (Ax = b)

Concepts demonstrated:

* Row operations
* Pivoting
* Numerical stability

---

### 2. LU Decomposition

Factorizes a matrix

[
A = LU
]

where:

* (L) = lower triangular matrix
* (U) = upper triangular matrix

Applications:

* Efficient solution of linear systems
* Matrix inversion
* Determinant computation

---

### 3. Singular Value Decomposition (SVD)

Decomposes a matrix into

[
A = U Σ V^T
]

where:

* (U) = left singular vectors
* (Σ) = singular values
* (V) = right singular vectors

Applications:

* Dimensionality reduction
* Principal Component Analysis (PCA)
* Data compression
* Noise reduction

---
### 5. QR Decomposition

QR Decomposition factorizes a matrix \(A\) into the product of two matrices:

\[
A = QR
\]

where:

- \(Q\) is an **orthogonal matrix**  
- \(R\) is an **upper triangular matrix**

An orthogonal matrix satisfies:

\[
Q^T Q = I
\]

which means its columns form an **orthonormal basis**.

QR decomposition is widely used because it is **numerically stable** compared to methods like Gaussian elimination.

---

### 5. Image Compression using SVD

Demonstrates how **low-rank approximations** can compress images.

Idea:

Instead of storing the full image matrix (A), approximate it as

[
A_k = U_k Σ_k V_k^T
]

where (k) is a small rank.

Benefits:

* Significant reduction in storage
* Controlled loss of information
* Visual demonstration of compression

The repository includes scripts that show:

* original image
* reconstructed images with different ranks
* compression trade-offs
* 
---

### 6. Steepest Descent Method

Optimization algorithm used to minimize quadratic functions:

[
f(x) = (1/2)x^T A x - b^T x
]

Features in the repository:

* Automatic step size for quadratic functions
* Visualization of the optimization path on contour plots
* Loss convergence plots

This helps illustrate the classic **zig-zag behaviour of steepest descent**.

---
### 7. Conjugate Gradient Method
Direct methods like Gaussian Elimination require \(O(n^3)\) operations and large memory usage.  
The **Conjugate Gradient method** is an **iterative algorithm** that solves systems more efficiently when:

- The matrix is **large**
- The matrix is **sparse**
- The matrix is **symmetric positive definite**

The algorithm improves upon **Steepest Descent** by generating **conjugate search directions**, allowing faster convergence
We want to solve

\[
Ax = b
\]

which is equivalent to minimizing the quadratic function

\[
f(x) = \frac{1}{2}x^T A x - b^T x
\]

The Conjugate Gradient method iteratively updates:

- Residual vector \(r_k\)
- Search direction \(p_k\)
- Solution vector \(x_k\)

Key formulas used:

alpha_k = (r_kᵀ r_k) / (p_kᵀ A p_k)

x_(k+1) = x_k + alpha_k p_k

r_{k+1} = r_k - alpha_k A p_k

beta_k = (r_(k+1)ᵀ r_(k+1)) / (r_kᵀ r_k)

p_(k+1) = r_(k+1) + beta_k p_k

1. Choose initial guess \(x_0\)
2. Compute initial residual:

\[
r_0 = b - Ax_0
\]

3. Set search direction:

\[
p_0 = r_0
\]

4. For each iteration:

- Compute step size \( \alpha_k \)
- Update solution \( x_{k+1} \)
- Update residual \( r_{k+1} \)
- Compute \( \beta_k \)
- Update search direction \( p_{k+1} \)

5. Stop when residual norm is small.

For an n*n matrix, the Conjugate Gradient method converges in **at most \(n\) iterations** in exact arithmetic.

In practice it often converges much faster.

---
# 🎯 Learning Goals

This repository helps demonstrate:

* Matrix factorizations
* Iterative optimization algorithms
* Numerical stability issues
* Visualization of optimization behaviour
* Practical use of linear algebra in applications
* And the most important goal - To have all linear algebra algorithms in a single repo!

---

# 📖 References

Some helpful resources:

* *Numerical Linear Algebra* — Lloyd N. Trefethen & David Bau
* *Matrix Computations* — Golub & Van Loan
* *Numerical Optimization* — Nocedal & Wright

---

# 🤝 Contributions

Contributions are welcome!

# 📜 License

This project is open-source and available under the **MIT License**.
