# Numerical Linear Algebra Algorithms (Python)

A collection of **core numerical linear algebra algorithms implemented in Python**, along with visualization and practical applications such as **image compression using SVD**.

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

### 4. Image Compression using SVD

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

### 5. Steepest Descent Method

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

# 📊 Visualization

Some scripts include visualization such as:

* Contour plots of quadratic functions
* Optimization paths
* Convergence plots
* Image reconstruction comparisons

These visualizations make the algorithms easier to understand.

---

# 🛠 Requirements

Install dependencies:

```bash
pip install numpy matplotlib pillow
```

Used libraries:

* `numpy`
* `matplotlib`
* `Pillow` (for image compression example)

---

# 📁 Structure

```
numerical-linear-algebra-algorithms
│
├── gaussian_elimination.py
├── lu_decomposition.py
├── svd.py
├── svd_image_compression.py
├── steepest_descent.py
│
└── README.md
```

---

# 🎯 Learning Goals

This repository helps demonstrate:

* Matrix factorizations
* Iterative optimization algorithms
* Numerical stability issues
* Visualization of optimization behaviour
* Practical use of linear algebra in applications

---

# 📖 References

Some helpful resources:

* *Numerical Linear Algebra* — Lloyd N. Trefethen & David Bau
* *Matrix Computations* — Golub & Van Loan
* *Numerical Optimization* — Nocedal & Wright

---

# 🤝 Contributions

Contributions are welcome!

Possible improvements:

* Add QR decomposition
* Implement GMRES
* Add PCA examples
* Improve visualization tools
* Add benchmarking comparisons

---

# 📜 License

This project is open-source and available under the **MIT License**.
