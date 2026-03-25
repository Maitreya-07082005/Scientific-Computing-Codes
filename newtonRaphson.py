import autograd.numpy as np
from autograd import grad, hessian

def f(x):
    return x[0]**2 + x[1]**4

def findAlpha(x, p, tol=1e-10):
    alpha = 0.1
    for i in range(1000):
        x_new = x + alpha * p
       
        phi_prime = np.dot(grad(f)(x_new), p)
        phi_double_prime = np.dot(p, np.dot(hessian(f)(x_new), p))
        
        if abs(phi_double_prime) < 1e-14:
            break
            
        alpha_new = alpha - phi_prime / phi_double_prime
        
        if (abs(alpha - alpha_new) / (max(1, alpha))) < tol:
            return alpha_new
            
        alpha = alpha_new
    return alpha
    
    
def steepestDescentNewtonRaphson(x_init, max_iter=1000, tol=1e-10):
    x = np.array(x_init, dtype=float)
    
    for k in range(max_iter):
        
        pk = -grad(f)(x)
       
        alpha = findAlpha(x, pk, tol)
        
        x_new = x + alpha * pk
        
        if (np.linalg.norm(x - x_new) / max(1, np.linalg.norm(x))) < tol:
            return x_new 
            
        x = x_new
    return x
    
start = np.array([1.0, 1.0])
result = steepestDescentNewtonRaphson(start)
print("The solution is:", result)

