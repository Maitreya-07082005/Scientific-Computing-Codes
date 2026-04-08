import autograd.numpy as np
from autograd import hessian,grad


def f(x):
 return x[0]**2 + ( x[1]**2 / 4 ) - 1

def argmin(xk,pk):
 
  alpha = 0.01 # random value for alpha
 
  for i in range(100):
   x_new = xk + alpha * pk
  
   phi_prime_alpha = np.dot(grad(f)(x_new) , pk)
 
   phi_double_prime_alpha = np.dot(pk , np.dot(hessian(f)(x_new),pk))
   
   if abs(phi_double_prime_alpha) < 1e-12:
            return alpha 
  
   alpha_new = alpha - phi_prime_alpha / phi_double_prime_alpha
  
   if (abs(alpha-alpha_new)/max(1,alpha)) < 1e-6:
    return alpha_new
  
   alpha = alpha_new
  
  return alpha
  


def modifiedNewtonMethod(x):
  xk = np.array(x,dtype=float)

  for k in range(100):
 
   pk =  -np.linalg.inv(hessian(f)(xk)) @ grad(f)(xk) 
   alphak = argmin(xk,pk)
   x_new = xk + alphak * pk
   if(np.linalg.norm(x - x_new) / max(1, np.linalg.norm(x))) < 1e-6:
    return x_new
   
   xk = x_new
 
  return xk 
 
 
 
x0 = np.array([1.0,1.0])
print("The minimizer is : ",modifiedNewtonMethod(x0))
