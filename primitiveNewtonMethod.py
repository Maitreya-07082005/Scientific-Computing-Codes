import autograd.numpy as np
from autograd import hessian,grad


def f(x):
 return x[0]**2 + ( x[1]**2 / 4 )


def modifiedNewtonMethod(x):
  xk = np.array(x,dtype=float)

  for k in range(1000):
 
   pk =  -np.linalg.inv(hessian(f)(xk)) @ grad(f)(xk) 
   alphak = 1
   x_new = xk + alphak * pk
   if np.linalg.norm(x_new - xk) < 1e-10:
            print(f"Number of iterations: {k}")
            return x_new
   
   xk = x_new
 
  return xk 
 
 
 
x0 = np.array([10.0,15.0])
print("The minimizer is : ",modifiedNewtonMethod(x0))
