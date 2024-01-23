import numpy as np
from scipy.optimize import minimize

def objective_function(x):
    return x[0]**2 + x[1]**2 - 2*x[0] + 1

def constraint1(x):
    return x[0] + x[1] - 1

def constraint2(x):
    return 2*x[0] - x[1] - 3

def lagrangian(x, lambda1, lambda2):
    return objective_function(x) + lambda1 * constraint1(x) + lambda2 * constraint2(x)

def augmented_lagrangian(x, lambda1, lambda2, rho):
    return lagrangian(x, lambda1, lambda2) + (rho/2) * (max(0, constraint1(x))**2 + max(0, constraint2(x))**2)

def alm_gradient_descent(initial_x, rho, max_iterations=1000, tolerance=1e-6):
    x = initial_x
    lambda1 = 0.0
    lambda2 = 0.0

    for iteration in range(max_iterations):
        result = minimize(lambda x: augmented_lagrangian(x, lambda1, lambda2, rho), x, method='BFGS')

        x = result.x
        lambda1 = max(0, lambda1 + rho * constraint1(x))
        lambda2 = max(0, lambda2 + rho * constraint2(x))

        if np.linalg.norm(result.jac) < tolerance:
            break

    return x

initial_guess = np.array([0.0, 0.0])
rho = 1.0
result = alm_gradient_descent(initial_guess, rho)

print("Optimal solution:", result)
print("Objective value at optimal solution:", objective_function(result))
print("Constraint 1 value at optimal solution:", constraint1(result))
print("Constraint 2 value at optimal solution:", constraint2(result))