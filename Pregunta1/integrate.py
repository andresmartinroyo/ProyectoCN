import numpy as np
from scipy.integrate import quad

def integrand(x):
    return np.sqrt(np.tan(np.sqrt(x)))

a = 0  # Lower limit of integration
b = 3.5  # Upper limit of integration

# Approximate the integral using the trapezoidal method
integral_approx, _ = quad(integrand, a, b)

print("Approximate integral value:", integral_approx)