import numpy as np
import scipy
import scipy.linalg 
from math import sqrt

def cubic_spline(x_data, y_data, x_interp):
  """
  Performs cubic spline interpolation for a given set of data points.

  Args:
      x_data: Array of data points (x-coordinates).
      y_data: Array of data points (y-coordinates).
      x_interp: Array of interpolation points (x-coordinates for which to interpolate y-values).

  Returns:
      y_interp: Array of interpolated y-values for the given interpolation points.
  """

  # Number of data points
  n = len(x_data)
 

  # Calculate differences between consecutive x-values
  h = np.diff(x_data)

  # Initialize empty arrays for coefficients
  a = np.zeros(n-1)
  b = np.zeros(n-1)
  c = np.zeros(n-1)
  d = np.zeros(n-1)

  # First derivative at endpoints (natural cubic splines)
  a[0] = 0
  c[n-2] = 0

  # Build the system of linear equations
  for i in range(1, n-2):
    diag = 2*(h[i] + h[i-1])
    off_diag = h[i]
    upper_diag = h[i-1]
    a[i] = diag
    b[i] = off_diag
    c[i] = upper_diag
    d[i] = 3*(y_data[i+1] - y_data[i]) / h[i] - 3*(y_data[i] - y_data[i-1]) / h[i-1]

  # Solve the tridiagonal system for coefficients
    print(np.diag(a).size)
    print(np.diag(b[:-1],-1).size)
    print(np.diag(c[1:], 1).size)
    print(d.size)


    result = scipy.linalg.solve( np.diag(a) + np.diag(b[:-1],-1) + np.diag(c[1:], 1), d)
    c[1:-1] =result

  # Calculate remaining coefficients
  for i in range(1, n-1):
    b[i] = (a[i] * c[i-1] - d[i-1]) / h[i]
    d[i] = (d[i] - b[i] * h[i]) / a[i]

  # Interpolate y-values for given points
  y_interp = []
  for x in x_interp:
    i = np.searchsorted(x_data, x, side='left') - 1
    t = (x - x_data[i]) / h[i]
    y_interp.append(a[i]*t**3 + b[i]*t**2 + c[i]*t + d[i] + y_data[i])

  return np.array(y_interp)

# Example usage

def cubic_interp1d(x0, x, y):
    """
    Interpolate a 1-D function using cubic splines.
      x0 : a float or an 1d-array
      x : (N,) array_like
          A 1-D array of real/complex values.
      y : (N,) array_like
          A 1-D array of real values. The length of y along the
          interpolation axis must be equal to the length of x.

    Implement a trick to generate at first step the cholesky matrice L of
    the tridiagonal matrice A (thus L is a bidiagonal matrice that
    can be solved in two distinct loops).

    additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf 
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # remove non finite values
    # indexes = np.isfinite(x)
    # x = x[indexes]
    # y = y[indexes]

    # check if sorted
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # allocate buffer matrices
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 # natural boundary
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    # solve [L.T][x] = [y]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # find index
    index = x.searchsorted(x0)
    np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0