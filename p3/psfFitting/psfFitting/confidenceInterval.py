import numpy as np
from numpy.linalg import inv

def confidence_interval(fvec, jac):
    """Returns the 95% confidence interval on parameters from
    non-linear fit results."""
    # residual sum of squares
    rss = np.sum(fvec**2)
    # number of data points and parameters
    n, p = jac.shape
    # the statistical degrees of freedom
    nmp = n - p
    # mean residual error
    ssq = rss / nmp
    # the Jacobian
    J = np.matrix(jac)
    # covariance matrix
    c = inv(J.T*J)
    # variance-covariance matrix.
    pcov = c * ssq
    # Diagonal terms provide error estimate based on uncorrelated parameters.
    # The sqrt convert from variance to std. dev. units.
    err = np.sqrt(np.diag(np.abs(pcov))) * 1.96  # std. dev. x 1.96 -> 95% conf
    # Here err is the full 95% area under the normal distribution curve. This
    # means that the plus-minus error is the half of this value
    return err