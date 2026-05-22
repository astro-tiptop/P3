import numpy as np
import math
import unittest
from p3.aoSystem.atmosphere import atmosphere, layer

def create_mock_atmosphere():
    """Create a test atmosphere with 3 layers."""
    return atmosphere(
        wvl=500e-9, 
        r0=0.15, 
        weights=[0.5, 0.3, 0.2], 
        heights=[0.0, 5000.0, 10000.0], 
        wSpeed=[5.0, 10.0, 20.0], 
        wDir=[0.0, 45.0, 90.0], 
        L0=25.0
    )

class TestAtmosphere(unittest.TestCase):
    """Regression tests for the atmosphere model."""

    def test_temporal_covariance_weights_bug(self):
        """
        Historical bug test:
        if tau = 0, temporal covariance must match total variance.
        In the old unweighted code, this returned N_layers * variance.
        """
        atm = create_mock_atmosphere()

        # Covariance at zero delay
        cov_zero = atm.temporalCovariance(tau=0.0)

        # Theoretical variance
        var_expected = atm.variance()

        # Verify with strict tolerance
        np.testing.assert_allclose(cov_zero, var_expected, rtol=1e-7)

    def test_angular_covariance_slab_equivalence(self):
        """
        Verify that the vectorized/weighted implementation
        matches the legacy slab-based logic.
        """
        atm = create_mock_atmosphere()
        theta = 10.0 / 206265.0  # 10 arcsec in radians

        # 1) Compute with the current method
        cov_new = atm.angularCovariance(theta)

        # 2) Compute with the slab logic emulated in this test
        cov_old = 0.0
        for l in range(atm.nL):
            atmSlab = atm.slab(l)
            atmSlab.r0 = atm.r0 * (atm.weights[l]) ** (-3.0 / 5.0)
            tmp = atmSlab.covariance(atmSlab.heights[0] * np.tan(theta))
            cov_old += tmp

        np.testing.assert_allclose(cov_new, cov_old, rtol=1e-7)

    def test_theta0_backend_safety(self):
        """
        Verify that theta0 extraction is robust with SciPy/fsolve
        and returns a valid scalar value.
        """
        atm = create_mock_atmosphere()

        # If not robust, this may fail or return a spurious array.
        th0 = atm.theta0

        self.assertIsInstance(th0, float)
        self.assertGreater(th0, 0.0)
        self.assertFalse(math.isinf(th0))


if __name__ == "__main__":
    unittest.main()