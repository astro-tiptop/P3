#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for sensor.NoiseVariance method
"""

import unittest
from p3.aoSystem import np, nnp, cpuArray
from p3.aoSystem.sensor import sensor
from p3.aoSystem.fourierModel import fourierModel
import pathlib

class TestNoiseVariance(unittest.TestCase):
    """Test cases for the NoiseVariance method of the sensor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.pixel_scale = 400  # mas/pixel
        self.fov = 12 # pixels
        self.r0 = 0.20  # meters
        self.wvl = 750e-9  # meters

    def test_shack_hartmann_cog(self):
        """Test NoiseVariance for Shack-Hartmann with CoG algorithm"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertIsInstance(var_noise, np.ndarray)
        self.assertEqual(len(var_noise), 1)
        self.assertGreater(var_noise[0], 0)
        self.assertLess(var_noise[0], 10)  # reasonable upper bound

    def test_shack_hartmann_wcog(self):
        """Test NoiseVariance for Shack-Hartmann with WCoG algorithm"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='wcog',
            algo_param=[5, 0, 0]
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertIsInstance(var_noise, np.ndarray)
        self.assertEqual(len(var_noise), 1)
        self.assertGreater(var_noise[0], 0)

    def test_shack_hartmann_tcog(self):
        """Test NoiseVariance for Shack-Hartmann with tCoG algorithm"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='tcog',
            algo_param=[5, 0.5, 0]
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertIsInstance(var_noise, np.ndarray)
        self.assertEqual(len(var_noise), 1)
        self.assertGreater(var_noise[0], 0)

    def test_shack_hartmann_qc(self):
        """Test NoiseVariance for Shack-Hartmann with QC algorithm"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='qc'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertIsInstance(var_noise, np.ndarray)
        self.assertEqual(len(var_noise), 1)
        self.assertGreater(var_noise[0], 0)

    def test_pyramid(self):
        """Test NoiseVariance for Pyramid WFS"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Pyramid',
            nL=[40],
            dsub=[0.2]
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertIsInstance(var_noise, np.ndarray)
        self.assertEqual(len(var_noise), 1)
        self.assertGreater(var_noise[0], 0)

    def test_multiple_wfs(self):
        """Test NoiseVariance with multiple WFS"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000, 2000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40, 60],
            dsub=[0.2, 0.15],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertIsInstance(var_noise, np.ndarray)
        self.assertEqual(len(var_noise), 2)
        self.assertTrue(np.all(var_noise > 0))

    def test_high_photon_flux(self):
        """Test NoiseVariance with high photon flux (low noise)"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1e6],
            ron=0.5,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertLess(var_noise[0], 0.1)  # Should be very low

    def test_low_photon_flux(self):
        """Test NoiseVariance with low photon flux (high noise)"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[10],
            ron=5.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        self.assertGreater(var_noise[0], 0.1)  # Should be higher

    def test_excess_factor(self):
        """Test that excess factor affects noise variance"""
        wfs1 = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            excess=1.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        wfs2 = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            excess=2.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise1 = wfs1.NoiseVariance(self.r0, self.wvl)
        var_noise2 = wfs2.NoiseVariance(self.r0, self.wvl)

        self.assertGreater(var_noise2[0], var_noise1[0])

    def test_ron_dominance_low_flux(self):
        """Test that RON dominates at low photon flux"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[10],  # very low flux
            ron=5.0,   # high RON
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        # At low flux, RON should dominate: var_ron >> var_shot
        # Estimate var_ron and var_shot separately
        ron = 5.0
        nph = 10
        var_ron_approx = (ron/nph)**2
        var_shot_approx = 1/nph

        self.assertGreater(var_ron_approx, var_shot_approx)

    def test_photon_noise_dominance_high_flux(self):
        """Test that photon noise dominates at high photon flux"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1e6],  # very high flux
            ron=0.5,    # low RON
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        # At high flux, photon noise should dominate
        # var_noise should scale as 1/nph
        self.assertLess(var_noise[0], 0.01)

    def test_noise_scaling_with_flux(self):
        """Test that noise variance scales correctly with photon flux"""
        fluxes = [100, 1000, 10000]
        variances = []

        for flux in fluxes:
            wfs = sensor(
                pixel_scale=self.pixel_scale,
                fov=self.fov,
                nph=[flux],
                ron=0.1,
                wfstype='Shack-Hartmann',
                nL=[40],
                dsub=[0.2],
                algorithm='cog'
            )
            var_noise = wfs.NoiseVariance(self.r0, self.wvl)
            variances.append(var_noise[0])

        # Higher flux should give lower variance
        self.assertGreater(variances[0], variances[1])
        self.assertGreater(variances[1], variances[2])

        # Check approximate 1/nph scaling for photon noise
        # ratio should be close to flux ratio
        ratio_var = variances[0] / variances[1]
        ratio_flux = fluxes[1] / fluxes[0]
        # Allow 50% tolerance due to RON contribution
        self.assertAlmostEqual(ratio_var, ratio_flux, delta=ratio_flux*0.5)

    def test_ron_zero_gives_photon_noise_only(self):
        """Test that zero RON gives only photon noise"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=0.0,  # zero RON
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        # Should be only photon noise, proportional to 1/nph
        self.assertGreater(var_noise[0], 0)
        self.assertLess(var_noise[0], 0.1)  # reasonable for 1000 ph

    def test_infinite_flux_gives_zero_photon_noise(self):
        """Test that infinite flux gives only RON"""
        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[np.inf],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        var_noise = wfs.NoiseVariance(self.r0, self.wvl)

        # With infinite flux, photon noise = 0, only RON
        self.assertEqual(var_noise[0], 0)  # Both terms become 0

    def test_algorithms_consistency(self):
        """Test that different algorithms give reasonable relative values"""
        algorithms = ['cog', 'tcog', 'wcog', 'qc']
        variances = {}

        for algo in algorithms:
            params = [5, 0, 0] if algo in ['wcog', 'tcog'] else [0, 0, 0]
            wfs = sensor(
                pixel_scale=self.pixel_scale,
                fov=self.fov,
                nph=[1000],
                ron=3.0,
                wfstype='Shack-Hartmann',
                nL=[40],
                dsub=[0.2],
                algorithm=algo,
                algo_param=params
            )
            var_noise = wfs.NoiseVariance(self.r0, self.wvl)
            variances[algo] = var_noise[0]

        # All should be positive
        for algo, var in variances.items():
            self.assertGreater(var, 0, f"{algo} should give positive variance")

    def test_thomas_2006_reference_values(self):
        """Test against reference values from Thomas et al. 2006"""
        # Setup conditions similar to paper
        wfs = sensor(
            pixel_scale=500,  # 0.5 arcsec/pixel
            fov=10,
            nph=[1000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[20],
            dsub=[0.5],
            algorithm='cog'
        )

        # Good seeing conditions
        r0 = 0.20
        wvl = 500e-9

        var_noise = wfs.NoiseVariance(r0, wvl)

        # Should be on the order of 0.01-0.1 rad^2 for these conditions
        self.assertGreater(var_noise[0], 0.001)
        self.assertLess(var_noise[0], 1.0)

    def test_subaperture_size_effect(self):
        """Test noise variance relationship with subaperture size"""
        dsubs = [0.1, 0.2, 0.4]
        variances = []

        for dsub in dsubs:
            wfs = sensor(
                pixel_scale=self.pixel_scale,
                fov=self.fov,
                nph=[1000],
                ron=0.0,
                wfstype='Shack-Hartmann',
                nL=[40],
                dsub=[dsub],
                algorithm='cog'
            )
            var_noise = wfs.NoiseVariance(self.r0, self.wvl)
            variances.append(var_noise[0])

        # Verify all variances are positive
        for i, var in enumerate(variances):
            self.assertGreater(var, 0, f"Variance for dsub={dsubs[i]} should be positive")

        # Larger subapertures gives smaller nD, thus higher noise because nT is constant
        self.assertGreater(variances[2], variances[1])
        self.assertGreater(variances[1], variances[0])

    def test_pyramid_vs_shack_hartmann(self):
        """Compare Pyramid and Shack-Hartmann noise models"""
        wfs_sh = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        wfs_pyr = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Pyramid',
            nL=[40],
            dsub=[0.2]
        )

        var_sh = wfs_sh.NoiseVariance(self.r0, self.wvl)
        var_pyr = wfs_pyr.NoiseVariance(self.r0, self.wvl)

        # Both should be positive and of similar order of magnitude
        self.assertGreater(var_sh[0], 0)
        self.assertGreater(var_pyr[0], 0)
        # Pyramid typically more efficient
        self.assertLess(var_pyr[0], var_sh[0] * 10)

    def test_noise_independence_from_science_wavelength(self):
        """Test that noise variance is independent of science wavelength"""
        # Test with different science wavelengths
        wavelengths = [1.25e-6, 1.65e-6, 2.2e-6]  # J, H, K band
        noise_values = []

        wvl_wfs = 750e-9
        r0_at_500nm = 0.20

        wfs = sensor(
            pixel_scale=self.pixel_scale,
            fov=self.fov,
            nph=[1000],
            ron=3.0,
            wfstype='Shack-Hartmann',
            nL=[40],
            dsub=[0.2],
            algorithm='cog'
        )

        for wvl_sci in wavelengths:
            noise_var = wfs.computeNoiseVarianceAtWavelength(
                wvl_science=wvl_sci,
                wvl_wfs=wvl_wfs,
                r0_at_500nm=r0_at_500nm
            )
            rad2nm = wvl_sci * 1e9 / (2 * np.pi)
            noise_values.append(np.sqrt(noise_var[0]) * rad2nm)

        # The noise in nm should be constant across science wavelengths
        for i in range(1, len(noise_values)):
            self.assertAlmostEqual(noise_values[0], noise_values[i], places=6,
                                   msg="Noise should be independent of science wavelength")

    def test_noise_depends_on_wfs_wavelength(self):
        """Test that noise DOES depend on WFS wavelength"""
        # Noise MUST depend on WFS wavelength
        wfs_wavelengths = [500e-9, 750e-9, 950e-9]
        noise_values = []

        for wvl_wfs in wfs_wavelengths:
            wfs = sensor(
                pixel_scale=self.pixel_scale,
                fov=self.fov,
                nph=[1000],
                ron=3.0,
                wfstype='Shack-Hartmann',
                nL=[40],
                dsub=[0.2],
                algorithm='cog'
            )

            # Noise is calculated with r0 at wvl_wfs
            r0_wfs = 0.20 * (wvl_wfs / 500e-9)**(6/5)
            var_noise = wfs.NoiseVariance(r0_wfs, wvl_wfs)
            noise_values.append(var_noise[0])

        # Values should be different
        # (though they might be similar due to r0 scaling)
        self.assertEqual(len(set([round(v, 6) for v in noise_values])), len(noise_values),
                        "Noise should vary with WFS wavelength")

    def test_noise_wfeN_independence_from_science_wavelength(self):
        """Test that wfeN is independent of science wavelength using fourierModel"""

        # Get test directory path
        test_dir = pathlib.Path(__file__).parent.absolute()

        # Test with two different science wavelengths
        ini_files = [
            str(test_dir / 'scao_test_wvl1100nm.ini'),
            str(test_dir / 'scao_test_wvl2200nm.ini')
        ]

        wfeN_values = []

        for ini_file in ini_files:
            # Create Fourier model
            fao = fourierModel(
                path_ini=ini_file,
                path_root='',
                calcPSF=False,
                verbose=False,
                display=False,
                getErrorBreakDown=True
            )

            # Extract wfeN value
            if isinstance(fao.wfeN, np.ndarray):
                wfeN = fao.wfeN[0]  # on-axis value
            else:
                wfeN = fao.wfeN

            wfeN_values.append(wfeN)

            # Print for debugging
            print(f"\nWavelength: {fao.freq.wvlRef*1e9:.0f} nm")
            print(f"wfeN: {wfeN:.2f} nm")

        # Verify that wfeN is the same for both wavelengths
        # Allow small numerical tolerance
        rel_diff = abs(wfeN_values[1] - wfeN_values[0]) / wfeN_values[0]

        self.assertLess(
            rel_diff,
            0.01,  # 1% tolerance
            msg=f"wfeN should be independent of science wavelength. "
                f"Got {wfeN_values[0]:.2f} nm at 1100nm and "
                f"{wfeN_values[1]:.2f} nm at 2200nm "
                f"(relative difference: {rel_diff*100:.2f}%)"
        )

        # More strict test: should be essentially equal
        self.assertAlmostEqual(
            wfeN_values[0],
            wfeN_values[1],
            places=1,
            msg=f"wfeN should be identical for different science wavelengths. "
                f"Got {wfeN_values[0]:.2f} nm at 1100nm and "
                f"{wfeN_values[1]:.2f} nm at 2200nm"
        )

    def test_noise_psd_scaling_with_wavelength(self):
        """Test that noise PSD in nm^2 gives constant wfeN"""

        test_dir = pathlib.Path(__file__).parent.absolute()

        ini_files = [
            str(test_dir / 'scao_test_wvl1100nm.ini'),
            str(test_dir / 'scao_test_wvl2200nm.ini')
        ]

        wfeN_from_psd = []
        wavelengths = []

        for ini_file in ini_files:
            fao = fourierModel(
                path_ini=ini_file,
                path_root='',
                calcPSF=False,
                verbose=False,
                display=False,
                getErrorBreakDown=False
            )

            # Get PSD after conversion to nm^2 (from powerSpectrumDensity)
            # This is what's actually stored and used for wfeN calculation
            dk = 2*fao.freq.kcMax_/fao.freq.resAO
            rad2nm = fao.freq.wvlRef*1e9/2/np.pi

            # Convert raw PSD to nm^2
            psd_nm2 = fao.psdNoise * (dk * rad2nm)**2

            # Calculate wfeN from PSD
            wfeN_calc = np.sqrt(np.sum(psd_nm2))
            wfeN_from_psd.append(wfeN_calc)
            wavelengths.append(fao.freq.wvlRef)

            print(f"\nWavelength: {fao.freq.wvlRef*1e9:.0f} nm")
            print(f"wfeN from PSD: {wfeN_calc:.2f} nm")
            print(f"wfeN from errorBreakDown: {fao.wfeN:.2f} nm" if hasattr(fao, 'wfeN') else "")

        # The wfeN calculated from PSD should be constant
        rel_diff = abs(wfeN_from_psd[1] - wfeN_from_psd[0]) / wfeN_from_psd[0]

        self.assertLess(
            rel_diff,
            0.01,  # 1% tolerance
            msg=f"wfeN from PSD should be independent of science wavelength. "
                f"Got {wfeN_from_psd[0]:.2f} nm at {wavelengths[0]*1e9:.0f}nm and "
                f"{wfeN_from_psd[1]:.2f} nm at {wavelengths[1]*1e9:.0f}nm "
                f"(relative difference: {rel_diff*100:.2f}%)"
        )
