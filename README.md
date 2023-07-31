# P3 - fourier PSF

P3 is a collection of libraries for Adaptive Optics (AO) Point Spread Function (PSF) modeling, simulation, fitting and reconstruction.

The list of libraries is:

* [aoSystem](https://github.com/astro-tiptop/P3/tree/main/p3/aoSystem) --> The main purpose of this library is to model an AO PSF. It is an analytical simulator based on a Fourier appoach.
Supported wavefront sensors are Shack-Hartmann and Pyramid.
Several kind of AO are supported: single conjugate AO (SCAO), single laser AO (SLAO), laser tomography AO (LTAO), ground layer AO (GLAO), multi conjugate AO (MCAO), but only one kind of guide star can be used (natural or laser).
In [TIPTOP](https://github.com/astro-tiptop/TIPTOP) it is used to compute the so-called high order PSF.
Examples and tests can be found in [p3/aoSystem/testing](https://github.com/astro-tiptop/P3/tree/main/p3/aoSystem/testing).
Examples of parameters file can be found in [p3/aoSystem/parFiles](https://github.com/astro-tiptop/P3/tree/main/p3/aoSystem/parFiles).
* [deepLoop](https://github.com/astro-tiptop/P3/tree/main/p3/deepLoop)
* [psfFitting](https://github.com/astro-tiptop/P3/tree/main/p3/psfFitting) --> This library can be used to fit parameters on a PSF model (aoSystem/fourierModel or psfao21).
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
* [psfao21](https://github.com/astro-tiptop/P3/tree/main/p3/psfao21) --> This library can be used to model with a small number of parameters the AO PSF. Its main purpose is PSF fitting.
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
* [psfr](https://github.com/astro-tiptop/P3/tree/main/p3/psfr) --> The main purpose of this library is to reconstruct PSF from telemetry data or from image fitting.
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
* [telemetry](https://github.com/astro-tiptop/P3/tree/main/p3/telemetry) --> This library is used to manage telemetry data for PSFR.
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
