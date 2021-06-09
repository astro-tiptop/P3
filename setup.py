"""
At the moment, this would allow the installation of P3 but submodules are installed individually, 
i.e. instead of "import P3 / from P3 install aoSystem", 

You need to do import the relevant submodules:
    e.g. import psfao21/aoSystem ... 
"""

from setuptools import setup, find_packages

setup(name='P3',
      url='https://github.com/oliviermartin-lam/P3.git',
      packages=find_packages())