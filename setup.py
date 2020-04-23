from setuptools import setup, find_packages
import numpy as np


setup(name = 'ds_hdp_hmm',
      description = 'Disentangled Sticky HDP-HMM',
      install_requires = ['numpy', 'scipy', 'matplotlib', 'munkres', 'seaborn', 'numba'],
      packages = find_packages('ds_hdp_hmm'),
      #package_dir = {'' : 'ds_hdp_hmm'}
      )
