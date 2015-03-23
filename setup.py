from setuptools import setup, find_packages
import numpy as np
import os.path as p


include_dirs = [np.get_include()]

requirements = ['menpo>=0.4.4',
                'opencv>=2.4.8']

setup(name='templatetracker',
      version='0.0.1',
      description='Correlation Filter Based Tracker',
      author='Joan Alabort-i-Medina',
      author_email='joan.alabort@gmail.com',
      include_dirs=include_dirs,
      packages=find_packages(),
      install_requires=requirements,
      scripts=[p.join('templatetracker', 'templatetracker')])
