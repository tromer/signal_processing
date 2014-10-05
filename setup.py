#!/usr/bin/python
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='signal_processing',
      version='0.1',
      description='natural interface for signal processing of signals and pulses',
      url='https://github.com/noamg/signal_processing',
      author='Noam Gavish',
      author_email='gavishnoam@gmail.com',
      license='_______',
      packages=['signal_processing'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pint',
      ],
      zip_safe=False)
