#!/usr/bin/env python

from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "WCSAC is designed to work with Python 3.6 and greater. " \
    + "Please install it before proceeding."

setup(
    name='wc_sac',
    packages=['wc_sac'],
    install_requires=[
        'gym~=0.15.3',
        'joblib==0.14.0',
        'matplotlib==3.1.1',
        'mpi4py==3.0.2',
        'mujoco_py==2.0.2.10',
        'numpy~=1.17.4',
        'seaborn==0.8.1',
        'tensorflow==1.13.1',
    ],
)
