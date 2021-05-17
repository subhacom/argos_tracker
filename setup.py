# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-04 4:13 PM
"""How to build distribution:

- Build wheel and tar (update MICRO-version number for test - once used, a filename cannot be reused)

      python -m build

- Check dist folder for old archives - delete them, rename files if required. Then upoload

      python -m twine upload --repository testpypi dist/*


- Install from TestPyPI

      pip install --index-url https://test.pypi.org/simple/ --no-deps argos-tracker --extra-index-url https://pypi.org/simple

This requires the argos-toolkit package.

``argos-tracker`` is PyPI name, the installed module is named
``argos_track``.
"""

from setuptools import setup, find_packages, Extension
import numpy


with open('README.md', 'r') as fh:
    long_description = fh.read()



setup(
    name='argos_tracker',
    version='0.1.0-11',
    author='Subhasis Ray',
    author_email='ray.subhasis@gmail.com',
    description='Software utility for tracking multiple objects (animals) in a video.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/subhacom/argos_tracker',
    project_urls={
    'Documentation': 'https://argos.readthedocs.io',
    'Source': 'https://github.com/subhacom/argos_tracker',
    'Tracker': 'https://github.com/subhacom/argos/issues',
    },
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
        'tables',
        'sortedcontainers',
        'pyqt5',
        'opencv-contrib-python',
        'pyyaml',
        'matplotlib',
        'argos_toolkit'
    ],
)
