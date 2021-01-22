from setuptools import find_packages, setup
from demo_prep import __version__

setup(
    name='demo_prep',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=['wheel'],
    version=__version__,
    description='Databricks Labs CICD Templates Sample Project',
    author='Christopher Chalcraft'
)
