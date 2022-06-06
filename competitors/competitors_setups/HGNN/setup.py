from setuptools import setup
from setuptools import find_packages

setup(name='hgnn',
      version='0.1',
      url='https://github.com/iMoonLab/HGNN',
      packages=['hgnn'],
      scripts=['hgnn/train_jl.py'],
      license='Proprietary License',
      include_package_data=True,
      zip_safe=False)