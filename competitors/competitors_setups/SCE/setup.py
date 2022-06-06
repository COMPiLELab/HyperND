from setuptools import setup
from setuptools import find_packages

setup(name='sce',
      version='0.1',
      url='https://github.com/szzhang17/Sparsest-Cut-Network-Embedding',
      packages=['sce'],
      scripts=['sce/train_jl.py'],
      license='Proprietary License',
      include_package_data=True,
      zip_safe=False)