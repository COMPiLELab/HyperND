from setuptools import setup
from setuptools import find_packages

setup(name='hypergcn',
      version='0.1',
      url='https://github.com/malllabiisc/HyperGCN',
      packages=['hypergcn'],
      scripts=['hypergcn/hypergcn_jl.py'],
      license='Proprietary License',
      include_package_data=True,
      zip_safe=False)