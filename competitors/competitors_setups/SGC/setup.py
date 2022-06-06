from setuptools import setup
from setuptools import find_packages

setup(name='sgc',
      version='0.1',
      url='https://github.com/Tiiiger/SGC',
      packages=['sgc'],
      scripts=['sgc/citation_jl.py'],
      license='Proprietary License',
      include_package_data=True,
      zip_safe=False)