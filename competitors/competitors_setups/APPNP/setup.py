from setuptools import setup
from setuptools import find_packages

setup(name='appnp',
      version='0.1',
      url='https://github.com/benedekrozemberczki/APPNP',
      packages=['appnp'],
      scripts=['appnp/main_jl.py'],
      license='Proprietary License',
      include_package_data=True,
      zip_safe=False)