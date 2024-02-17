from setuptools import setup
from version import __version__ as version


setup(name='parallel_mc_battery',
      version=version,
      description='Helper class to orchestrate in\
          parallel Monte Carlo simulations for an arbitrary number of models, with low-level parameter granularity.',
      author='Vladimir Rotariu',
      url='https://github.com/vladimirrotariu/parallel-monte-carlo-simulations',
      license='Apache 2.0',
      install_requires=['apache_beam', 'numpy', 'pydantic'],
      classifiers=['Programming Language :: Python :: 3'],
    )
