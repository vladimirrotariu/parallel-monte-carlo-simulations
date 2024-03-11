from setuptools import (
    setup,
    find_packages,
)

version = "0.0.1"

setup(
    name="parallel_simulations",
    version=version,
    description="Helper class to orchestrate in parallel Monte Carlo simulations for an arbitrary number of models, with low-level parameter granularity.",
    author="Vladimir Rotariu",
    url="https://github.com/vladimirrotariu/parallel-monte-carlo-simulations",
    license="Apache 2.0",
    install_requires=[
        "apache_beam",
        "numpy",
        "pydantic",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    packages=find_packages(),
)
