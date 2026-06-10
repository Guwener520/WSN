from setuptools import setup, find_packages

setup(
    name="wsn",
    version="0.1.0",
    description="WSN Coverage Optimization using Improved Swarm Intelligence Algorithms",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "plotly>=5.14",
        "scipy>=1.10",
        "imageio>=2.31",
    ],
)
