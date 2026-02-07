"""
Setup script for MG-MOTRv2
"""

from setuptools import setup, find_packages

setup(
    name="mg-motrv2",
    version="0.1.0",
    description="Multi-Granularity Multi-Object Tracking with Transformer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
)
