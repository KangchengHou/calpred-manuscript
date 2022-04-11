from distutils.version import LooseVersion
from io import open

import setuptools
from setuptools import setup

# Added support for environment markers in install_requires.
if LooseVersion(setuptools.__version__) < "36.2":
    raise ImportError("setuptools>=36.2 is required")

setup(
    name="calprs",
    version="0.1",
    description="Calibrated PRS",
    author="Kangcheng Hou, Ziqi Xu",
    author_email="kangchenghou@gmail.com",
    packages=["calprs"],
    setup_requires=["numpy>=1.10"],
    entry_points={"console_scripts": ["calprs=calprs._cli:cli"]},
)
